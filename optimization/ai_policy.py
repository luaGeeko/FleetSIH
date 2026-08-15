import os
import math
import numpy as np
from stable_baselines3 import PPO
from core.entities import RouteStop

class AIPolicyOptimizer:
    # REMOVED the default argument. Now model_path is strictly required.
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.obs_shape = 72  # Safe fallback

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Error: Model file '{model_path}' does not exist!")

        try:
            self.model = PPO.load(self.model_path)
            self.obs_shape = self.model.observation_space.shape[0]
            version_label = "V1 (84-dim, Advanced Logistics)" if self.obs_shape == 84 else "V0 (72-dim, Baseline)"
            print(f"✅ Loaded Policy: {self.model_path} [{version_label}]")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load PPO policy from '{model_path}': {e}")

    def optimize(self, vehicles: dict, shipments: dict, time_step: int, state=None):
        if not self.model:
            return

        pending_shipments = [s for s in shipments.values() if s.status == "pending"]
        if not pending_shipments:
            return

        v_keys = list(vehicles.keys())
        is_v1 = (self.obs_shape == 84)

        if is_v1:
            pending_shipments.sort(key=lambda s: (-s.priority, s.deadline))

        for target_shipment in pending_shipments:
            obs = []
            
            obs.extend([
                target_shipment.pickup[0] / 100.0, target_shipment.pickup[1] / 100.0,
                target_shipment.destination[0] / 100.0, target_shipment.destination[1] / 100.0,
                target_shipment.weight / 3.0, target_shipment.priority / 2.0
            ])
            if is_v1:
                deadline_rem = (target_shipment.deadline - time_step) / 200.0
                obs.append(np.clip(deadline_rem, -2.0, 2.0))
            
            v_count = 0
            for vid in v_keys:
                if v_count >= 10: 
                    break
                v = vehicles[vid]
                cap_rem = v.capacity - v.current_payload
                obs.extend([
                    v.position[0] / 100.0, v.position[1] / 100.0,
                    1.0 if v.status == "idle" else 0.0,
                    1.0 if v.status == "broken_down" else 0.0,
                    cap_rem / 7.0, v.current_payload / 7.0
                ])
                if is_v1:
                    dist_to_pickup = math.hypot(
                        v.position[0] - target_shipment.pickup[0], 
                        v.position[1] - target_shipment.pickup[1]
                    ) / 141.4
                    obs.append(dist_to_pickup)
                v_count += 1
            
            for _ in range(10 - v_count):
                obs.extend([0.0] * (7 if is_v1 else 6))

            obs.append(min(time_step / 500.0, 1.0))
            for i in range(1, 6):
                severity = state.traffic_zones.get(f"Zone_{i}", 0.0) if state else 0.0
                obs.append(float(severity))

            if is_v1:
                active_shipments = [s for s in shipments.values() if s.status in ["pending", "assigned"]]
                if active_shipments:
                    slacks = [s.deadline - time_step for s in active_shipments]
                    worst_slack = np.clip(min(slacks) / 200.0, -2.0, 2.0)
                else:
                    worst_slack = 1.0
                obs.append(worst_slack)

            obs_array = np.array(obs, dtype=np.float32)

            action, _ = self.model.predict(obs_array, deterministic=True)
            assigned = False

            if action < len(v_keys):
                chosen_vehicle_id = v_keys[action]
                vehicle = vehicles[chosen_vehicle_id]
                cap_rem = vehicle.capacity - vehicle.current_payload

                if vehicle.status == "idle" and cap_rem >= target_shipment.weight:
                    self._assign(vehicle, target_shipment, time_step)
                    assigned = True

            if not assigned:
                idle_vehicles = [
                    v for v in vehicles.values() 
                    if v.status == "idle" and (v.capacity - v.current_payload) >= target_shipment.weight
                ]
                if idle_vehicles:
                    best_vehicle = min(
                        idle_vehicles, 
                        key=lambda v: math.hypot(
                            v.position[0] - target_shipment.pickup[0], 
                            v.position[1] - target_shipment.pickup[1]
                        )
                    )
                    self._assign(best_vehicle, target_shipment, time_step)
                    assigned = True

            if assigned:
                break

    def _assign(self, vehicle, target_shipment, time_step):
        target_shipment.status = "assigned"
        target_shipment.assigned_vehicle_id = vehicle.id
        target_shipment.assigned_tick = time_step
        
        vehicle.assigned_shipments.append(target_shipment.id)
        vehicle.status = "en_route"
        vehicle.route = [
            RouteStop(location=target_shipment.pickup, shipment_id=target_shipment.id, action="pickup"),
            RouteStop(location=target_shipment.destination, shipment_id=target_shipment.id, action="delivery")
        ]