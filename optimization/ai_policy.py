import math
import numpy as np
from stable_baselines3 import PPO
from core.entities import RouteStop

class AIPolicyOptimizer:
    def __init__(self, model_path="models/fleet_ppo_v2_500k.zip"):
        try:
            self.model = PPO.load(model_path)
            print("Successfully loaded PPO model for Streamlit.")
        except Exception as e:
            self.model = None
            print(f"Could not load PPO model: {e}")

    def optimize(self, vehicles: dict, shipments: dict, time_step: int):
        if not self.model:
            return

        pending_shipments = [s for s in shipments.values() if s.status == "pending"]
        if not pending_shipments:
            return

        v_keys = list(vehicles.keys())

        # Iterate through the queue so we don't get stuck on one shipment
        for target_shipment in pending_shipments:
            
            # 1. Construct the 42-dim observation
            obs = []
            obs.extend([
                target_shipment.pickup[0] / 100.0, target_shipment.pickup[1] / 100.0,
                target_shipment.destination[0] / 100.0, target_shipment.destination[1] / 100.0,
                target_shipment.weight / 3.0, target_shipment.priority / 2.0
            ])
            
            v_count = 0
            for vid, v in vehicles.items():
                if v_count >= 10: # <-- CHANGE TO 10
                    break
                cap_rem = v.capacity - v.current_payload
                obs.extend([
                    v.position[0] / 100.0, v.position[1] / 100.0,
                    1.0 if v.status == "idle" else 0.0,
                    1.0 if v.status == "broken_down" else 0.0,
                    cap_rem / 7.0, v.current_payload / 7.0
                ])
                v_count += 1
            
            # Pad unused vehicle slots up to 10
            for _ in range(10 - v_count): # <-- CHANGE TO 10
                obs.extend([0.0] * 6)

            obs.append(min(time_step / 500.0, 1.0))
            for i in range(1, 6):
                obs.append(0.0)

            obs_array = np.array(obs, dtype=np.float32)

            # 2. Ask the AI for a decision
            action, _ = self.model.predict(obs_array, deterministic=True)
            assigned = False

            if action < len(v_keys):
                chosen_vehicle_id = v_keys[action]
                vehicle = vehicles[chosen_vehicle_id]
                cap_rem = vehicle.capacity - vehicle.current_payload

                if vehicle.status == "idle" and cap_rem >= target_shipment.weight:
                    # AI made a valid, safe choice!
                    self._assign(vehicle, target_shipment, time_step)
                    assigned = True

            # 3. SAFETY OVERRIDE LAYER
            # If the AI chose a busy/broken vehicle, it causes a deadlock.
            # The safety layer intercepts the failure and forces a valid fallback.
            if not assigned:
                idle_vehicles = [
                    v for v in vehicles.values() 
                    if v.status == "idle" and (v.capacity - v.current_payload) >= target_shipment.weight
                ]
                
                if idle_vehicles:
                    # Reroute to the closest available idle vehicle
                    best_vehicle = min(
                        idle_vehicles, 
                        key=lambda v: math.hypot(v.position[0] - target_shipment.pickup[0], v.position[1] - target_shipment.pickup[1])
                    )
                    self._assign(best_vehicle, target_shipment, time_step)
                    assigned = True

            # Only assign ONE shipment per tick to allow the environment state to update
            if assigned:
                break

    def _assign(self, vehicle, target_shipment, time_step):
        """Helper method to cleanly execute the assignment."""
        target_shipment.status = "assigned"
        target_shipment.assigned_vehicle_id = vehicle.id
        target_shipment.assigned_tick = time_step
        
        vehicle.assigned_shipments.append(target_shipment.id)
        vehicle.status = "en_route"
        vehicle.route = [
            RouteStop(location=target_shipment.pickup, shipment_id=target_shipment.id, action="pickup"),
            RouteStop(location=target_shipment.destination, shipment_id=target_shipment.id, action="delivery")
        ]