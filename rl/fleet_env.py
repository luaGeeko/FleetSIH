import gymnasium as gym
import numpy as np
import math
from core.simulator import FleetSimulator
from core.entities import RouteStop, Vehicle, Shipment

class FleetEnv(gym.Env):
    def __init__(self, max_vehicles=10):
        super(FleetEnv, self).__init__()
        self.simulator = FleetSimulator()
        self.max_vehicles = max_vehicles

        # Action Space: 10 Possible Vehicle Slots
        self.action_space = gym.spaces.Discrete(self.max_vehicles)

        # Observation Space: 72 Dimensions Total
        # Target (6) + Fleet (10 * 6 = 60) + Env (6) = 72
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(72,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.simulator.vehicles.clear()
        self.simulator.shipments.clear()
        self.simulator.state.time_step = 0
        self.simulator.state.active_alerts.clear()
        self.simulator.state.traffic_zones.clear()
        
        # 1. Randomize active fleet size (3 to 10 vehicles)
        num_vehicles = self.np_random.integers(3, 11) 
        for i in range(num_vehicles):
            vid = f"V{i+1:02d}"
            pos = (self.np_random.uniform(0, 100), self.np_random.uniform(0, 100))
            cap = self.np_random.integers(3, 8) 
            self.simulator.vehicles[vid] = Vehicle(id=vid, position=pos, capacity=cap)

        # 2. Randomize heavy workload (5 to 20 shipments)
        num_shipments = self.np_random.integers(5, 21)
        for i in range(num_shipments):
            sid = f"S{i+101}"
            p_loc = (self.np_random.uniform(0, 100), self.np_random.uniform(0, 100))
            d_loc = (self.np_random.uniform(0, 100), self.np_random.uniform(0, 100))
            w = self.np_random.integers(1, 4) 
            prio = self.np_random.integers(1, 3) 
            deadline = self.simulator.state.time_step + self.np_random.integers(50, 200)
            self.simulator.shipments[sid] = Shipment(
                id=sid, pickup=p_loc, destination=d_loc, 
                weight=w, priority=prio, deadline=deadline
            )

        self.simulator.set_strategy("AI") 
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        
        pending_shipments = [s for s in self.simulator.shipments.values() if s.status == "pending"]
        
        if pending_shipments:
            target_shipment = pending_shipments[0]
            v_keys = list(self.simulator.vehicles.keys())
            
            # Did the AI pick a valid, currently existing vehicle?
            if action < len(v_keys):
                target_vehicle_id = v_keys[action]
                vehicle = self.simulator.vehicles[target_vehicle_id]
                capacity_remaining = vehicle.capacity - vehicle.current_payload
                
                if vehicle.status == "idle" and capacity_remaining >= target_shipment.weight:
                    target_shipment.status = "assigned"
                    target_shipment.assigned_vehicle_id = vehicle.id
                    target_shipment.assigned_tick = self.simulator.state.time_step
                    
                    vehicle.assigned_shipments.append(target_shipment.id)
                    vehicle.status = "en_route"
                    vehicle.route = [
                        RouteStop(location=target_shipment.pickup, shipment_id=target_shipment.id, action="pickup"),
                        RouteStop(location=target_shipment.destination, shipment_id=target_shipment.id, action="delivery")
                    ]
                else:
                    reward -= 5.0 # Existing but busy/broken/full
            else:
                reward -= 10.0 # Picked a "ghost" vehicle slot that doesn't exist in this scenario
        
        old_positions = {vid: v.position for vid, v in self.simulator.vehicles.items()}

        if self.np_random.random() < 0.01:
            v_keys = list(self.simulator.vehicles.keys())
            if v_keys:
                random_vid = self.np_random.choice(v_keys)
                if self.simulator.vehicles[random_vid].status != "broken_down":
                    self.simulator.inject_breakdown(random_vid)
                
        if self.np_random.random() < 0.02:
            self.simulator.inject_traffic(f"Zone_{self.np_random.integers(1,6)}", severity=0.3)

        delivered_before = [s.id for s in self.simulator.shipments.values() if s.status == "delivered"]
        self.simulator.step()
        delivered_after = [s for s in self.simulator.shipments.values() if s.status == "delivered"]
        
        dist_this_step = 0.0
        for vid, vehicle in self.simulator.vehicles.items():
            old_p = old_positions[vid]
            dist_this_step += math.hypot(vehicle.position[0] - old_p[0], vehicle.position[1] - old_p[1])
        reward -= (dist_this_step * 0.02)

        newly_delivered = [s for s in delivered_after if s.id not in delivered_before]
        for s in newly_delivered:
            reward += 10.0 
            if s.delivery_tick <= s.deadline:
                reward += 5.0 
            else:
                reward -= 5.0 

        idle_count = sum(1 for v in self.simulator.vehicles.values() if v.status == "idle")
        reward -= (idle_count * 0.01)

        if len(delivered_after) == len(self.simulator.shipments):
            terminated = True
            
        if self.simulator.state.time_step > 500: 
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        obs = []
        
        pending_shipments = [s for s in self.simulator.shipments.values() if s.status == "pending"]
        
        # 1. TARGET SHIPMENT (6 dims)
        if pending_shipments:
            ts = pending_shipments[0]
            obs.extend([
                ts.pickup[0] / 100.0, ts.pickup[1] / 100.0,
                ts.destination[0] / 100.0, ts.destination[1] / 100.0,
                ts.weight / 3.0, 
                ts.priority / 2.0 
            ])
        else:
            obs.extend([0.0]*6)
            
        # 2. FLEET STATE (60 dims)
        v_count = 0
        for vid, v in self.simulator.vehicles.items():
            if v_count >= self.max_vehicles:
                break
            cap_rem = v.capacity - v.current_payload
            obs.extend([
                v.position[0] / 100.0, v.position[1] / 100.0,
                1.0 if v.status == "idle" else 0.0,
                1.0 if v.status == "broken_down" else 0.0,
                cap_rem / 7.0, 
                v.current_payload / 7.0
            ])
            v_count += 1
            
        # Zero-pad the ghost vehicles (forces is_idle=0, cap_rem=0, effectively making them unpickable)
        for _ in range(self.max_vehicles - v_count):
            obs.extend([0.0] * 6)

        # 3. ENVIRONMENT STATE (6 dims)
        obs.append(min(self.simulator.state.time_step / 500.0, 1.0))
        for i in range(1, 6):
            severity = self.simulator.state.traffic_zones.get(f"Zone_{i}", 0.0)
            obs.append(float(severity))

        return np.array(obs, dtype=np.float32)