import gymnasium as gym
import numpy as np
import math
from core.simulator import FleetSimulator
from core.entities import RouteStop, Vehicle, Shipment

class FleetEnvV1(gym.Env):
    """
    V1: Advanced Logistics Formulation.
    Incorporates time-window urgency, pickup-distance awareness, and separates 
    service-level penalties (lateness, unserved shipments) into an explicit cost signal.
    """
    def __init__(self, max_vehicles=10):
        super(FleetEnvV1, self).__init__()
        self.simulator = FleetSimulator()
        self.max_vehicles = max_vehicles

        # Action Space: 10 Possible Vehicle Slots
        self.action_space = gym.spaces.Discrete(self.max_vehicles)

        # Observation Space: 84 Dimensions Total
        # Target (7) + Fleet (10 * 7 = 70) + Env (7) = 84
        # Range extended below 0 to account for negative slack (late shipments)
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(84,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.simulator.vehicles.clear()
        self.simulator.shipments.clear()
        self.simulator.state.time_step = 0
        self.simulator.state.active_alerts.clear()
        self.simulator.state.traffic_zones.clear()
        
        # Track cumulative episode cost for training log aggregation
        self.episode_cost = 0.0
        
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
        cost = 0.0  # V1: Separated constraint signal for lateness and unserved penalty
        terminated = False
        
        pending_shipments = [s for s in self.simulator.shipments.values() if s.status == "pending"]
        
        if pending_shipments:
            # FIX: Priority + Urgency ordering instead of arbitrary first-pending
            # Sorts by highest priority first (descending), then earliest deadline (ascending)
            pending_shipments.sort(key=lambda s: (-s.priority, s.deadline))
            target_shipment = pending_shipments[0]
            
            v_keys = list(self.simulator.vehicles.keys())
            
            # Check valid vehicle selection
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
                    cost += 60.0  # Severe operational constraint violation
            else:
                reward -= 10.0 # Ghost vehicle slot
                cost += 60.0   # Severe operational constraint violation
        
        old_positions = {vid: v.position for vid, v in self.simulator.vehicles.items()}

        # V1 Disruptions
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
        
        # Operational distance penalty (Efficiency)
        dist_this_step = 0.0
        for vid, vehicle in self.simulator.vehicles.items():
            old_p = old_positions[vid]
            dist_this_step += math.hypot(vehicle.position[0] - old_p[0], vehicle.position[1] - old_p[1])
        reward -= (dist_this_step * 0.02)

        # Evaluate completed deliveries
        newly_delivered = [s for s in delivered_after if s.id not in delivered_before]
        for s in newly_delivered:
            reward += 10.0 
            if s.delivery_tick <= s.deadline:
                reward += 5.0 # On-time efficiency bonus
            else:
                # V1: Lateness is logged as a cost constraint, NOT a reward penalty
                lateness = s.delivery_tick - s.deadline
                cost += lateness

        # FIX: Idle-vehicle penalty
        idle_count = sum(1 for v in self.simulator.vehicles.values() if v.status == "idle")
        reward -= (idle_count * 0.01)

        # Termination checks
        if len(delivered_after) == len(self.simulator.shipments):
            terminated = True
            
        if self.simulator.state.time_step > 500: 
            terminated = True
            # V1: Assess heavy cost for unfulfilled demand at horizon end
            unserved_count = sum(1 for s in self.simulator.shipments.values() if s.status != "delivered")
            cost += (unserved_count * 60.0)

        # FIX: Track episode cost for training aggregation
        self.episode_cost += cost

        info = {
            "step_cost": cost,
            "episode_cost": self.episode_cost
        }

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        obs = []
        
        pending_shipments = [s for s in self.simulator.shipments.values() if s.status == "pending"]
        target_pos = None

        # 1. TARGET SHIPMENT (7 dims)
        if pending_shipments:
            # Re-sort for the observation just in case
            pending_shipments.sort(key=lambda s: (-s.priority, s.deadline))
            ts = pending_shipments[0]
            target_pos = ts.pickup
            deadline_rem = (ts.deadline - self.simulator.state.time_step) / 200.0 # V1 Urgency scaling
            
            obs.extend([
                ts.pickup[0] / 100.0, ts.pickup[1] / 100.0,
                ts.destination[0] / 100.0, ts.destination[1] / 100.0,
                ts.weight / 3.0, 
                ts.priority / 2.0,
                np.clip(deadline_rem, -2.0, 2.0) # V1 Urgency representation
            ])
        else:
            obs.extend([0.0]*7)
            
        # 2. FLEET STATE (70 dims)
        v_count = 0
        for vid, v in self.simulator.vehicles.items():
            if v_count >= self.max_vehicles:
                break
            cap_rem = v.capacity - v.current_payload
            
            # FIX: Spatial awareness - distance to the current pickup
            dist_to_pickup = 0.0
            if target_pos:
                dist_to_pickup = math.hypot(v.position[0] - target_pos[0], v.position[1] - target_pos[1]) / 141.4
                
            obs.extend([
                v.position[0] / 100.0, v.position[1] / 100.0,
                1.0 if v.status == "idle" else 0.0,
                1.0 if v.status == "broken_down" else 0.0,
                cap_rem / 7.0, 
                v.current_payload / 7.0,
                dist_to_pickup # V1: Pickup-distance awareness
            ])
            v_count += 1
            
        # Zero-pad ghost vehicles
        for _ in range(self.max_vehicles - v_count):
            obs.extend([0.0] * 7)

        # 3. ENVIRONMENT STATE (7 dims)
        obs.append(min(self.simulator.state.time_step / 500.0, 1.0))
        
        # Traffic zones
        for i in range(1, 6):
            severity = self.simulator.state.traffic_zones.get(f"Zone_{i}", 0.0)
            obs.append(float(severity))

        # V1: Fleet-wide worst slack indicator
        active_shipments = [s for s in self.simulator.shipments.values() if s.status in ["pending", "assigned"]]
        if active_shipments:
            slacks = [s.deadline - self.simulator.state.time_step for s in active_shipments]
            worst_slack = np.clip(min(slacks) / 200.0, -2.0, 2.0)
        else:
            worst_slack = 1.0
        obs.append(worst_slack)

        return np.array(obs, dtype=np.float32)