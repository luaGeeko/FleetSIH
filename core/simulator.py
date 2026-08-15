import math
from typing import Dict
from core.entities import Vehicle, Shipment, EnvironmentState, RouteStop
from optimization.baseline import GreedyBaselineOptimizer
from optimization.ortools_solver import ORToolsOptimizer
from optimization.ai_policy import AIPolicyOptimizer

class FleetSimulator:
    def __init__(self):
        self.vehicles: Dict[str, Vehicle] = {}
        self.shipments: Dict[str, Shipment] = {}
        self.state = EnvironmentState()

        # baseline solvers 
        self.greedy_optimizer = GreedyBaselineOptimizer()
        self.ortools_optimizer = ORToolsOptimizer()
        
        # FIXED: Initialized as None. No arguments required, preventing any accidental default loading!
        self.ai_optimizer = None
        self.current_strategy = "Greedy Dispatch" # Default

    def set_strategy(self, strategy_name: str):
        self.current_strategy = strategy_name

    def set_ai_optimizer(self, optimizer: AIPolicyOptimizer):
        """Injects a pre-loaded AI optimizer into the simulator."""
        self.ai_optimizer = optimizer

    def generate_deterministic_scenario(self):
        """Creates Scenario 1: A locked 0-100 grid for reproducible baselines."""
        self.vehicles.clear()
        self.shipments.clear()
        self.state = EnvironmentState()

        coords = [(10.0, 10.0), (80.0, 20.0), (40.0, 50.0), (20.0, 80.0), (90.0, 90.0)]
        for i, pos in enumerate(coords):
            vid = f"V{i+1:02d}"
            self.vehicles[vid] = Vehicle(id=vid, position=pos)

        self.shipments = {
            "S101": Shipment(id="S101", pickup=(15.0, 15.0), destination=(90.0, 90.0)),
            "S102": Shipment(id="S102", pickup=(85.0, 25.0), destination=(10.0, 85.0)),
            "S103": Shipment(id="S103", pickup=(45.0, 55.0), destination=(5.0, 5.0))
        }

    def step(self):
        self.state.time_step += 1

        if self.current_strategy == "OR-Tools CVRP":
            self.ortools_optimizer.optimize(self.vehicles, self.shipments, self.state.time_step)
        elif self.current_strategy == "AI Coordinator":
            if self.ai_optimizer:
                self.ai_optimizer.optimize(self.vehicles, self.shipments, self.state.time_step, self.state)
        else:
            self.greedy_optimizer.optimize(self.vehicles, self.shipments, self.state.time_step)

        # Physical movement
        for vid, vehicle in self.vehicles.items():
            if vehicle.status == "broken_down" or not vehicle.route:
                continue
                
            vehicle.time_active += 1 
            
            target_stop = vehicle.route[0]
            dx = target_stop.location[0] - vehicle.position[0]
            dy = target_stop.location[1] - vehicle.position[1]
            distance = math.hypot(dx, dy)
            
            actual_speed = vehicle.speed
            
            if distance <= actual_speed:
                vehicle.position = target_stop.location
                vehicle.total_distance += distance
                shipment = self.shipments[target_stop.shipment_id]
                
                if target_stop.action == "pickup":
                    shipment.status = "picked_up"
                    shipment.pickup_tick = self.state.time_step
                    vehicle.current_payload += shipment.weight
                    
                elif target_stop.action == "delivery":
                    shipment.status = "delivered"
                    shipment.delivery_tick = self.state.time_step
                    vehicle.current_payload -= shipment.weight
                    if shipment.id in vehicle.assigned_shipments:
                        vehicle.assigned_shipments.remove(shipment.id)
                        
                vehicle.route.pop(0)
                if not vehicle.route:
                    vehicle.status = "idle"
            else:
                step_x = (dx / distance) * actual_speed
                step_y = (dy / distance) * actual_speed
                vehicle.position = (vehicle.position[0] + step_x, vehicle.position[1] + step_y)
                vehicle.total_distance += actual_speed

    def inject_breakdown(self, target_vid: str):
        if target_vid in self.vehicles:
            vehicle = self.vehicles[target_vid]
            vehicle.status = "broken_down"
            vehicle.route.clear()
            
            for sid in vehicle.assigned_shipments:
                if self.shipments[sid].status != "delivered":
                    self.shipments[sid].status = "pending"
                    self.shipments[sid].assigned_vehicle_id = None
            
            vehicle.assigned_shipments.clear()
            vehicle.current_payload = 0
            self.state.active_alerts.append(f"CRITICAL: {target_vid} breakdown at {vehicle.position}")

    def inject_traffic(self, zone_id: str, severity: float = 0.5):
        self.state.traffic_zones[zone_id] = severity
        self.state.active_alerts.append(f"TRAFFIC WARNING: Zone {zone_id} speed reduced by {int((1-severity)*100)}%")

    def inject_demand_surge(self, num_new=5):
        start_idx = len(self.shipments) + 101
        for i in range(num_new):
            sid = f"S{start_idx+i}"
            p_loc = (80.0 + (i*2), 80.0 + (i*2))
            d_loc = (20.0, 20.0)
            self.shipments[sid] = Shipment(id=sid, pickup=p_loc, destination=d_loc)
        self.state.active_alerts.append(f"DEMAND SURGE: +{num_new} shipments added.")

    def get_metrics(self):
        total_v = len(self.vehicles)
        broken_v = sum(1 for v in self.vehicles.values() if v.status == "broken_down")
        moving_v = sum(1 for v in self.vehicles.values() if v.status == "en_route")
        
        if self.state.time_step > 0:
            avg_utilization = (sum(v.time_active for v in self.vehicles.values()) / (total_v * self.state.time_step)) * 100
        else:
            avg_utilization = 0.0
            
        total_dist = sum(v.total_distance for v in self.vehicles.values())

        total_s = max(1, len(self.shipments))
        delivered_shipments = [s for s in self.shipments.values() if s.status == "delivered"]
        pending_s = sum(1 for s in self.shipments.values() if s.status == "pending")
        
        completion_rate = (len(delivered_shipments) / total_s) * 100
        on_time_count = sum(1 for s in delivered_shipments if s.delivery_tick <= s.deadline)
        on_time_rate = (on_time_count / max(1, len(delivered_shipments))) * 100
        
        avg_delivery_time = (sum(s.delivery_tick - s.created_tick for s in delivered_shipments) / max(1, len(delivered_shipments)))

        return {
            "time_step": self.state.time_step,
            "total_vehicles": total_v,
            "available_vehicles": total_v - broken_v,
            "moving_vehicles": moving_v,
            "broken_vehicles": broken_v,
            "utilization": avg_utilization,
            "total_distance": total_dist,
            "pending_shipments": pending_s,
            "delivered_shipments": len(delivered_shipments),
            "completion_rate": completion_rate,
            "on_time_rate": on_time_rate,
            "avg_delivery_time": avg_delivery_time,
            "alerts": self.state.active_alerts[-3:]
        }