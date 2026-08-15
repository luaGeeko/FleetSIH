import math
from typing import Dict
from core.entities import Vehicle, Shipment, RouteStop

class GreedyBaselineOptimizer:
    
    def optimize(self, vehicles: Dict[str, Vehicle], shipments: Dict[str, Shipment], current_tick: int):
        pending_shipments = [s for s in shipments.values() if s.status == "pending"]
        if not pending_shipments:
            return

        available_vehicles = [v for v in vehicles.values() if v.status == "idle"]

        for shipment in pending_shipments:
            if not available_vehicles:
                break
                
            best_vehicle = None
            best_dist = float('inf')
            
            for vehicle in available_vehicles:
                # Capacity Check
                if shipment.weight > vehicle.capacity:
                    continue
                    
                dx = vehicle.position[0] - shipment.pickup[0]
                dy = vehicle.position[1] - shipment.pickup[1]
                dist = math.hypot(dx, dy)
                
                if dist < best_dist:
                    best_dist = dist
                    best_vehicle = vehicle
            
            if best_vehicle:
                shipment.status = "assigned"
                shipment.assigned_vehicle_id = best_vehicle.id
                shipment.assigned_tick = current_tick
                
                best_vehicle.assigned_shipments.append(shipment.id)
                best_vehicle.status = "en_route"
                
                best_vehicle.route = [
                    RouteStop(location=shipment.pickup, shipment_id=shipment.id, action="pickup"),
                    RouteStop(location=shipment.destination, shipment_id=shipment.id, action="delivery")
                ]
                available_vehicles.remove(best_vehicle)