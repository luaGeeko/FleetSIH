import math
from typing import Dict
from ortools.linear_solver import pywraplp
from core.entities import Vehicle, Shipment, RouteStop

class ORToolsOptimizer:
    
    def optimize(self, vehicles: Dict[str, Vehicle], shipments: Dict[str, Shipment], current_tick: int):
        pending_shipments = [s for s in shipments.values() if s.status == "pending"]
        available_vehicles = [v for v in vehicles.values() if v.status == "idle"]

        if not pending_shipments or not available_vehicles:
            return

        print(f"\n[Tick {current_tick}] OR-Tools evaluating {len(pending_shipments)} shipments against {len(available_vehicles)} vehicles...")

        costs = []
        for v in available_vehicles:
            vehicle_costs = []
            for s in pending_shipments:
                if s.weight > v.capacity:
                    vehicle_costs.append(999999) 
                else:
                    dist = math.hypot(v.position[0] - s.pickup[0], v.position[1] - s.pickup[1])
                    vehicle_costs.append(dist)
            costs.append(vehicle_costs)

        num_vehicles = len(available_vehicles)
        num_shipments = len(pending_shipments)

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print("[ERROR] OR-Tools CBC Solver could not be initialized.")
            return

        x = {}
        for i in range(num_vehicles):
            for j in range(num_shipments):
                x[i, j] = solver.IntVar(0, 1, '')

        # Constraints
        for i in range(num_vehicles):
            solver.Add(solver.Sum([x[i, j] for j in range(num_shipments)]) <= 1)

        for j in range(num_shipments):
            solver.Add(solver.Sum([x[i, j] for i in range(num_vehicles)]) <= 1)

        # THE FIX: Force the solver to assign the maximum possible number of shipments
        max_assignments = min(num_vehicles, num_shipments)
        solver.Add(solver.Sum([x[i, j] for i in range(num_vehicles) for j in range(num_shipments)]) == max_assignments)

        # Objective
        objective_terms = []
        for i in range(num_vehicles):
            for j in range(num_shipments):
                objective_terms.append(costs[i][j] * x[i, j])
        solver.Minimize(solver.Sum(objective_terms))

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            assigned_count = 0
            for i in range(num_vehicles):
                for j in range(num_shipments):
                    if x[i, j].solution_value() > 0.5:
                        vehicle = available_vehicles[i]
                        shipment = pending_shipments[j]

                        shipment.status = "assigned"
                        shipment.assigned_vehicle_id = vehicle.id
                        shipment.assigned_tick = current_tick
                        
                        vehicle.assigned_shipments.append(shipment.id)
                        vehicle.status = "en_route"
                        
                        vehicle.route = [
                            RouteStop(location=shipment.pickup, shipment_id=shipment.id, action="pickup"),
                            RouteStop(location=shipment.destination, shipment_id=shipment.id, action="delivery")
                        ]
                        assigned_count += 1
            print(f" -> Success: Assigned {assigned_count} optimal routes.")
        else:
            print(f" -> Failed: Solver returned status code {status}")