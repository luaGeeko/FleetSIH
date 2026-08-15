import numpy as np
import pandas as pd
import random
import os
import argparse

from core.simulator import FleetSimulator
from core.entities import Vehicle, Shipment
from optimization.ai_policy import AIPolicyOptimizer

def setup_deterministic_scenario(sim, num_vehicles, seed):
    """Generates the exact same starting state across strategies for a given seed."""
    np.random.seed(seed)
    random.seed(seed)
    
    sim.vehicles.clear()
    sim.shipments.clear()
    sim.state.time_step = 0
    sim.state.active_alerts.clear()
    sim.state.traffic_zones.clear()

    # Spawn vehicles
    for i in range(num_vehicles):
        vid = f"V{i+1:02d}"
        pos = (np.random.uniform(10, 90), np.random.uniform(10, 90))
        sim.vehicles[vid] = Vehicle(id=vid, position=pos, capacity=7)

    # Proportional Workload: 3 shipments per vehicle
    num_shipments = num_vehicles * 3 
    for i in range(num_shipments):
        sid = f"S{i+101}"
        p_loc = (np.random.uniform(5, 95), np.random.uniform(5, 95))
        d_loc = (np.random.uniform(5, 95), np.random.uniform(5, 95))
        sim.shipments[sid] = Shipment(
            id=sid, 
            pickup=p_loc, 
            destination=d_loc, 
            weight=np.random.randint(1, 4), 
            priority=np.random.randint(1, 3), 
            deadline=300
        )

def run_experiment(scenario_name, strategy_name, ai_optimizer, num_vehicles, seed):
    sim = FleetSimulator()
    
    # Inject the pre-loaded model cleanly so the simulator never re-initializes AIPolicyOptimizer()
    if ai_optimizer:
        sim.set_ai_optimizer(ai_optimizer)
        
    setup_deterministic_scenario(sim, num_vehicles=num_vehicles, seed=seed)
    sim.set_strategy(strategy_name)

    disrupted_shipments = {} 
    recovery_times = []      
    
    for tick in range(1, 251):
        # Inject Disruptions
        if scenario_name == "Traffic Spike" and tick == 50:
            sim.inject_traffic("Zone_3", severity=0.8)
        elif scenario_name == "Vehicle Breakdown" and tick == 75:
            if "V01" in sim.vehicles:
                sim.inject_breakdown("V01")
        elif scenario_name == "Demand Surge" and tick == 100:
            sim.inject_demand_surge(num_new=num_vehicles)
        elif scenario_name == "Chaos Day":
            if tick == 40:
                sim.inject_traffic("Zone_2", severity=0.9)
            elif tick == 80 and "V02" in sim.vehicles:
                sim.inject_breakdown("V02")
            elif tick == 120:
                sim.inject_demand_surge(num_new=int(num_vehicles * 1.5))

        # Track Recovery Metrics
        for sid, shipment in sim.shipments.items():
            if shipment.status == "pending" and shipment.assigned_tick is not None:
                if sid not in disrupted_shipments:
                    disrupted_shipments[sid] = tick
            if shipment.status == "assigned" and sid in disrupted_shipments:
                recovery_time = tick - disrupted_shipments[sid]
                recovery_times.append(recovery_time)
                del disrupted_shipments[sid] 

        # Execute Strategy (Simulator step handles AI optimization internally via set_ai_optimizer)
        sim.step()

        if all(s.status == "delivered" for s in sim.shipments.values()):
            break

    base_metrics = sim.get_metrics()
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
    
    total_disruptions = len(recovery_times) + len(disrupted_shipments)
    recovery_rate = (len(recovery_times) / total_disruptions) * 100 if total_disruptions > 0 else 100.0

    # Service-Level Constraint Cost Calculation
    constraint_cost = 0.0
    for s in sim.shipments.values():
        if s.status == "delivered":
            if s.delivery_tick > s.deadline:
                constraint_cost += (s.delivery_tick - s.deadline)
        else:
            constraint_cost += 60.0

    # Determine Model Used and Observation Shape for transparency
    model_used = ai_optimizer.model_path if strategy_name == "AI Coordinator" and ai_optimizer else "N/A"
    obs_shape = ai_optimizer.obs_shape if strategy_name == "AI Coordinator" and ai_optimizer else "N/A"

    return {
        "Seed": seed,
        "Fleet Size": num_vehicles,
        "Scenario": scenario_name,
        "Strategy": strategy_name,
        "Model Used": model_used,
        "Observation Shape": obs_shape,
        "Completion (%)": round(base_metrics["completion_rate"], 1),
        "On-Time (%)": round(base_metrics["on_time_rate"], 1),
        "Constraint Cost": round(constraint_cost, 1), 
        "Distance (km)": round(base_metrics["total_distance"], 1),
        "Utilization (%)": round(base_metrics["utilization"], 1),
        "Recovery Rate (%)": round(recovery_rate, 1),
        "Avg Recovery (Ticks)": round(avg_recovery_time, 1)
    }

def main():
    parser = argparse.ArgumentParser(description="Run Smart Fleet Platform Benchmark Suite")
    parser.add_argument(
        "--version", 
        type=str, 
        choices=["v0", "v1"], 
        default="v1", 
        help="Select version: 'v0' or 'v1'"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        help="Optional: Explicit path to model .zip file"
    )
    args = parser.parse_args()

    # Determine model path
    if args.model_path:
        selected_model_path = args.model_path
    elif args.version == "v1":
        selected_model_path = "models/fleet_ppo_v1_advanced_500k.zip"
    else:
        selected_model_path = "models/fleet_ppo_v0_500k.zip"
        
    print(f"🚀 Initializing Benchmark [{args.version.upper()}]...")
    
    # SINGLE INSTANTIATION: Load the model once here
    ai_opt = AIPolicyOptimizer(model_path=selected_model_path)

    if not ai_opt.model:
        print(f"❌ Aborting: Could not load valid policy from '{selected_model_path}'")
        return

    seeds = [42, 101, 777]
    fleet_sizes = [3, 5, 6, 10]
    scenarios = ["Normal Day", "Traffic Spike", "Vehicle Breakdown", "Demand Surge", "Chaos Day"]
    strategies = ["Greedy Dispatch", "OR-Tools CVRP", "AI Coordinator"]
    
    results = []
    
    for f_size in fleet_sizes:
        print(f"\n🚛 Evaluating Fleet Size: {f_size} (Workload: {f_size * 3} shipments)")
        for seed in seeds:
            for strategy in strategies:
                baseline_distance = None
                
                for scenario in scenarios:
                    res = run_experiment(scenario, strategy, ai_opt, f_size, seed)
                    
                    if scenario == "Normal Day":
                        baseline_distance = res["Distance (km)"]
                        res["Adaptation Cost (%)"] = 0.0
                    else:
                        if baseline_distance and baseline_distance > 0:
                            cost = ((res["Distance (km)"] - baseline_distance) / baseline_distance) * 100.0
                            res["Adaptation Cost (%)"] = round(cost, 1)
                        else:
                            res["Adaptation Cost (%)"] = 0.0
                            
                    results.append(res)
            
    df = pd.DataFrame(results)
    os.makedirs("evaluation/results", exist_ok=True)
    
    csv_filename = f"evaluation/results/benchmark_{args.version}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Benchmark complete! Output saved to: {csv_filename}")

if __name__ == "__main__":
    main()