import numpy as np
from stable_baselines3 import PPO
from core.simulator import FleetSimulator
from rl.fleet_env import FleetEnv

def run_evaluation(strategy_name: str, model=None, num_episodes=5):
    """
    Evaluates a specific strategy (Greedy, OR-Tools, or AI) across multiple episodes
    and aggregates key performance metrics.
    """
    env = FleetEnv()
    
    total_completions = []
    total_ontime = []
    total_distances = []
    total_utilizations = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep) # Fixed seed for fair comparison
        
        # Configure strategy on the inner simulator
        if strategy_name == "Greedy":
            env.simulator.set_strategy("Greedy Dispatch")
        elif strategy_name == "OR-Tools":
            env.simulator.set_strategy("OR-Tools CVRP")
        elif strategy_name == "AI":
            env.simulator.set_strategy("AI")

        terminated = False
        while not terminated:
            if strategy_name == "AI" and model is not None:
                # PPO chooses the action
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Classical baselines use their internal optimization in simulator.step()
                # We pass a dummy action (0) since the simulator overrides it via strategy
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)

        # Grab final metrics from the episode
        metrics = env.simulator.get_metrics()
        total_completions.append(metrics["completion_rate"])
        total_ontime.append(metrics["on_time_rate"])
        total_distances.append(metrics["total_distance"])
        total_utilizations.append(metrics["utilization"])

    return {
        "Strategy": strategy_name,
        "Completion Rate": f"{np.mean(total_completions):.1f}%",
        "On-Time Delivery": f"{np.mean(total_ontime):.1f}%",
        "Fleet Utilization": f"{np.mean(total_utilizations):.1f}%",
        "Total Distance": f"{np.mean(total_distances):.1f} km"
    }

def main():
    print("📊 Starting Three-Way Fleet Evaluation...")
    
    # Load trained PPO model if available
    model_path = "models/fleet_ppo_v1.zip"
    ai_model = None
    try:
        ai_model = PPO.load(model_path)
        print("✅ Loaded trained PPO model successfully.")
    except Exception as e:
        print(f"⚠️ Could not load PPO model ({e}). Skipping AI evaluation.")

    results = []

    # 1. Evaluate Greedy Baseline
    print("Evaluating Greedy Baseline...")
    results.append(run_evaluation("Greedy", num_episodes=5))

    # 2. Evaluate OR-Tools Benchmark
    print("Evaluating OR-Tools CVRP...")
    results.append(run_evaluation("OR-Tools", num_episodes=5))

    # 3. Evaluate AI Coordinator
    if ai_model:
        print("Evaluating PPO AI Coordinator...")
        results.append(run_evaluation("AI", model=ai_model, num_episodes=5))

    # Print Final Comparison Table
    print("\n" + "="*60)
    print("🏆 FINAL COMPARISON REPORT (Averaged across test episodes)")
    print("="*60)
    
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()