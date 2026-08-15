import os
from stable_baselines3 import PPO
from rl.fleet_env import FleetEnv

def main():
    env = FleetEnv(max_vehicles=10)

    print("♻️ Loading existing V2 model...")
    # Load the existing model and attach the environment and tensorboard
    model = PPO.load(
        "models/fleet_ppo_v2", 
        env=env, 
        tensorboard_log="./ppo_fleet_tensorboard/"
    )

    print("🚀 Resuming training for 450,000 more timesteps...")
    # reset_num_timesteps=False keeps your TensorBoard graph continuous!
    model.learn(total_timesteps=450_000, progress_bar=True, reset_num_timesteps=False)

    model.save("models/fleet_ppo_v2_500k")
    print("✅ Extended training complete! Saved as fleet_ppo_v2_500k.zip")

if __name__ == "__main__":
    main()