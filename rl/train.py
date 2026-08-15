import os
from stable_baselines3 import PPO
from rl.fleet_env import FleetEnv

def main():
    os.makedirs("models", exist_ok=True)

    # Initialize the 10-vehicle scalable environment
    env = FleetEnv(max_vehicles=10)

    # Setup PPO Agent with TensorBoard logging enabled
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./ppo_fleet_tensorboard/"
    )

    print("🚀 Starting PPO V2 Training (Scalable 10-Vehicle Architecture)...")
    # You can turn progress_bar=True back on if you installed 'rich' earlier
    model.learn(total_timesteps=50_000, progress_bar=False)

    # Save as v2 so we don't overwrite your 5-vehicle baseline
    model.save("models/fleet_ppo_v2")
    print("✅ Training complete! Model saved to models/fleet_ppo_v2.zip")

if __name__ == "__main__":
    main()