import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl.fleet_env_v1 import FleetEnvV1

class CostLoggingCallback(BaseCallback):
    """
    Custom callback to log the separated service-level cost 
    (lateness + unserved shipments) to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(CostLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if any environments just finished an episode
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode_cost" in info:
                    # Log the final cumulative constraint cost for the episode
                    self.logger.record("rollout/ep_constraint_cost", info["episode_cost"])
        return True

def main():
    os.makedirs("models", exist_ok=True)

    # Initialize the NEW advanced logistics environment
    env = FleetEnvV1(max_vehicles=10)

    # Setup PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./ppo_fleet_v1_tensorboard/"
    )

    # Instantiate our custom logging callback
    cost_logger = CostLoggingCallback()

    print("🚀 Starting PPO Training for V1 (Advanced Logistics Formulation)...")
    
    # Pass the callback into the learn method
    model.learn(total_timesteps=50_000, progress_bar=True, callback=cost_logger)

    # Save cleanly under a v1 nomenclature to protect your old baselines
    model.save("models/fleet_ppo_v1_advanced")
    print("✅ Training complete! Model saved to models/fleet_ppo_v1_advanced.zip")

if __name__ == "__main__":
    main()