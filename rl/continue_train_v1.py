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
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode_cost" in info:
                    self.logger.record("rollout/ep_constraint_cost", info["episode_cost"])
        return True

def main():
    env = FleetEnvV1(max_vehicles=10)

    print("♻️ Loading existing V1 advanced model...")
    # Load the V1 model, attaching the environment and tensorboard
    model = PPO.load(
        "models/fleet_ppo_v1_advanced", 
        env=env, 
        tensorboard_log="./ppo_fleet_v1_tensorboard/"
    )

    cost_logger = CostLoggingCallback()

    print("🚀 Resuming V1 training for 450,000 more timesteps...")
    # reset_num_timesteps=False keeps the TensorBoard graph continuous
    model.learn(
        total_timesteps=450_000, 
        progress_bar=True, 
        reset_num_timesteps=False,
        callback=cost_logger
    )

    model.save("models/fleet_ppo_v1_advanced_500k")
    print("✅ Extended training complete! Saved as fleet_ppo_v1_advanced_500k.zip")

if __name__ == "__main__":
    main()