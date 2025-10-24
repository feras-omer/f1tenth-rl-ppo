import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from f110_rl_env import F110RLEnv

if __name__ == "__main__":
    logdir = "./ppo_f110_logs"
    os.makedirs(logdir, exist_ok=True)

    def make_env():
        return F110RLEnv(map_name="example_map")

    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=200_000)
    model.save("ppo_f110_agent")

    print("✅ Training complete. Model saved as ppo_f110_agent.zip")
