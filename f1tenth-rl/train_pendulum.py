import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
from stable_baselines3 import PPO  # or SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = gym.make("Pendulum-v1")
env = Monitor(env)  # Logs to TensorBoard
env = DummyVecEnv([lambda: env])  # Vectorized env for SB3

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./pendulum_tensorboard/")
model.learn(total_timesteps=50_000)
model.save("ppo_pendulum")

print("✅ Training complete and model saved.")
