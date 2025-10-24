import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")  # 'human' mode to display a window

# Simple manual interaction: reset and take a few random steps
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    # Render the environment
    env.render()  
    if terminated or truncated:
        obs, info = env.reset()  # reset if episode ends

env.close()  # close the render window

# 2. Train a PPO agent on CartPole for a short duration
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=10000)  # train for 10k timesteps as a test:contentReference[oaicite:32]{index=32}

# After training, test the learned agent (without training, just run a few episodes)
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)  # choose action from trained model
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()
