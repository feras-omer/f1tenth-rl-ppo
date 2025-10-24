import time
import gym
import numpy as np
from stable_baselines3 import PPO
from f110_rl_env import F110RLEnv  # Make sure this is the correct path to your env

if __name__ == "__main__":
    print("🏁 Running trained PPO agent in F1TENTH simulator...")

    env = F110RLEnv(map_name='example_map')
    model = PPO.load("ppo_f110_agent")

    obs, _ = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _,_ = env.step(action)
        env.render()

        # Keep window responsive
        try:
            import pyglet
            pyglet.clock.tick()
            if hasattr(pyglet.app, 'platform_event_loop'):
                pyglet.app.platform_event_loop.dispatch_posted_events()
        except Exception:
            pass

        time.sleep(0.01)

        if done:
            obs, _ = env.reset()

