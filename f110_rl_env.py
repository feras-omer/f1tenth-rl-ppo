import gym
import numpy as np
from gym import spaces
from f110_gym.envs.base_classes import Integrator
import os


class F110RLEnv(gym.Env):
    def __init__(self, map_name='example_map', map_ext='.png', timestep=0.01, num_agents=1):
        super(F110RLEnv, self).__init__()

        self.render_mode = None  # <-- Add this line

        # Use the path relative to this script's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(current_dir, map_name)

        self.sim = gym.make(
            'f110_gym:f110-v0',
            map=map_path,
            map_ext=map_ext,
            num_agents=num_agents,
            timestep=timestep,
            integrator=Integrator.RK4,
        )

        self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(1080,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 4.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        poses = np.array([[0.0, 0.0, 0.0]])  # x, y, theta
        obs, _, _, _ = self.sim.reset(poses)
        return obs['scans'][0].astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high).reshape(1, 2)
        obs, _, done, info = self.sim.step(action)
        scans = obs['scans'][0].astype(np.float32)

        forward_reward = 1.0 * action[0, 1]
        turn_penalty = 0.0 * abs(action[0, 0])
        collision_penalty = -5.0 if np.min(scans) < 0.2 else 0.0
        reward = forward_reward + collision_penalty + turn_penalty
        done = done or (collision_penalty < 0)

        return scans, reward, done, False, info

    def render(self):
        self.sim.render()
