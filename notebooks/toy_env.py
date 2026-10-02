"""
Toy Gymnasium environment - just to learn the API.
A 1D grid where the agent walks left/right trying to reach position 5.

Nothing to do with cyber - but teaches the shape of Gymnasium.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ToyGridEnv(gym.Env):
    """Agent starts at position 0, goal is position 5, world size = 6."""

    def __init__(self):
        super().__init__()

        # World size: positions 0..5
        self.size = 6
        self.goal = 5

        # --- Observation space ---
        # What the agent sees: its current position (a single integer 0..5)
        self.observation_space = spaces.Discrete(self.size)

        # --- Action space ---
        # What the agent can do: 0 = move left, 1 = move right
        self.action_space = spaces.Discrete(2)

        # Current position (set properly in reset())
        self.pos = 0

    def reset(self, seed=None, options=None):
        """Start a fresh episode."""
        super().reset(seed=seed)
        self.pos = 0
        obs = self.pos
        info = {}
        return obs, info

    def step(self, action):
        """Apply action, advance one tick."""
        if action == 0:
            self.pos = max(0, self.pos - 1)
        else:
            self.pos = min(self.size - 1, self.pos + 1)

        if self.pos == self.goal:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False

        truncated = False
        obs = self.pos
        info = {}
        return obs, reward, terminated, truncated, info
