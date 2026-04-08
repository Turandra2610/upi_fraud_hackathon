import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# Updated to match the class name in your models.py
from models import FraudPolicyNet

class UPIFraudEnv(gym.Env):
    def __init__(self):
        super(UPIFraudEnv, self).__init__()
        # 0: Allow, 1: Flag, 2: Block
        self.action_space = spaces.Discrete(3)
        # Input features (e.g., amount, time, location)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a random state for the transaction
        self.state = np.random.rand(3).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # amount is the first feature in our state
        amount = self.state[0]
        reward = 0.0
        
        # Reward Logic for Hackathon
        if amount > 0.8: # Likely Fraudulent High-Value Transaction
            reward = 10.0 if action == 2 else -20.0
        elif amount < 0.2: # Likely Safe Small Transaction
            reward = 5.0 if action == 0 else -10.0
        else: # Normal Transaction
            reward = 1.0
            
        # For a single transaction check, we terminate immediately
        terminated = True 
        return self.state, reward, terminated, False, {}
