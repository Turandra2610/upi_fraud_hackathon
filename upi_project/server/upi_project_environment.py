import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# This line is crucial: it tells the server to look for models.py in the folder above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your models.py file
from models import FraudPolicy

class UPIFraudEnv(gym.Env):
    def __init__(self):
        super(UPIFraudEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(3).astype(np.float32)
        return self.state, {}

    def step(self, action):
        amount = self.state[0]
        reward = 0.0
        
        # Reward Logic for Hackathon
        if amount > 0.8: # Likely Fraud
            reward = 10.0 if action == 2 else -20.0
        elif amount < 0.2: # Likely Safe
            reward = 5.0 if action == 0 else -10.0
        else:
            reward = 1.0
            
        terminated = True 
        return self.state, reward, terminated, False, {}