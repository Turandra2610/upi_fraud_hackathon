import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UPIFraudEnv(gym.Env):
    def __init__(self):
        super(UPIFraudEnv, self).__init__()
        # Actions: 0 = Allow, 1 = Flag, 2 = Block
        self.action_space = spaces.Discrete(3)
        
        # Observations: [Transaction Amount, User Frequency, Location Risk (0-1)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a random "Transaction" to investigate
        self.state = np.random.rand(3).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Logic: If amount > 0.8 (high) and action is 'Allow' (0), penalize!
        amount = self.state[0]
        reward = 0
        
        if amount > 0.8 and action == 0: # Failed to block fraud
            reward = -5.0
        elif amount > 0.8 and action == 2: # Successfully blocked fraud
            reward = 10.0
        elif amount < 0.2 and action == 2: # Blocked a safe user (False Positive)
            reward = -2.0
        else:
            reward = 1.0 # Standard correct processing
            
        terminated = True # Episode ends after one transaction check
        return self.state, reward, terminated, False, {}
    