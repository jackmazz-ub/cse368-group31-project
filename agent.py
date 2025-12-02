import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from collections import deque

from snake import Directions

"""
=====================================================================================================
| CONSTANTS |
=============
"""

# q algorithm parameters
ALPHA = 0.001
GAMMA = 0.990
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TAU = 0.01

BATCH_SIZE = 64

"""
=====================================================================================================
| AGENT CLASS |
===============
"""

class Agent(gym.Env):
    def __init__(self, gameboard, snake_ptr, apple_ptr):
        self.action_space = spaces.Discrete(4) # four directions (NSEW)
        self.observation_space = spaces.Box(
            low=-500, # lower bound
            high=500, # upper bound
            shape=(25,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )
        
        self.snake_ptr = snake_ptr
        self.gameboard = gameboard
        self.apple_ptr = apple_ptr
        
        self.reward = 0
    
    def train(self):
        # train and store results in a file
        # update current training data
        pass
    
    def move(self):
        # choose random direction (temporary)
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
        
        snake = self.snake_ptr.value # dereference snake pointer
        return snake.move(direc, self.apple_ptr)

