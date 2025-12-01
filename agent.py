# import gymnasium as gym
# import numpy as np
import random

from snake import Directions

"""
=====================================================================================================
| CONSTANTS |
=============
"""

TRAINING_DATA_FILENAME = "data/training-data.csv"

ALPHA = 0.8
# GAMMA = 

"""
=====================================================================================================
| AGENT CLASS |
===============
"""

class Agent():
    def __init__(self, gameboard, snake, apple):
        # self.action_space = spaces.Discrete(4) # four directions (NSEW)
        """self.observation_space = spaces.Box(
            low=-500, # lower bound
            high=500, # upper bound
            shape=(25,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )"""
        
        self.snake = snake
        self.gameboard = gameboard
        self.apple = apple
    
    def train(self):
        # train and store results in a file
        # update current training data
        pass
    
    def move(self):
        # choose random direction (temporary)
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
        
        return self.snake.move(direc, self.apple)

