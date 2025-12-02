import gymnasium as gym
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn.functional as F

import os

from collections import deque
from enum import IntEnum
from gymnasium import spaces

from apple import Apple
from gameboard import Markers
from snake import Snake, Directions

"""
=====================================================================================================
| CONSTANTS |
=============
"""

TRAINING_DATA_FILENAME = "data/training-data.h5"

# q algorithm parameters
ALPHA = 0.001
GAMMA = 0.990
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TAU = 0.01

EPISODES = 12000
BATCH_SIZE = 64
MEMORY_SIZE = 500000

RETRAIN = False

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

# identifiers for all possible actions
class Actions(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3

action_direcs = {
    Actions.MOVE_NORTH: Directions.NORTH,
    Actions.MOVE_SOUTH: Directions.SOUTH,
    Actions.MOVE_EAST: Directions.EAST,
    Actions.MOVE_WEST: Directions.WEST,
}

"""
=====================================================================================================
| AGENT CLASS |
===============
"""

class DQN(nn.Module):    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, action_size)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_step(self, state, target):
        self.optimizer.zero_grad()
        pred = self.forward(state)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class Agent(gym.Env):
    def __init__(self, gameboard, snake_ptr, apple_ptr):
        self.action_space = spaces.Discrete(4) # four directions (NSEW)
        self.observation_space = spaces.Box(
            low=-500, # lower bound
            high=500, # upper bound
            shape=(5,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )
        
        self.state_size = self.observation_space.shape[0]
        self.action_size = self.action_space.n
        
        self.policy_model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.memory = deque(maxlen=MEMORY_SIZE)  
        
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=ALPHA)
        self.loss_fn = nn.MSELoss()
        
        self.gameboard = gameboard
        self.snake_ptr = snake_ptr
        self.apple_ptr = apple_ptr
        
        if os.path.isfile(TRAINING_DATA_FILENAME):
            self.load_training_data()
        else:
            self.train()
            
        
    def get_state(self):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer

        head_pos = [snake.head.row, snake.head.col]
        apple_pos = [apple.row, apple.col]
        
        delta = [0, 0]
        if apple_pos[0] is not None and apple_pos[1] is not None:
            delta = [
                head_pos[0] - apple_pos[0], # row difference
                head_pos[1] - apple_pos[1], # col difference
            ]

        state = np.array([
            snake.head.row,
            snake.head.col,
            delta[0],
            delta[1],
            snake.length,
        ])

        return state
    
    def reset_state(self):
        init_snake(self.gameboard, self.snake_ptr)
        init_apple(self.gameboard, self.apple_ptr)
    
    def move(self):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        
        action = self.exploit(self.get_state())
        direc = action_direcs[action]
        return snake.move(direc, self.apple_ptr)
    
    def step(self, action):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        
        direc = action_direcs[action]
        success, ate_apple = snake.move(direc, self.apple_ptr)
        if ate_apple:
            init_apple(self.gameboard, self.apple_ptr)
            print("ate apple!")
        
        # increase reward significantly if reached 100% length
        if apple.row is None and apple.col is None:
            return self.get_state(), 500, True
        
        head_pos = [snake.head.row, snake.head.col]
        apple_pos = [apple.row, apple.col]
        dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))
        
        if not success:
            return self.get_state(), -100, True
            
        apple_reward = 500 if ate_apple else 0
        reward = (150 - dist)/10 + apple_reward
        return self.get_state(), reward, False
        
    def train(self):    
        epsilon = EPSILON_START
    
        for i in range(EPISODES):
            self.reset_state()
            state = self.get_state()
            total_reward = 0
            done = False
            
            j = 0
            while not done:
                if np.random.rand() <= epsilon:
                    action = self.explore()
                else:
                    action = self.exploit(state)
                
                next_state, reward, done = self.step(action)
                total_reward += reward
                
                self.memory.append((
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                ))
                
                state = next_state
                
                if done:
                    self.save_training_data()
                    break
            
            print(f"Episode: {i}, Reward: {total_reward}")
            
            self.replay()
            epsilon = max(epsilon*EPSILON_DECAY, EPSILON_MIN)
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            
            if not done:
                next_action = torch.argmax(self.policy_model(next_state_tensor)).item()
                next_q = self.target_model(next_state_tensor)[0][next_action].item()
            
            target_q = reward if done else reward + GAMMA * next_q
            
            target_f = self.policy_model(state_tensor)
            target_f = target_f.clone()
            target_f[0][action] = target_q

            self.policy_model.train_step(state_tensor, target_f)

        # params = zip(self.target_model.parameters(), self.policy_model.parameters())
        # for target_param, online_param in params:
        #     target_param.data.copy_(TAU*online_param.data + (1.0 - TAU)*target_param.data)
    
    def explore(self):
        return random.randrange(self.action_size)
    
    def exploit(self, state):
        tensor = torch.tensor(state, dtype=torch.float)
        return np.argmax(self.policy_model(tensor).detach().numpy())
    
    def load_training_data(self):
        self.policy_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
        self.target_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
    
    def save_training_data(self):
        torch.save(self.target_model.state_dict(), TRAINING_DATA_FILENAME)

"""
=====================================================================================================
| INITIALIZATION (TEMPORARY) |
==============================
"""

# snake spawn settings
V_SPAWN_PCT = 0.500 # percent of display to spawn vertically
H_SPAWN_PCT = 0.325 # percent of display to spawn horizontally
FOOTER_PCT = 0.1 # percent of display to use for the footer

# snake body settings
SNAKE_INIT_LENGTH = 4 # initial length
SNAKE_GROW_RATE = 5 # growth per apple

grid_rows = 45 # gameboard number of rows
grid_cols = 94 # gameboard number of columns

def init_snake(gameboard, snake_ptr):
    if snake_ptr.value is not None:
        snake_ptr.value.destroy()

    # choose starting head location
    # row & col chosen s.t. the snake's body will not spawn OOB
    length = SNAKE_INIT_LENGTH
    row = random.randint(length, grid_rows-length-1)
    col = random.randint(length, grid_cols-length-1)

    """
    Spawn Sectors
    ---------- ---------- ----------
    |        | | NORTH  | |        |
    |        | | SECTOR | |        |
    | WEST   | ---------- | EAST   |
    | SECTOR | ---------- | SECTOR |
    |        | | SOUTH  | |        |
    |        | | SECTOR | |        |
    ---------- ---------- ----------
    
    These sectors define areas of the board which are 'too close to the edge'.
    Snakes which spawn in these areas will begin facing away from the edge to
    reduce unfair spawns.
    """
    
    v_spawn = int(grid_rows * V_SPAWN_PCT) # defines the width of the East and West sectors
    h_spawn = int(grid_cols * H_SPAWN_PCT) # defines the height of the North and South sectors
    
    direc = None
    
    # start facing East if spawning in the West Sector
    if col < h_spawn:
        direc = Directions.EAST
        
    # start facing West if spawning in the East Sector
    elif col > grid_cols - h_spawn:
        direc = Directions.WEST
        
    # start facing South if spawning in the North Sector
    elif row < v_spawn:
        direc = Directions.SOUTH
        
    # start facing North if spawning in the South Sector
    elif row > grid_rows - v_spawn:
        direc = Directions.NORTH
        
    # start in a random direction if not spawning in any sectors
    else:
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1

    snake_ptr.value = Snake(gameboard, length, row, col, direc)

def init_apple(gameboard, apple_ptr):    
    # remove current apple if it exists
    if apple_ptr.value is not None:
        apple_ptr.value.destroy()
    
    # find all possible positions (not occupied by anything)
    # if none are found, end the game (the player won)
    empty_cells = gameboard.list_cells(Markers.FLOOR)
    if len(empty_cells) == 0:
        game_over = True
        return

    # spawn apple at random empty cell
    cell = random.choice(empty_cells)
    apple_ptr.value = Apple(gameboard, cell.row, cell.col)

