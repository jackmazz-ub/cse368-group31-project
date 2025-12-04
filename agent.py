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
ALPHA = 0.001 # learning rate
GAMMA = 0.990 # discount factor
EPSILON_START = 1.0 # exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TAU = 0.01

EPISODES = 1000 # number of episodes during training (originally 12000)
MAX_STEPS = 5000 # max number of steps per episode

BATCH_SIZE = 64
MEMORY_SIZE = 500000

# reward/penalty specification
WIN_REWARD = 50000 # reward for reaching 100% length
APPLE_REWARD = 5000 # reward for eating an apple
APPLE_DIST_MULTIPLIER = 0.1 # distance to apple reward multiplier
LOSS_PENALTY = -5000 # penalty for crashing

RETRAIN = False # set to True to train on every initialization

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

# map actions to directions
action_direcs = {
    Actions.MOVE_NORTH: Directions.NORTH,
    Actions.MOVE_SOUTH: Directions.SOUTH,
    Actions.MOVE_EAST: Directions.EAST,
    Actions.MOVE_WEST: Directions.WEST,
}

"""
=====================================================================================================
| INITIALIZATION CONSTANTS (TEMPORARY) |
========================================
"""

# snake spawn settings
V_SPAWN_PCT = 0.500 # percent of display to spawn vertically
H_SPAWN_PCT = 0.325 # percent of display to spawn horizontally
FOOTER_PCT = 0.1 # percent of display to use for the footer

# snake body settings
SNAKE_INIT_LENGTH = 4 # initial length
SNAKE_GROW_RATE = 5 # growth per apple

grid_rows = 10 # gameboard number of rows
grid_cols = 10 # gameboard number of columns

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
    
        # determine state size
        self.state_space = spaces.Box(
            low=-500, # lower bound
            high=500, # upper bound
            shape=(5,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )
        self.state_size = self.state_space.shape[0]
        
        # determine number of actions
        self.action_space = spaces.Discrete(4) # four directions (NSEW)
        self.action_size = self.action_space.n
        
        # initialize network
        self.policy_model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.memory = deque(maxlen=MEMORY_SIZE)  
        
        # initialize game elments
        self.gameboard = gameboard
        self.snake_ptr = snake_ptr # reference to main snake
        self.apple_ptr = apple_ptr # reference to main apple
        
        # load training data if not retraining, and if the traning data exists
        if not RETRAIN and os.path.isfile(TRAINING_DATA_FILENAME):
            self.load_training_data()
        else:
            self.train()
            
    # return the current state information
    def get_state(self):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        
        head_pos = [snake.head.row, snake.head.col] # position of snake head
        apple_pos = [apple.row, apple.col] # position of apple
        
        # calculate difference between head and snake positions
        # prevent calculation if no apple exists
        delta = [0, 0]
        if apple_pos[0] is not None and apple_pos[1] is not None:
            delta = [
                head_pos[0] - apple_pos[0], # row difference
                head_pos[1] - apple_pos[1], # col difference
            ]
        
        # state information
        state = np.array([
            snake.head.row,
            snake.head.col,
            delta[0],
            delta[1],
            snake.length,
        ])

        return state
    
    # reset the snake and apple
    def reset_state(self):
        self.init_snake()
        self.init_apple()
    
    # move the snake using the best action
    def move(self):
        snake = self.snake_ptr.value # dereference snake pointer
        
        action = self.exploit(self.get_state()) # get the best action for the current state
        direc = action_direcs[action] # get the direction that corresponds to the action
        return snake.move(direc)
    
    # take and action for training
    # return the state and reward for taking the action
    def step(self, action):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        
        # initialize reward
        reward = 0.0
        
        # move the snake depending on the provided action
        direc = action_direcs[action] # get the direction that corresponds to the provided action
        success, ate_apple = snake.move(direc)
        
        # return penalty if the snake crashes
        if not success:
            return self.get_state(), LOSS_PENALTY, True
        
        # reset the apple if it was eaten, and increase the reward
        if ate_apple:
            reward += APPLE_REWARD
            self.init_apple()
            apple = self.apple_ptr.value # update apple
            print("ate apple!")
        
        # return reward if the snake reaches 100% length
        if apple.row is None and apple.col is None:
            print("victory!")
            return self.get_state(), WIN_REWARD, True
        
        # calculate the euclidian distance between the head and apple 
        head_pos = [snake.head.row, snake.head.col]  # position of snake head
        apple_pos = [apple.row, apple.col] # position of apple
        dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))
        
        # increase reward depending on distance to apple
        reward += dist*APPLE_DIST_MULTIPLIER
        
        return self.get_state(), reward, False
    
    # run though the training loop
    # store the final training data
    def train(self):
        
        # initialize exploration rate
        epsilon = EPSILON_START
        
        for i in range(EPISODES):
        
            self.reset_state() # reset the state
            state = self.get_state() # get the initial state
            
            total_reward = 0.0
            done = False
            
            # take actions until the snake crashes, or wins
            for j in range(MAX_STEPS):            
                # explore or exploit based on P(epsilon)
                if np.random.rand() <= epsilon:
                    action = self.explore()
                else:
                    action = self.exploit(state)
                
                next_state, reward, done = self.step(action) # take the action
                total_reward += reward # accumulate the reward
                
                # add the step information to memory
                self.memory.append((
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                ))
                
                # update the state
                state = next_state
                
                # end the episode if snake wins/loses
                if done:
                    break
            
            # print training information
            print(f"Episode: {i+1}, \tSteps: {j}, \tReward: {int(total_reward)}")
            
            self.replay()
            
            # reduce exploration rate
            epsilon = max(epsilon*EPSILON_DECAY, EPSILON_MIN)
        
        # store training data
        self.save_training_data()
    
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

        params = zip(self.target_model.parameters(), self.policy_model.parameters())
        for target_param, online_param in params:
            target_param.data.copy_(TAU*online_param.data + (1.0 - TAU)*target_param.data)
    
    # return random action
    def explore(self):
        return random.randrange(self.action_size)
    
    # return the best action depending on the provided state
    def exploit(self, state):
        tensor = torch.tensor(state, dtype=torch.float)
        return np.argmax(self.target_model(tensor).detach().numpy())
    
    # load the training data at TRAINING_DATA_FILENAME
    def load_training_data(self):
        self.policy_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
        self.target_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
    
    # store the training data at TRAINING_DATA_FILENAME
    def save_training_data(self):
        torch.save(self.target_model.state_dict(), TRAINING_DATA_FILENAME)

    """
    =====================================================================================================
    | INITIALIZATION (TEMPORARY) |
    ==============================
    """

    def init_snake(self):
        if self.snake_ptr.value is not None:
            self.snake_ptr.value.destroy()

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

        self.snake_ptr.value = Snake(self.gameboard, self.apple_ptr, length, row, col, direc)

    def init_apple(self):    
        # remove current apple if it exists
        if self.apple_ptr.value is not None:
            self.apple_ptr.value.destroy()
        
        # find all possible positions (not occupied by anything)
        # if none are found, end the game (the player won)
        empty_cells = self.gameboard.list_cells(Markers.FLOOR)
        if len(empty_cells) == 0:
            game_over = True
            return

        # spawn apple at random empty cell
        cell = random.choice(empty_cells)
        self.apple_ptr.value = Apple(self.gameboard, cell.row, cell.col)

