import gymnasium as gym
import numpy as np
import random
import os
from util.pointer import Pointer

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from enum import IntEnum
from gymnasium import spaces

from apple import apple_ptr
from gameboard import Markers
from snake import Directions, snake_ptr

"""
=====================================================================================================
| CONSTANTS |
=============
"""

TRAINING_DATA_FILENAME = "data/training-data.h5"

ALPHA = 0.001 # learning rate
GAMMA = 0.99 # discount factor
EPSILON_START = 1.0 # exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TAU = 0.01

EPISODES = 5000  # Increased from 1000 for more training
MAX_STEPS = 5000

BATCH_SIZE = 64
MEMORY_SIZE = 500000

WIN_REWARD = 50000
APPLE_REWARD = 5000
LOSS_PENALTY = -5000

RETRAIN = False # set to True to train on every initialization

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

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

agent_ptr = Pointer()

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
    def __init__(self):
        self.observation_space = spaces.Box(
            low=-500, # lower bound
            high=500, # upper bound
            shape=(5,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )
        self.state_size = self.observation_space.shape[0]
        
        self.action_space = spaces.Discrete(4) # four directions (NSEW)
        self.action_size = self.action_space.n
        
        # initialize network
        self.policy_model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.memory = deque(maxlen=MEMORY_SIZE)  
        
        # load training data if not retraining, and if the traning data exists
        if not RETRAIN and os.path.isfile(TRAINING_DATA_FILENAME):
            self.load_training_data()
        else:
            self.train()
            
    def get_state(self):
        head_pos = [snake_ptr.value.head.row, snake_ptr.value.head.col]
        apple_pos = [apple_ptr.value.row, apple_ptr.value.col]
        
        # calculate difference between head and snake positions
        delta = [0, 0]
        if apple_ptr.value.placed:
            delta = [
                head_pos[0] - apple_pos[0], # row difference
                head_pos[1] - apple_pos[1], # col difference
            ]
        
        state = np.array([
            snake_ptr.value.head.row,
            snake_ptr.value.head.col,
            delta[0],
            delta[1],
            snake_ptr.value.length,
        ])

        return state
    
    def move(self):
        # get the best action for the current state
        action = self.exploit(self.get_state())
        
        direc = action_direcs[action]
        snake_ptr.value.turn(direc)
        return snake_ptr.value.move()
    
    def step(self, action):
        reward = 0.0
        
        direc = action_direcs[action]f
        snake_ptr.value.turn(direc)
        ate_apple = snake_ptr.value.move()
        
        # return penalty if the snake crashes
        if snake_ptr.value.crashing:
            return self.get_state(), LOSS_PENALTY, True
        
        # reset the apple if it was eaten, and increase the reward
        if ate_apple:
            reward += APPLE_REWARD
            print("ate apple!")
        
        # return reward if the snake reaches 100% length
        if not apple_ptr.value.placed:
            print("victory!")
            return self.get_state(), WIN_REWARD, True
        
        # calculate the euclidian distance between the head and apple 
        head_pos = [snake_ptr.value.head.row, snake_ptr.value.head.col]
        apple_pos = [apple_ptr.value.row, apple_ptr.value.col]
        dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))
        
        # increase reward depending on distance to apple
        reward += 0.1*dist
        
        return self.get_state(), reward, False
    
    def train(self):
        epsilon = EPSILON_START
        
        for i in range(EPISODES):
            # reset the state
            snake_ptr.value.place()
            apple_ptr.value.place()
            
            state = self.get_state()
            
            total_reward = 0.0
            done = False
            
            for j in range(MAX_STEPS):            
                # explore or exploit based on P(epsilon)
                if np.random.rand() <= epsilon:
                    action = self.explore()
                else:
                    action = self.exploit(state)
                
                next_state, reward, done = self.step(action)
                total_reward += reward
                
                # add the step information to memory
                self.memory.append((
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                ))
                
                state = next_state
                
                if done:
                    break
            
            print(f"Episode: {i+1}, \tSteps: {j}, \tEpsilon: {round(epsilon, 2)}, \tReward: {int(total_reward)}")
            
            self.replay()
            
            # reduce exploration rate
            epsilon = max(epsilon*EPSILON_DECAY, EPSILON_MIN)
        
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
            target_param.data.copy_(TAU*online_param.data + (1.0-TAU)*target_param.data)
    
    def explore(self):
        return random.randrange(self.action_size)
    
    def exploit(self, state):
        tensor = torch.tensor(state, dtype=torch.float)
        return np.argmax(self.target_model(tensor).detach().numpy())
    
    def load_training_data(self):
        self.policy_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
        self.target_model.load_state_dict(torch.load(TRAINING_DATA_FILENAME))
    
    def save_training_data(self):
        torch.save(self.target_model.state_dict(), TRAINING_DATA_FILENAME)

