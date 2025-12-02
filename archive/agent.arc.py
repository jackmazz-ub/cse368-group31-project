import gymnasium as gym
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.nn.functional as F

from gymnasium import spaces
from collections import deque

from apple import Apple
from gameboard import Markers
from snake import Snake, Directions

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

EPISODES = 12000
BATCH_SIZE = 64

"""
=====================================================================================================
| AGENT CLASS |
===============
"""

class Net(nn.Module):
    # torch.manual_seed(5)
    # np.random.seed(5)
    
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, self.action_size)
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
            shape=(25,), # 1-dimensional, 25 values
            dtype=np.float32 # data type (float)
        )
        
        self.state_size = self.observation_space.shape[0]
        self.action_size = self.action_space.n
        
        self.model = Net(self.state_size, self.action_size)
        self.target_model = Net(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=500000)
        
        self.snake_ptr = snake_ptr
        self.gameboard = gameboard
        self.apple_ptr = apple_ptr
        
        self.reward = 0
        self.done = False
    
    def move(self):
        # choose random direction (temporary)
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
        
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        return snake.move(direc, self.apple_ptr)
    
    def train(self):
        reward_list_training = []
        apple_list_training = []
        timestep_list_training = []
        
        eps_list = []
        max_apple_count = 1
        reached = False
    
        for i in range(EPISODES):
            state = self.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            episode_reward = 0
            timestep = 0
            
            while(not done):
                action = self.act(state)
                next_state, reward, done, info = self.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                timestep+=1
                
                if done:
                    if e % 1000 == 0:
                        print("Episode: {}/{}, Episode Reward: {}, Epsilon: {:.2}, Episode Apple Count: {}, Timestep: {}"
                              .format(i, EPISODES, episode_reward, EPSILON, self.apple_count, timestep))
                    
                    reward_list_training.append(episode_reward)
                    apple_list_training.append(self.apple_count)
                    timestep_list_training.append(timestep)
                    
                    self.save("billodal_syadavil_akhilshr_project_ddqn_1.h5")
                    break
                    
            if len(self.memory) > batch_size:
                self.replay()
            
            eps_list.append(EPSILON)
            if len(self.memory) > 60000:
                if not reached:
                    train_start_at = i
                    reached = True
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon*((self.epsilon_min/1)**(1/(EPISODES-train_start_at)))
    
    def step(self, direc):
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer
        
        success, ate_apple = snake.move(direc, self.apple_ptr)
        
        # increase reward significantly if reached 100% length
        
        head_pos = [snake.head.row, snake.head.col]
        apple_pos = [apple.row, apple.col]
        dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos)) # distance from apple
        
        if not success:
            self.reward = -100
            self.done = True
        elif success:
            apple_reward = 500 if ate_apple else 0
            self.reward = (150 - dist)/10 + apple_reward
            
        delta = [
            head_pos[0] - apple_pos[0], # row difference
            head_pos[1] - apple_pos[1], # col difference
        ]
        
        observation = np.array([
            snake.head.row,
            snake.head.col,
            delta[0],
            delta[1],
            snake.length,
        ]) #+ list(self.prev_actions))
        
        info = {} # not needed at this time
        
        return observation, self.reward, self.done, info
    
    def reset(self): 
        init_snake(self.gameboard, self.snake_ptr)
        init_apple(self.gameboard, self.apple_ptr)
        
        snake = self.snake_ptr.value # dereference snake pointer
        apple = self.apple_ptr.value # dereference apple pointer

        self.done = False

        head_pos = [snake.head.row, snake.head.col]
        apple_pos = [apple.row, apple.col]

        delta = [
            head_pos[0] - apple_pos[0], # row difference
            head_pos[1] - apple_pos[1], # col difference
        ]

        observation = np.array([
            snake.head.row,
            snake.head.col,
            delta[0],
            delta[1],
            snake.length,
        ]) #+ list(self.prev_actions))

        return observation
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # explore
        if np.random.rand() <= EPSILON:
            return random.randrange(self.action_size)
        
        # exploit
        tensor = torch.tensor(next_state, dtype=torch.float)
        return np.argmax(self.model(tensor).detach().numpy())

    def replay(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                tensor = torch.tensor(next_state, dtype=torch.float)
                next_action = np.argmax(self.model(tensor).detach().numpy())
                target = (reward + GAMMA * self.target_model(tensor)[0][next_action].detach().numpy())
            else:
                target = reward
                
            target_f = self.model(torch.tensor(state, dtype=torch.float))
            target_f[0][action] = target

            self.model.train_step(torch.tensor(state, dtype=torch.float), target_f)

        # if EPSILON > EPSILON_MIN:
        #     EPSILON *= EPSILON_DECAY

        params = zip(self.target_model.parameters(), self.model.parameters())
        for target_param, online_param in params:
            target_param.data.copy_(TAU*online_param.data + (1.0 - TAU)*target_param.data)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.target_model.state_dict(), name)

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
    
