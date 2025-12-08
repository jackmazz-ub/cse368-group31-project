import gymnasium as gym
import math
import matplotlib.pyplot as plt
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
from gameboard import Markers, gameboard_ptr
from snake import Directions, snake_ptr

"""
=====================================================================================================
| CONSTANTS |
=============
"""

TRAINING_DATA_FILENAME = "data/training-data.h5"
TRAINING_PLOT_FILENAME = "data/training-plot.pdf"
DIAGNOSIS_PLOT_FILENAME = "data/diagnosis-plot.pdf"

ALPHA = 0.0005 # learning rate - reduced for more stable learning
GAMMA = 0.95 # discount factor - reduced to focus more on immediate rewards (apples)
EPSILON_START = 1.0 # exploration rate
EPSILON_DECAY = 0.998 # slower decay = more exploration = more risk-taking
EPSILON_MIN = 0.05 # higher minimum = always some exploration
TAU = 0.005 # slower target network update for stability

TRAINING_EPISODES = 5000
DIAGNOSIS_EPISODES = 10000
MAX_STEPS = 2000 # Reduced - forces snake to be efficient

BATCH_SIZE = 128  # Increased for more stable learning
MEMORY_SIZE = 100000  # Reduced to save memory and speed up sampling
UPDATE_FREQUENCY = 4  # Update network every N steps

WIN_REWARD = 50000
APPLE_REWARD = 15000  # Very high reward for eating apples
LOSS_PENALTY = -5000  # Reduced penalty - don't be TOO afraid of dying
STEP_PENALTY = -1  # Small penalty for each step to encourage efficiency

RETRAIN = False # set to True to train on every initialization
DIAGNOSE = False # set to True to diagnose on every initialization

STAT_PREC = 3 # number of decimal points to round when calculating statistics

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
        # Optimized architecture: deeper network with dropout for better generalization
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Prevent overfitting
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
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
            shape=(11,), # Expanded to include danger detection
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
        
        if DIAGNOSE:
            self.diagnose()
            
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

        # Detect danger in all 4 directions (wall or snake body)
        danger_north = 1 if gameboard_ptr.value.is_blocked(head_pos[0] - 1, head_pos[1]) else 0
        danger_south = 1 if gameboard_ptr.value.is_blocked(head_pos[0] + 1, head_pos[1]) else 0
        danger_east = 1 if gameboard_ptr.value.is_blocked(head_pos[0], head_pos[1] + 1) else 0
        danger_west = 1 if gameboard_ptr.value.is_blocked(head_pos[0], head_pos[1] - 1) else 0

        # Current direction
        current_dir = snake_ptr.value.head.direc
        dir_north = 1 if current_dir == Directions.NORTH else 0
        dir_south = 1 if current_dir == Directions.SOUTH else 0
        dir_east = 1 if current_dir == Directions.EAST else 0
        dir_west = 1 if current_dir == Directions.WEST else 0

        state = np.array([
            delta[0],  # row difference to apple
            delta[1],  # col difference to apple
            danger_north,
            danger_south,
            danger_east,
            danger_west,
            dir_north,
            dir_south,
            dir_east,
            dir_west,
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
        # Get previous distance to apple
        head_pos = [snake_ptr.value.head.row, snake_ptr.value.head.col]
        apple_pos = [apple_ptr.value.row, apple_ptr.value.col]
        prev_dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))

        direc = action_direcs[action]
        snake_ptr.value.turn(direc)
        ate_apple = snake_ptr.value.move()

        # return penalty if the snake crashes
        if snake_ptr.value.crashing:
            return self.get_state(), LOSS_PENALTY, True

        # reset the apple if it was eaten, and increase the reward
        if ate_apple:
            reward = APPLE_REWARD
        else:
            # Calculate new distance to apple
            head_pos = [snake_ptr.value.head.row, snake_ptr.value.head.col]
            apple_pos = [apple_ptr.value.row, apple_ptr.value.col]
            new_dist = np.linalg.norm(np.array(head_pos) - np.array(apple_pos))

            # Much stronger rewards for getting closer to apple
            distance_change = prev_dist - new_dist
            reward = distance_change * 100  # Scale up the reward/penalty based on distance change

            # Heavy penalty for moving away from apple
            if new_dist > prev_dist:
                reward -= 50  # Extra penalty for moving away

        # return reward if the snake reaches 100% length
        if not apple_ptr.value.placed:
            print("victory!")
            return self.get_state(), WIN_REWARD, True

        return self.get_state(), reward, False
    
    def train(self):
        scores = []
        mean_steps = 0.0
        mean_score = 0.0
        min_score = gameboard_ptr.value.rows * gameboard_ptr.value.cols
        max_score = 0
    
        epsilon = EPSILON_START
        step_count = 0

        for i in range(1, TRAINING_EPISODES+1):        
            # reset the state
            snake_ptr.value.place()
            apple_ptr.value.place()

            state = self.get_state()

            total_reward = 0.0
            done = False

            for j in range(1, MAX_STEPS+1):
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
                step_count += 1

                # Update network every UPDATE_FREQUENCY steps for efficiency
                if step_count % UPDATE_FREQUENCY == 0:
                    self.replay()

                if done:
                    break
            
            n = i
            steps = j
            score = snake_ptr.value.length
            
            scores.append(score)
            mean_steps = (mean_steps*n + steps)/n
            mean_score = (mean_score*n + score)/n
            min_score = min(score, min_score)
            max_score = max(score, max_score)
            
            print(f"Training Episode {i}/{TRAINING_EPISODES}, \tSteps: {j}, \tEpsilon: {round(epsilon, 3)},   \tReward: {int(total_reward)}, \t\tScore: {score}")

            # reduce exploration rate
            epsilon = max(epsilon*EPSILON_DECAY, EPSILON_MIN)

            # Save progress every 100 episodes
            if i % 100 == 0:
                self.save_training_data()
                self.save_training_plot(scores, i)
                print(f"Progress saved at episode {i}")

        self.save_training_data()
        self.save_training_plot(
            scores, 
            mean_steps, 
            mean_score, 
            min_score, 
            max_score, 
            TRAINING_EPISODES,
        )
        
        print("Training complete!")
    
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
    
    def diagnose(self):
        scores = {} # score -> number of occurrences
        mean_steps = 0.0
        mean_score = 0.0
        min_score = gameboard_ptr.value.rows * gameboard_ptr.value.cols
        max_score = 0
        for i in range(1, DIAGNOSIS_EPISODES+1):
            # reset the state
            snake_ptr.value.place()
            apple_ptr.value.place()
            
            for j in range(1, MAX_STEPS):
                self.move() # exploit
                if snake_ptr.value.crashing:
                    break
            
            n = i
            steps = j
            score = snake_ptr.value.length
            
            if score not in scores:
                scores[score] = 0
            scores[score] += 1
            
            mean_steps = (mean_steps*n + steps)/n
            mean_score = (mean_score*n + score)/n
            min_score = min(score, min_score)
            max_score = max(score, max_score)
            
            print(f"Diagnosis Episode {i}/{DIAGNOSIS_EPISODES}, \tScore: {score}")
        
        self.save_diagnostics_plot(
            scores, 
            mean_steps,
            mean_score,
            min_score,
            max_score,
            DIAGNOSIS_EPISODES,
        )
    
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
        
    def save_training_plot(self, scores, mean_steps, mean_score, min_score, max_score, n):
        # average score
        mean = 0.0
        for score in scores:
            mean += score
        mean = mean/n
        
        # standard deviation from mean
        stdev = 0.0
        for score in scores:
            stdev += (score - mean)**2
        stdev = math.sqrt(1/(n-1)*stdev)
        
        var = stdev**2
        
        # round to STAT_PREC decimal points
        mean = round(mean, STAT_PREC)
        stdev = round(stdev, STAT_PREC)
        var = round(var, STAT_PREC)
    
        plt_text = (
            f"Score Range: [{min_score}, {max_score}]\n"
            f"Mean Score: {mean_score}\n"
            f"Min Score {min_score}\n"
            f"Min Score {max_score}\n"
            f"Mean Steps {mean_steps}\n"
        )
        
        plt.figure()
        
        plt.plot(range(1, len(scores)+1)[::10], scores[::10])
        
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title(f"Scores Over {n} Episodes (Before Training)")

        plt.gcf().text(0.02, -0.05, plt_text, fontsize=11, va="top")
        plt.tight_layout()
        plt.savefig(TRAINING_PLOT_FILENAME, format="pdf", bbox_inches="tight")
    
    def save_diagnostics_plot(self, scores, mean_steps, mean_score, min_score, max_score, n):    
        plt_data = [0]*(max_score+1)
        for i in range(len(plt_data)):
            if i not in scores:
                plt_data[i] = 0
            else:
                plt_data[i] = scores[i]
        
        plt_text = (
            f"Score Range: [{min_score}, {max_score}]\n"
            f"Mean Score: {mean_score}\n"
            f"Min Score {min_score}\n"
            f"Min Score {max_score}\n"
            f"Mean Steps {mean_steps}\n"
        )
        
        plt.figure()
        
        plt.bar(range(len(plt_data)), plt_data)
        
        plt.xlabel("Score")
        plt.ylabel("Number of Occurrences")
        plt.title(f"Scores Over {n} Episodes (After Training)")

        plt.gcf().text(0.02, -0.05, plt_text, fontsize=11, va='top')
        plt.tight_layout()
        plt.savefig(DIAGNOSIS_PLOT_FILENAME, format="pdf", bbox_inches="tight")
        
