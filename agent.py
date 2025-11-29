import random
from enum import IntEnum

from snake import Directions

class Actions(IntEnum):
    MOVE_UP = 1
    MOVE_DOWN = -1
    MOVE_LEFT = -2
    MOVE_RIGHT = 2

class Agent:
    def choose_direction(self):
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
        return direc

