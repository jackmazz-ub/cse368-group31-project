import random
from snake import Directions

"""
=====================================================================================================
| AGENT CLASS |
===============
"""

class Agent:
    def choose_direction(self):
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
        return direc

