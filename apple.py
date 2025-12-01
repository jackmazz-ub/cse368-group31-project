import random
from gameboard import Markers

"""
=====================================================================================================
| APPLE CLASS |
===============
"""

class Apple:
    def __init__(self, gameboard, row, col):
        self.gameboard = gameboard
        self.row = row
        self.col = col
        self.active = True
        
        self.gameboard.set_marker(self.row, self.col, Markers.APPLE)
    
    def destroy(self):
        if not self.active:
            return
    
        self.gameboard.set_marker(self.row, self.col, Markers.FLOOR)
        self.row = None
        self.col = None
        self.active = False

