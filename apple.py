import random
from util.pointer import Pointer

from gameboard import Markers, gameboard_ptr

"""
=====================================================================================================
| APPLE CLASS |
===============
"""

apple_ptr = Pointer()

class Apple:
    def __init__(self):
        self.row = None
        self.col = None
        self.placed = False
        self.place()
    
    def place(self):
        if self.placed:
            self.remove()
    
        # find all possible positions (not occupied by anything)
        # if none are found, cancel placement
        empty_cells = gameboard_ptr.value.list_cells(match_markers=[Markers.FLOOR])
        if len(empty_cells) == 0:
            return

        cell = random.choice(empty_cells)
        self.row = cell.row
        self.col = cell.col
        self.placed = True
        
        gameboard_ptr.value.set_marker(self.row, self.col, Markers.APPLE)
    
    def remove(self):
        if not self.placed:
            return
    
        gameboard_ptr.value.set_marker(
            self.row, self.col, 
            Markers.FLOOR,
            match_markers=[Markers.APPLE],
        )
        
        self.row = None
        self.col = None
        self.placed = False

