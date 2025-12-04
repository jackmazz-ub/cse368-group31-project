import random
from enum import IntEnum
from util.pointer import Pointer

from apple import apple_ptr
from gameboard import Markers, gameboard_ptr

"""
=====================================================================================================
| CONSTANTS |
=============
"""

SNAKE_INIT_LENGTH = 1 # initial length
SNAKE_GROW_RATE = 1 # growth per apple

V_SPAWN_PCT = 0.1 # percent of display to spawn vertically (originally 0.500)
H_SPAWN_PCT = 0.1 # percent of display to spawn horizontally (originally 0.500)

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

class Directions(IntEnum):
    NORTH = 1
    SOUTH = -1
    EAST = 2
    WEST = -2

direc_dists = {
    Directions.NORTH: (-1, 0),
    Directions.SOUTH: (1, 0),
    Directions.EAST: (0, 1),
    Directions.WEST: (0, -1),
}

"""
=====================================================================================================
| SNAKE CLASS |
===============
"""

snake_ptr = Pointer()

class Segment:
    def __init__(self, row, col, direc):
        self.row = row
        self.col = col
        self.direc = direc
        self.link = None

class Snake:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
        self.placed = False
        self.turn_direc = None
        self.crashing = False
        self.place()
    
    def place(self):
        if self.placed:
            self.remove()
        
        border_spawn = random.randint(0, 3)
        
        # start facing South if spawning on the North border
        if border_spawn == 0:
            row = 0
            col = random.randint(SNAKE_INIT_LENGTH-1, gameboard_ptr.value.cols-1)
            direc = Directions.SOUTH
        
        # start facing North if spawning on the South border
        elif border_spawn == 1:
            row = gameboard_ptr.value.rows-1
            col = random.randint(SNAKE_INIT_LENGTH-1, gameboard_ptr.value.cols-1)
            direc = Directions.NORTH
        
        # start facing West if spawning on the East border
        elif border_spawn == 2:
            row = random.randint(SNAKE_INIT_LENGTH-1, gameboard_ptr.value.rows-1)
            col = gameboard_ptr.value.cols-1
            direc = Directions.WEST
        
        # start facing East if spawning on the West border
        elif border_spawn == 3:
            row = random.randint(SNAKE_INIT_LENGTH-1, gameboard_ptr.value.rows-1)
            col = 0
            direc = Directions.EAST
        
        self.head = Segment(row, col, direc)
        self.tail = self.head
        self.length = 1
        self.placed = True
        self.turn_direc = None
        self.crashing = False
        
        gameboard_ptr.value.set_marker(row, col, Markers.SNAKE)
        self.grow(SNAKE_INIT_LENGTH-1) # grow to the specified length-1 (the head counts as 1 length)
    
    def remove(self):
        if not self.placed:
            return
    
        seg = self.head
        while seg is not None:
            gameboard_ptr.value.set_marker(seg.row, seg.col, Markers.FLOOR)
            seg.row = None
            seg.col = None
            seg = seg.link
        
        self.placed = False
    
    def crash(self):
        if not self.placed:
            return
    
        row = self.head.row
        col = self.head.col
        gameboard_ptr.value.set_marker(row, col, Markers.SNAKE_CRASH)
    
    def turn(self, direc):
        # turning backwards is invalid, turn forwards instead
        if direc == -1*self.head.direc:
            direc = -1*direc
        
        self.turn_direc = direc
    
    def move(self):
        if not self.placed:
            return False
        
        direc = self.turn_direc
        if direc is None:
            direc = self.head.direc

        new_head_row = self.head.row + direc_dists[direc][0]
        new_head_col = self.head.col + direc_dists[direc][1]
        ate_apple = new_head_row == apple_ptr.value.row and new_head_col == apple_ptr.value.col
        if ate_apple:
            self.grow(SNAKE_GROW_RATE)
            apple_ptr.value.place()
        
        # move all segments in the chain
        seg = self.head
        while seg is not None:
            row = seg.row + direc_dists[direc][0]
            col = seg.col + direc_dists[direc][1]
            
            # crash if moving into a blocked cell
            if gameboard_ptr.value.is_blocked(row, col):
                self.crashing = True
                return False
                
            prev_row = seg.row
            prev_col = seg.col
            seg.row = row
            seg.col = col

            direc_swap = seg.direc
            seg.direc = direc
            direc = direc_swap

            seg = seg.link
            
            gameboard_ptr.value.set_marker(row, col, Markers.SNAKE)
            gameboard_ptr.value.set_marker(prev_row, prev_col, Markers.FLOOR)
        
        # if moving into an open cell, stop crashing
        self.crashing = False
        
        return ate_apple
    
    def grow(self, length):
        if not self.placed:
            return
            
        # add segments to the chain
        for i in range(length):
            row = self.tail.row - direc_dists[self.tail.direc][0]
            col = self.tail.col - direc_dists[self.tail.direc][1]
            
            # stop adding segments if snake will grow into a wall or snake body
            if gameboard_ptr.value.is_blocked(row, col):
                return
            
            self.tail.link = Segment(row, col, self.tail.direc)
            self.tail = self.tail.link
            self.length += 1
            
            gameboard_ptr.value.set_marker(row, col, Markers.SNAKE)

