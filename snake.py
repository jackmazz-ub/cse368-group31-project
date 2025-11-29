from enum import IntEnum
from gameboard import Markers

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

# identifiers for each direction
# South is the negation of North
# West is the negation of East
class Directions(IntEnum):
    NORTH = 1
    SOUTH = -1
    EAST = 2
    WEST = -2

# map directions to their x-y distances
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

# a single data object of the snake
class Segment:
    def __init__(self, row, col, direc):
        self.row = row
        self.col = col
        self.direc = direc
        self.link = None

# a linked-list of segments
class Snake:
    def __init__(self, gameboard, length, row, col, direc):
        self.gameboard = gameboard
        self.head = Segment(row, col, direc)
        self.tail = self.head
        self.length = 1
        self.active = True
        
        self.gameboard.set_marker(row, col, Markers.SNAKE)
        self.grow(length-1) # grow to the specified length-1 (the head counts as 1 length)
    
    # remove self from the gameboard
    # prevent future movements
    def destroy(self):
        seg = self.head
        while seg is not None:
            self.gameboard.set_marker(seg.row, seg.col, Markers.FLOOR)
            seg = seg.link
        
        self.active = False
    
    # style head to be crash-colored
    def crash(self):
        row = self.head.row
        col = self.head.col
        self.gameboard.set_marker(row, col, Markers.SNAKE_CRASH)
    
    # attempt a move, return false if movement failed
    # return a second boolean if an apple was eaten
    def move(self, direc):
        
        # prevent movement if snake is destroyed
        if not self.active:
            return False, False
            
        # move in default direction (direction of the head) if one isn't specified
        if direc is None:
            direc = self.head.direc
            
        # moving backwards is invalid, move forwards instead
        if direc == -1*self.head.direc:
            direc = -1*direc

        # check if the head will move onto an apple
        new_head_row = self.head.row + direc_dists[direc][0]
        new_head_col = self.head.col + direc_dists[direc][1]
        ate_apple = self.gameboard.get_marker(new_head_row, new_head_col) == Markers.APPLE
        
        # move all segments in the chain
        seg = self.head
        while seg is not None:
            row = seg.row + direc_dists[direc][0]
            col = seg.col + direc_dists[direc][1]
            
            # if moving into a blocked cell, return success=false
            if self.gameboard.is_blocked(row, col):
                return False, False

            prev_row = seg.row
            prev_col = seg.col
            seg.row = row
            seg.col = col

            direc_swap = seg.direc
            seg.direc = direc
            direc = direc_swap

            seg = seg.link

            self.gameboard.set_marker(row, col, Markers.SNAKE)
            self.gameboard.set_marker(prev_row, prev_col, Markers.FLOOR)
        
        # if moving into an open cell, return success=true
        return True, ate_apple
    
    # add segments to the snake
    # return number of segments successfully added
    def grow(self, length):
        
        # prevent growth if snake is destroyed
        if not self.active:
            return
            
        # add segments to the chain
        for i in range(length):
            row = self.tail.row - direc_dists[self.tail.direc][0]
            col = self.tail.col - direc_dists[self.tail.direc][1]
            
            # stop adding segments if snake will go OOB, or into a snake body
            # return number of segments added up to this point
            if self.gameboard.is_blocked():
                return i
            
            self.tail.link = Segment(row, col, self.tail.direc)
            self.tail = self.tail.link
            self.length+=1
            
            self.gameboard.set_marker(row, col, Markers.SNAKE)
        
        # all segements were added successfully
        return length
    
