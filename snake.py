from enum import IntEnum
from gameboard import Markers

class Directions(IntEnum):
    NORTH = 1
    SOUTH = -1
    EAST = 2
    WEST = -2

DIREC_DISTS = {
    Directions.NORTH: (-1, 0),
    Directions.SOUTH: (1, 0),
    Directions.EAST: (0, 1),
    Directions.WEST: (0, -1),
}

class Segment:
    def __init__(self, row, col, direc):
        self.row = row
        self.col = col
        self.direc = direc
        self.link = None

class Snake:
    def __init__(self, gameboard, length, row, col, direc):
        self.gameboard = gameboard
        self.head = Segment(row, col, direc)
        self.tail = self.head
        self.length = 1
        self.grow(length-1)
    
    def move(self, direc):
        if direc is None:
            direc = self.head.direc
        
        if direc == -1*self.head.direc:
            return True
    
        seg = self.head
        while seg is not None:
            row = seg.row + DIREC_DISTS[direc][0]
            col = seg.col + DIREC_DISTS[direc][1]
            
            if self.gameboard.is_blocked(row, col):
                return False
            
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
        
        return True
    
    def grow(self, length):
        for i in range(length):
            row = self.tail.row - DIREC_DISTS[self.tail.direc][0]
            col = self.tail.col - DIREC_DISTS[self.tail.direc][1]
            
            is_wall = self.gameboard.get_marker(row, col) == Markers.WALL
            is_snake = self.gameboard.get_marker(row, col) == Markers.SNAKE
            if is_wall or is_snake:
                return i
            
            self.tail.link = Segment(row, col, self.tail.direc)
            self.tail = self.tail.link
            self.length+=1
            
            self.gameboard.set_marker(row, col, Markers.SNAKE)
        
        return length
    
