from enum import IntEnum
from gameboard import Markers

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
        self.active = True
        
        self.gameboard.set_marker(row, col, Markers.SNAKE)
        self.grow(length-1)
    
    def destroy(self):
        seg = self.head
        while seg is not None:
            self.gameboard.set_marker(seg.row, seg.col, Markers.FLOOR)
            seg = seg.link
        
        self.active = False
    
    def crash(self):
        row = self.head.row
        col = self.head.col
        self.gameboard.set_marker(row, col, Markers.SNAKE_CRASH)
    
    def move(self, direc):
        if not self.active:
            return False, False

        if direc is None:
            direc = self.head.direc

        if direc == -1*self.head.direc:
            direc = -1*direc

        # Check if the head will move onto an apple
        new_head_row = self.head.row + direc_dists[direc][0]
        new_head_col = self.head.col + direc_dists[direc][1]
        ate_apple = self.gameboard.get_marker(new_head_row, new_head_col) == Markers.APPLE

        seg = self.head
        while seg is not None:
            row = seg.row + direc_dists[direc][0]
            col = seg.col + direc_dists[direc][1]

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

        return True, ate_apple
    
    def grow(self, length):
        if not self.active:
            return
    
        for i in range(length):
            row = self.tail.row - direc_dists[self.tail.direc][0]
            col = self.tail.col - direc_dists[self.tail.direc][1]
            
            is_wall = self.gameboard.get_marker(row, col) == Markers.WALL
            is_snake = self.gameboard.get_marker(row, col) == Markers.SNAKE
            if is_wall or is_snake:
                return i
            
            self.tail.link = Segment(row, col, self.tail.direc)
            self.tail = self.tail.link
            self.length+=1
            
            self.gameboard.set_marker(row, col, Markers.SNAKE)
        
        return length
    
