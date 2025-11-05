import pygame
from enum import IntEnum

GRID_X = 0
GRID_Y = 0

CELL_WIDTH = 10
CELL_HEIGHT = 10

BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
FLOOR_COLOR = (0, 0, 0)
WALL_COLOR = (0, 0, 255)
APPLE_COLOR = (255, 0, 0)
SNAKE_COLOR = (0, 255, 0)

PURPLE_COLOR = (255, 0, 255) # debug purposes

class Markers(IntEnum):
    FLOOR = 0
    WALL = 1
    APPLE = 2
    SNAKE = 3
    
    PURPLE = 4 # debug purposes

marker_colors = {
    Markers.FLOOR: FLOOR_COLOR,
    Markers.WALL: WALL_COLOR,
    Markers.APPLE: APPLE_COLOR,
    Markers.SNAKE: SNAKE_COLOR,
    
    Markers.PURPLE: PURPLE_COLOR, # debug purposes
}

class Gameboard:
    def __init__(self, rows, cols):        
        self.markers = []
        self.rows = rows
        self.cols = cols
        
        for i in range(rows):
            self.markers.append([])
            for j in range(cols):
                self.markers[i].append(Markers.FLOOR)

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        
        for i in range(self.rows+2):
            for j in range(self.cols+2):
                x = GRID_X + j * CELL_WIDTH
                y = GRID_Y + i * CELL_HEIGHT
                rect = (x, y, CELL_WIDTH, CELL_HEIGHT)
                
                row = i-1
                col = j-1
                
                color = marker_colors[self.get_marker(row, col)]
                pygame.draw.rect(screen, color, rect)
        
        pygame.display.update()
    
    def get_marker(self, row, col):
        row_in = row >= 0 and row < self.rows
        col_in = col >= 0 and col < self.cols
        if row_in and col_in:
            return self.markers[row][col]
        
        return Markers.WALL
    
    def set_marker(self, row, col, marker):
        if self.get_marker(row, col) != Markers.WALL:
            self.markers[row][col] = marker
    
    def is_blocked(self, row, col):
        marker = self.get_marker(row, col)
        return marker == Markers.WALL or marker == Markers.SNAKE
        
