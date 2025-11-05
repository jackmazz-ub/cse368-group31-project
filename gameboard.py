import pygame
from enum import IntEnum

# initial grid position
GRID_X = 0
GRID_Y = 0

# size of each cell in the grid
CELL_WIDTH = 10
CELL_HEIGHT = 10

# style settings
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
FLOOR_COLOR = (0, 0, 0)
WALL_COLOR = (0, 0, 255)
APPLE_COLOR = (255, 0, 0)
SNAKE_COLOR = (0, 255, 0)

# used to identify the type of content within a cell
class Markers(IntEnum):
    FLOOR = 0
    WALL = 1
    APPLE = 2
    SNAKE = 3

# map markers to colors
MARKER_COLORS = {
    Markers.FLOOR: FLOOR_COLOR,
    Markers.WALL: WALL_COLOR,
    Markers.APPLE: APPLE_COLOR,
    Markers.SNAKE: SNAKE_COLOR,
}

class Gameboard:
    def __init__(self, rows, cols):        
        self.markers = []
        self.rows = rows
        self.cols = cols
        
        # initialize all markers to floor (no content)
        for i in range(rows):
            self.markers.append([])
            for j in range(cols):
                self.markers[i].append(Markers.FLOOR)

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        
        # iterate by +2 to add a layer of walls around the grid
        for i in range(self.rows+2):
            for j in range(self.cols+2):
            
                # calculate cell position given a row and column
                x = GRID_X + j * CELL_WIDTH
                y = GRID_Y + i * CELL_HEIGHT
                rect = (x, y, CELL_WIDTH, CELL_HEIGHT)
                
                # decrement row and column (needed to offset walls)
                row = i-1
                col = j-1
                
                color = MARKER_COLORS[self.get_marker(row, col)]
                pygame.draw.rect(screen, color, rect)
        
        pygame.display.update()
    
    def get_marker(self, row, col):
        row_in = row >= 0 and row < self.rows
        col_in = col >= 0 and col < self.cols
        if row_in and col_in:
            return self.markers[row][col]
        
        return Markers.WALL
    
    def set_marker(self, row, col, marker):
        if self.get_marker != Markers.WALL:
            self.markers[row][col] = marker
    
    def is_blocked(self, row, col):
        marker = self.get_marker(row, col)
        return marker == Markers.WALL or marker == Markers.SNAKE
        
