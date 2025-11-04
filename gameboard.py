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

class Gameboard:
    def __init__(self, rows, cols, snake):        
        self.cells = []
        self.rows = rows
        self.cols = cols
        self.snake = snake
        
        # initialize all markers to floor (no content)
        for i in range(rows):
            self.cells.append([])
            for j in range(cols):
                self.cells[i].append(Markers.FLOOR)

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        
        # represent the snake as markers
        for segment in self.snake:
            if self.in_bounds(segment.row, segment.col):
                self.cells[segment.row][segment.col] = Markers.SNAKE
        
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
                
                # make the cell a wall if it is out-of-bounds
                marker = Markers.WALL
                if self.in_bounds(row, col):
                    marker = self.cells[row][col]
                
                # choose color depending on marker
                # this is the primary purpose of the marker system
                color = FLOOR_COLOR
                match marker:
                    case Markers.SNAKE:
                        color = SNAKE_COLOR
                    case Markers.APPLE:
                        color = APPLE_COLOR
                    case Markers.WALL:
                        color = WALL_COLOR
                
                pygame.draw.rect(screen, color, rect)
        
        pygame.display.update()
        
        # remove snake from markers for the next round
        for segment in self.snake:
            if self.in_bounds(segment.row, segment.col):
                self.cells[segment.row][segment.col] = Markers.FLOOR
    
    # check whether a row and column are within the bounds of the grid
    def in_bounds(self, row, col):
        row_in = row >= 0 and row < self.rows
        col_in = col >= 0 and col < self.cols
        return row_in and col_in
        
