import pygame
from enum import IntEnum

"""
=====================================================================================================
| CONSTANTS |
=============
"""

# positioning and sizing
GRID_X = 0 # gameboard x-position
GRID_Y = 0 # gameboard y-position
CELL_WIDTH = 10 # width of each square on the gameboard
CELL_HEIGHT = 10 # height of each square on the gameboard

# styling
BG_COLOR = (0, 0, 0) # background color
FG_COLOR = (255, 255, 255) # color of text and other foreground elements
FONT_FILEPATH = None
FONT_SIZE = 36

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

# the identifying data for each cell
# determines what 'object' is occupying the cell
class Markers(IntEnum):
    FLOOR = 0
    WALL = 1
    APPLE = 2
    SNAKE = 3
    SNAKE_CRASH = 4

# map markers to colors
marker_colors = {
    Markers.FLOOR: (0, 0, 0),
    Markers.WALL: (0, 0, 255),
    Markers.APPLE: (255, 0, 0),
    Markers.SNAKE: (0, 255, 0),
    Markers.SNAKE_CRASH: (0, 150, 0),
}

"""
=====================================================================================================
| GAMEBOARD CLASS |
===================
"""

# data object to represent a single cell on the board
class Cell:
    def __init__(self, row, col, marker):
        self.row = row
        self.col = col
        self.marker = marker

class Gameboard:
    def __init__(self, rows, cols):
        self.cells = []
        self.rows = rows
        self.cols = cols
        self.font = None
        
        # initialize the gameboard to contain only floors (empty spaces)
        for i in range(rows):
            self.cells.append([])
            for j in range(cols):
                cell = Cell(i, j, Markers.FLOOR)
                self.cells[i].append(cell)
                
    def draw(self, screen, game_mode=None, score=None, elapsed_time=None):
        screen.fill(BG_COLOR) # draw the background
        
        # draw each cell depending on it's marker
        # draw two extra rows and columns to represent the walls
        for i in range(self.rows+2):
            for j in range(self.cols+2):
                x = GRID_X + j * CELL_WIDTH # calculate the x-position of the cell
                y = GRID_Y + i * CELL_HEIGHT # calculate the y-position of the cell
                rect = (x, y, CELL_WIDTH, CELL_HEIGHT)
                
                row = i-1
                col = j-1

                color = marker_colors[self.get_marker(row, col)]
                pygame.draw.rect(screen, color, rect)

        # draw scoreboard and timer
        if game_mode is not None or score is not None or elapsed_time is not None:
            if self.font is None:
                self.font = pygame.font.Font(FONT_FILEPATH, FONT_SIZE)
            
            footer_y = (self.rows + 2) * CELL_HEIGHT + 10
            
            score_x = 10
            timer_x = ((self.cols + 2) * CELL_WIDTH) // 2
            game_mode_x = (self.cols + 2) * CELL_WIDTH - 10

            # draw score
            if score is not None:
                pct = max(round(score/(self.rows*self.cols) * 100, 2), 0.01)              
                score_text = f"Length: {score}  ({pct}%)"
                text_surface = self.font.render(score_text, True, FG_COLOR)
                screen.blit(text_surface, (score_x, footer_y))

            # draw timer
            if elapsed_time is not None:
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                timer_text = f"Time: {minutes:02d}:{seconds:02d}"
                text_surface = self.font.render(timer_text, True, FG_COLOR)
                screen.blit(text_surface, (timer_x - text_surface.get_width()//2, footer_y))
            
            # draw game mode
            if game_mode is not None:
                text_surface = self.font.render(game_mode, True, FG_COLOR)
                screen.blit(text_surface, (game_mode_x - text_surface.get_width(), footer_y))

        pygame.display.update()
    
    # accumulate cells in a list
    # include only those whose marker matches 'match' (if provided)
    def list_cells(self, match=None):
        cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.cells[i][j]
                if match is None or cell.marker == match:
                    cells.append(cell)
        
        return cells
    
    # return a marker if in bounds, else return a wall
    def get_marker(self, row, col):
        row_in = row >= 0 and row < self.rows
        col_in = col >= 0 and col < self.cols
        if row_in and col_in:
            return self.cells[row][col].marker
        
        return Markers.WALL
    
    # set the marker if not OOB
    def set_marker(self, row, col, marker):
        if self.get_marker(row, col) != Markers.WALL:
            self.cells[row][col].marker = marker
    
    # determine if a cell is occupied by a wall or snake body
    def is_blocked(self, row, col):
        marker = self.get_marker(row, col)
        return marker == Markers.WALL or marker == Markers.SNAKE
        
