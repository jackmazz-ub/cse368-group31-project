import pygame
from enum import IntEnum

GRID_X = 0
GRID_Y = 0

CELL_WIDTH = 10
CELL_HEIGHT = 10

BG_COLOR = (0, 0, 0)
FG_COLOR = (255, 255, 255)
SCORE_COLOR = (255, 255, 255)

class Markers(IntEnum):
    FLOOR = 0
    WALL = 1
    APPLE = 2
    SNAKE = 3
    SNAKE_CRASH = 4

marker_colors = {
    Markers.FLOOR: (0, 0, 0),
    Markers.WALL: (0, 0, 255),
    Markers.APPLE: (255, 0, 0),
    Markers.SNAKE: (0, 255, 0),
    Markers.SNAKE_CRASH: (255, 255, 0),
}

class Gameboard:
    def __init__(self, rows, cols):
        self.markers = []
        self.rows = rows
        self.cols = cols
        self.font = None

        for i in range(rows):
            self.markers.append([])
            for j in range(cols):
                self.markers[i].append(Markers.FLOOR)

    def draw(self, screen, score=None):
        screen.fill(BG_COLOR)

        for i in range(self.rows+2):
            for j in range(self.cols+2):
                x = GRID_X + j * CELL_WIDTH
                y = GRID_Y + i * CELL_HEIGHT
                rect = (x, y, CELL_WIDTH, CELL_HEIGHT)

                row = i-1
                col = j-1

                color = marker_colors[self.get_marker(row, col)]
                pygame.draw.rect(screen, color, rect)

        # Draw scoreboard
        if score is not None:
            if self.font is None:
                self.font = pygame.font.Font(None, 36)

            score_text = f"Score: {score}"
            text_surface = self.font.render(score_text, True, SCORE_COLOR)

            # Position the score in the footer area
            score_y = (self.rows + 2) * CELL_HEIGHT + 10
            score_x = 10
            screen.blit(text_surface, (score_x, score_y))

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
        
