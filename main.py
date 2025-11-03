import pygame
import random
import sys
import os

from ui.gameboard import Gameboard
from util.snake import Snake

# initial display settings
SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = False # enable/disable fullscreen on startup
VSYNC = True

# number of cells on the gameboard
GRID_ROWS = 90
GRID_COLS = 175

# initial game state values
SNAKE_LENGTH = 4
SNAKE_ROW = random.randint(0, GRID_ROWS-1)
SNAKE_COL = random.randint(0, GRID_COLS-1)
SNAKE_DIRECTION = random.randint(0, 3)

screen = None
gameboard = None
snake = None
active = None

def toggle_fullscreen():
    global screen
    
    fullscreen = screen.get_flags() & pygame.FULLSCREEN != 0
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if not fullscreen else pygame.RESIZABLE, # set fullscreen or windowed
        vsync=VSYNC,
    )

def main(argv):
    global screen
    global fullscreen
    global gameboard
    global snake
    global active
    
    # init game state
    snake = Snake(SNAKE_LENGTH, SNAKE_ROW, SNAKE_COL, SNAKE_DIRECTION)
    gameboard = Gameboard(GRID_ROWS, GRID_COLS, snake)
    
    # set display position
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (SCREEN_X, SCREEN_Y)
    
    # init display
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if FULLSCREEN else pygame.RESIZABLE, # set fullscreen or windowed
        vsync=VSYNC,
    )
    
    active = True
    while active:
        gameboard.draw(screen)
        for event in  pygame.event.get():
        
            # press F11 to toggle fullscreen
            if event.type == pygame.KEYUP and event.key == pygame.K_F11:
                toggle_fullscreen()
            
            # press ESC to quit
            elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                active = False
                
            # quit on all quit events
            elif event.type == pygame.QUIT:
                active = False

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

