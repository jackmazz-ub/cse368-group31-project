import pygame
import random
import sys
import os
import time

from gameboard import Gameboard
from snake import Snake, Directions

# initial display settings
SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = True # enable/disable fullscreen on startup
VSYNC = True

# number of cells on the gameboard
GRID_ROWS = 90
GRID_COLS = 175

# initial game state values
SNAKE_LENGTH = 300
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
    direction = SNAKE_DIRECTION
    
    # set display position
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (SCREEN_X, SCREEN_Y)
    
    # init display
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if FULLSCREEN else pygame.RESIZABLE, # set fullscreen or windowed
        vsync=VSYNC,
    )
    
    # gameloop
    active = True
    while active:
        gameboard.draw(screen)
        for event in  pygame.event.get():
        
            # press w to move north
            # press a to move west
            # press s to move south
            # press d to move east
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                direction = Directions.NORTH
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                direction = Directions.WEST
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                direction = Directions.SOUTH
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                direction = Directions.EAST
        
            # press F11 to toggle fullscreen
            elif event.type == pygame.KEYUP and event.key == pygame.K_F11:
                toggle_fullscreen()
            
            # press ESC to quit
            elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                active = False
                
            # quit on all quit events
            elif event.type == pygame.QUIT:
                active = False
        
        snake.move(direction)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

