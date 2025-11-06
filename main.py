import pygame
import random
import sys
import os
import time

from gameboard import Gameboard, CELL_WIDTH, CELL_HEIGHT
from snake import Snake, Directions

SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = False
VSYNC = True

ROW_SECT_PRCT = 0.5
COL_SECT_PRCT = 0.325
ROW_MARGIN_PRCT = 0.05
COL_MARGIN_PRCT = 0.05
FOOTER_PRCT = 0.1

TICK_RATE = 60

SNAKE_LENGTH = 4

screen = None
display_info = None

gameboard = None
grid_rows = None
grid_cols = None

snake = None
snake_direc = None

active = None

def init_gameboard():
    global gameboard
    global grid_rows
    global grid_cols
    
    grid_rows = int(display_info.current_h//CELL_HEIGHT * (1-FOOTER_PRCT)) - 2
    grid_cols = display_info.current_w//CELL_WIDTH - 2
    
    gameboard = Gameboard(grid_rows, grid_cols)

def init_snake():
    global snake
    
    if snake is not None:
        snake.destroy()
    
    row = random.randint(SNAKE_LENGTH, grid_rows-SNAKE_LENGTH-1)
    col = random.randint(SNAKE_LENGTH, grid_cols-SNAKE_LENGTH-1)
    
    row_sect = int(grid_rows * ROW_SECT_PRCT)
    col_sect = int(grid_cols * COL_SECT_PRCT)
    
    direc = None
    if col < col_sect:
        direc = Directions.EAST
    elif col > grid_cols - col_sect:
        direc = Directions.WEST
    elif row < row_sect:
        direc = Directions.SOUTH
    elif row > grid_rows - row_sect:
        direc = Directions.NORTH
    else:
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
    
    snake = Snake(gameboard, SNAKE_LENGTH, row, col, direc)

def on_keyup(event):
    global active
    
    if event.key == pygame.K_r:
        init_snake()
    elif event.key == pygame.K_F11:
        toggle_fullscreen()
    elif event.key == pygame.K_ESCAPE:
        active = False

def on_keydown(event):
    global snake_direc

    if event.key == pygame.K_UP or event.key == pygame.K_w:
        snake_direc = Directions.NORTH
    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
        snake_direc = Directions.WEST
    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
        snake_direc = Directions.SOUTH
    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
        snake_direc = Directions.EAST

def toggle_fullscreen():
    global screen
    
    fullscreen = screen.get_flags() & pygame.FULLSCREEN != 0
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if not fullscreen else pygame.RESIZABLE,
        vsync=VSYNC,
    )

def main(argv):
    global screen
    global display_info
    global gameboard
    global snake_direc
    
    global active
    
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (SCREEN_X, SCREEN_Y)
    
    pygame.init()
    display_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if FULLSCREEN else pygame.RESIZABLE,
        vsync=VSYNC,
    )
    
    init_gameboard()
    init_snake()
    clock = pygame.time.Clock()
    
    active = True
    while active:
        gameboard.draw(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                on_keyup(event)
            elif event.type == pygame.KEYDOWN:
                on_keydown(event)
            elif event.type == pygame.QUIT:
                active = False
        
        snake.move(snake_direc)
        snake_direc = None
        
        clock.tick(TICK_RATE)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

