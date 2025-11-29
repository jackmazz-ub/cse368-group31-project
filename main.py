import pygame
import random
import sys
import os
import time

from gameboard import Gameboard, CELL_WIDTH, CELL_HEIGHT, Markers
from snake import Snake, Directions

"""
=====================================================================================================
| CONSTANTS |
=============
"""

# display settings
SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = True
VSYNC = True

# snake spawn settings
V_SPAWN_PCT = 0.500 # percent of display to spawn vertically
H_SPAWN_PCT = 0.325 # percent of display to spawn horizontally
FOOTER_PCT = 0.1 # percent of display to use for the footer

# game timing
TICKRATE = 30 # display refresh rate
CRASH_DELAY = 50 # ms until snake crashes - gives player time to react

# events
CRASH_EVENT = pygame.USEREVENT

# snake body settings
SNAKE_INIT_LENGTH = 4 # initial length
SNAKE_GROW_RATE = 5 # growth per apple

"""
=====================================================================================================
| GLOBAL VARIABLES |
====================
"""

screen = None
display_info = None

gameboard = None
grid_rows = None # gameboard number of rows
grid_cols = None # gameboard number of columns

snake = None
snake_direc = None # current snake direction
snake_crashing = None # bool if snake is currently crashing (waiting on a CRASH_EVENT for CRASH_DELAY ms)

active = None # bool if the app is supposed to be running (app closes once this becomes false)
game_over = None # bool if the game has ended and pending a restart

"""
=====================================================================================================
| INITIALIZATION |
==================
"""

def init_gameboard():
    global gameboard
    global grid_rows
    global grid_cols

    # use window dimensions if windowed, or display dimensions if fullscreen
    width = SCREEN_WIDTH if not FULLSCREEN else display_info.current_w
    height = SCREEN_HEIGHT if not FULLSCREEN else display_info.current_h
    
    # calculate number of rows and columns for the gameboard
    grid_rows = int(height//CELL_HEIGHT * (1-FOOTER_PCT) - 2)
    grid_cols = int(width//CELL_WIDTH - 2)

    gameboard = Gameboard(grid_rows, grid_cols)
    
    spawn_apple()

def init_snake():
    global snake
    global snake_crashing
    
    # remove current snake if it exists
    if snake is not None:
        snake.destroy()
        
    # choose starting head location
    # row & col chosen s.t. the snake's body will not spawn OOB
    length = SNAKE_INIT_LENGTH
    row = random.randint(length, grid_rows-length-1)
    col = random.randint(length, grid_cols-length-1)

    """
    Spawn Sectors
    ---------- ---------- ----------
    |        | | NORTH  | |        |
    |        | | SECTOR | |        |
    | WEST   | ---------- | EAST   |
    | SECTOR | ---------- | SECTOR |
    |        | | SOUTH  | |        |
    |        | | SECTOR | |        |
    ---------- ---------- ----------
    
    These sectors define areas of the board which are 'too close to the edge'.
    Snakes which spawn in these areas will begin facing away from the edge to
    reduce unfair spawns.
    """
    
    v_spawn = int(grid_rows * V_SPAWN_PCT) # defines the width of the East and West sectors
    h_spawn = int(grid_cols * H_SPAWN_PCT) # defines the height of the North and South sectors
    
    direc = None
    # start facing East if spawning in the West Sector
    if col < h_spawn:
        direc = Directions.EAST
    # start facing West if spawning in the East Sector
    elif col > grid_cols - h_spawn:
        direc = Directions.WEST
    # start facing South if spawning in the North Sector
    elif row < v_spawn:
        direc = Directions.SOUTH
    # start facing North if spawning in the South Sector
    elif row > grid_rows - v_spawn:
        direc = Directions.NORTH
    # start in a random direction if not spawning in any sectors
    else:
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1

    snake = Snake(gameboard, length, row, col, direc)
    snake_crashing = False

def spawn_apple():
    # spawn an apple at a random empty location on the gameboard
    empty_cells = []

    # find all empty cells
    for row in range(grid_rows):
        for col in range(grid_cols):
            if gameboard.get_marker(row, col) == Markers.FLOOR:
                empty_cells.append((row, col))

    # spawn apple at random empty cell
    if empty_cells:
        row, col = random.choice(empty_cells)
        gameboard.set_marker(row, col, Markers.APPLE)
        return True

    return False

"""
=====================================================================================================
| KEY EVENT HANDLERS |
======================
"""

def on_keyup(event):
    global active
    global game_over
    
    # restart the game on 'R' key-release
    if event.key == pygame.K_r:
        game_over = False
        init_snake()
    
    # toggle fullscreen on 'F11' key-release
    elif event.key == pygame.K_F11:
        toggle_fullscreen()
    
    # exit game on 'ESC' key-release
    elif event.key == pygame.K_ESCAPE:
        active = False

def on_keydown(event):
    global snake_direc
    
    # change snake direction to North on 'W' or 'Up-Arrow' key-press
    if event.key == pygame.K_UP or event.key == pygame.K_w:
        snake_direc = Directions.NORTH
    
    # change snake direction to West on 'A' or 'Left-Arrow' key-press
    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
        snake_direc = Directions.WEST
    
    # change snake direction to South on 'S' or 'Down-Arrow' key-press
    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
        snake_direc = Directions.SOUTH
    
    # change snake direction to East on 'D' or 'Right-Arrow' key-press
    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
        snake_direc = Directions.EAST

"""
=====================================================================================================
| DISPLAY FUNCTIONS |
=====================
"""

def toggle_fullscreen():
    global screen
    
    # enable fullscreen if currently disabled, and vice-versa
    fullscreen = screen.get_flags() & pygame.FULLSCREEN != 0
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if not fullscreen else pygame.RESIZABLE,
        vsync=VSYNC,
    )

"""
=====================================================================================================
| MAIN |
========
"""

def main(argv):
    global screen
    global display_info
    
    global gameboard
    global snake_direc
    
    global active
    global game_over
    
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
    
    # initialize timer
    start_time = pygame.time.get_ticks() / 1000.0
    
    active = True
    game_over = False
    
    # mainloop
    while active:
    
        # calculate elapsed time
        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - start_time if not game_over else elapsed_time

        gameboard.draw(screen, snake.length, elapsed_time)
        
        # activate events
        for event in pygame.event.get():
            if event.type == CRASH_EVENT and snake_crashing:
                game_over = True
                snake.crash()
            elif event.type == pygame.KEYUP:
                on_keyup(event)
            elif event.type == pygame.KEYDOWN:
                on_keydown(event)
            elif event.type == pygame.QUIT:
                active = False
        
        # resume game (if it's not over)
        if not game_over:
        
            # move the snake
            # success: bool if snake moved successfully (did not hit a wall or body)(
            # ate_apple: bool if snake moved on top of an apple
            success, ate_apple = snake.move(snake_direc)
            
            # if the snake moved successfully ...
            if success:
            
                # if apple was eaten, grow, and spawn a new one
                if ate_apple:
                    snake.grow(SNAKE_GROW_RATE)
                    spawn_apple()
                
                snake_crashing = False # snake is not crashing (since the move was successfull)
                snake_direc = None # have snake move in default direction (the direction of the head)
            
            # if the snake hit a wall (and the CRASH_EVENT has not yet been queued)
            elif not snake_crashing:
                snake_crashing = True # the snake is now crashing
                pygame.time.set_timer(CRASH_EVENT, CRASH_DELAY) # activate the CRASH_EVENT in CRASH_DELAY ms
        
        # delay the mainloop by TICKRATE
        clock.tick(TICKRATE)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

