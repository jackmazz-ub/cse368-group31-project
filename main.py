import pygame
import random
import sys
import os
import time
from enum import IntEnum

from agent import Agent
from apple import Apple
from gameboard import Gameboard, CELL_WIDTH, CELL_HEIGHT, Markers
from snake import Snake, Directions

"""
=====================================================================================================
| CONSTANTS |
=============
"""

# display settings
SCREEN_TITLE = "Q-Snake"
SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = False
VSYNC = True

# snake spawn settings
V_SPAWN_PCT = 0.500 # percent of display to spawn vertically
H_SPAWN_PCT = 0.325 # percent of display to spawn horizontally
FOOTER_PCT = 0.1 # percent of display to use for the footer

# game timing
TICKRATE = 30 # display refresh rate
CRASH_DELAY = 50 # ms until snake crashes - gives player time to react

# events
CRASH_EVENT = pygame.USEREVENT # sent when snake crashes
TIMER_EVENT = pygame.USEREVENT+1 # send when a second elapses

# snake body settings
SNAKE_INIT_LENGTH = 4 # initial length
SNAKE_GROW_RATE = 5 # growth per apple

# fixed gameboard dimensions
FIXED_GRID_ROWS = 10 # set to None if using variable-sized grid (originally 45)
FIXED_GRID_COLS = 10 # set to None if using variable-sized grid (originally 94)

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

class Pointer:
    def __init__(self, value=None):
        self.value = value

# identifiers for the game modes
class Gamemodes(IntEnum):
    MANUAL = 0
    AUTO = 1

# map game modes to titles
gamemode_titles = {
    Gamemodes.MANUAL: "Manual",
    Gamemodes.AUTO: "Autonomous"
}

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

snake_ptr = Pointer() # reference to a snake
snake_direc = None # current snake direction
snake_crashing = None # bool if snake is currently crashing (waiting on a CRASH_EVENT for CRASH_DELAY ms)

apple_ptr = Pointer() # reference to an apple

agent = None

active = None # bool if the app is supposed to be running (app closes once this becomes false)
game_over = None # bool if the game has ended and pending a restart
gamemode = None # manual or autonomous modes

time = None # elapsed time

"""
=====================================================================================================
| INITIALIZATION |
==================
"""

def restart_game(mode=None):
    global game_over
    global gamemode
    
    # if the gamemode is not provided, use the current gamemode
    if mode is None:
        mode = gamemode
    
    # if the provided gamemode is the same as the current one, take no effect
    elif mode == gamemode:
        return

    game_over = False
    gamemode = mode
    init_snake()
    init_apple()
    init_timer()

def init_gameboard():
    global gameboard
    global grid_rows
    global grid_cols

    # use window dimensions if windowed, or display dimensions if fullscreen
    width = SCREEN_WIDTH if not FULLSCREEN else display_info.current_w
    height = SCREEN_HEIGHT if not FULLSCREEN else display_info.current_h
    
    # calculate number of rows, unless a fixed number is provided
    if FIXED_GRID_ROWS is None:
        grid_rows = int(height//CELL_HEIGHT * (1-FOOTER_PCT) - 2)
    else:
        grid_rows = FIXED_GRID_ROWS
    
    # calculate number of columns, unless a fixed number is provided
    if FIXED_GRID_COLS is None:
        grid_cols = int(width//CELL_WIDTH - 2)
    else:
        grid_cols = FIXED_GRID_COLS

    gameboard = Gameboard(grid_rows, grid_cols)

def init_snake():
    global snake_ptr
    global snake_crashing
    
    # remove current snake if it exists
    if snake_ptr.value is not None:
        snake_ptr.value.destroy()
        
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

    snake_ptr.value = Snake(gameboard, apple_ptr, length, row, col, direc)
    snake_crashing = False

def init_apple():
    global apple_ptr
    global game_over
    
    # remove current apple if it exists
    if apple_ptr.value is not None:
        apple_ptr.value.destroy()
    
    # find all possible positions (not occupied by anything)
    # if none are found, end the game (the player won)
    empty_cells = gameboard.list_cells(Markers.FLOOR)
    if len(empty_cells) == 0:
        game_over = True
        return

    # spawn apple at random empty cell
    cell = random.choice(empty_cells)
    apple_ptr.value = Apple(gameboard, cell.row, cell.col)

def init_agent():
    global agent
    agent = Agent(gameboard, snake_ptr, apple_ptr)

def init_timer():
    global time
    
    time = 0
    pygame.time.set_timer(TIMER_EVENT, 1000) # send TIMER_EVENT after 1 second

"""
=====================================================================================================
| KEY EVENT HANDLERS |
======================
"""

def on_keyup(event):
    global active
    
    ctrl_active = (event.mod & pygame.KMOD_CTRL) or (event.mod & pygame.KMOD_META)
    
    # restart the game on 'R' key-release
    if ctrl_active and event.key == pygame.K_r:
        restart_game()
        
    # start manual game on 'Ctrl-M' or 'F1' key-release
    elif (ctrl_active and event.key == pygame.K_m) or event.key == pygame.K_F1:
        restart_game(Gamemodes.MANUAL)
        
    # start autonomous game on 'Ctrl-A' or 'F2' key-release
    elif (ctrl_active and event.key == pygame.K_a) or event.key == pygame.K_F2:
        restart_game(Gamemodes.AUTO)
        
    # toggle fullscreen on 'F11' key-release
    elif event.key == pygame.K_F11:
        toggle_fullscreen()
        
    # exit game on 'ESC' key-release
    elif event.key == pygame.K_ESCAPE:
        active = False
        
    # grow snake on 'G' key-release (use only for development purposes)
    # elif event.key == pygame.K_g and snake is not None:
    #     snake_ptr.value.grow(5)

def on_keydown(event):
    global snake_direc
    
    if gamemode == Gamemodes.MANUAL:
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
    global snake_crashing
    
    global active
    global game_over
    global gamemode
    
    global time
    
    # initialize game components
    init_gameboard()
    init_snake()
    init_apple()
    snake_crashing = False
    
    # initialize autonomous player
    init_agent()
    
    # reset snake and apple
    init_snake()
    init_apple()
    
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (SCREEN_X, SCREEN_Y)

    pygame.init()
    display_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.FULLSCREEN if FULLSCREEN else pygame.RESIZABLE,
        vsync=VSYNC,
    )
    
    # set the window title
    pygame.display.set_caption(SCREEN_TITLE)
    
    init_timer()
    clock = pygame.time.Clock()
    
    active = True
    game_over = False
    gamemode = Gamemodes.AUTO
    
    # mainloop
    while active:        
        gameboard.draw(
            screen, 
            gamemode_titles[gamemode],
            snake_ptr.value.length, 
            time
        )
        
        # activate events
        for event in pygame.event.get():
            if event.type == CRASH_EVENT and snake_crashing:
                game_over = True
                snake_ptr.value.crash()
                if gamemode == Gamemodes.AUTO:
                    restart_game()
                
            elif event.type == TIMER_EVENT and not game_over:
                time+=1
            elif event.type == pygame.KEYUP:
                on_keyup(event)
            elif event.type == pygame.KEYDOWN:
                on_keydown(event)
            elif event.type == pygame.QUIT:
                active = False
        
        # resume game (if it's not over)
        if not game_over:
        
            # move the snake
            # success: bool if snake moved successfully (did not hit a wall or body)
            # ate_apple: bool if snake moved on top of an apple
            if gamemode == Gamemodes.MANUAL:
                success, ate_apple = snake_ptr.value.move(snake_direc)
            elif gamemode == Gamemodes.AUTO:
                success, ate_apple = agent.move()
            
            # if the snake moved successfully ...
            if success:
                # if apple was eaten, grow, and spawn a new one
                if ate_apple:
                    snake_ptr.value.grow(SNAKE_GROW_RATE)
                    init_apple()
                
                snake_crashing = False # snake is not crashing (since the move was successfull)
                snake_direc = None # have snake move in default direction (the direction of the head)
            
            # if the snake hit a wall (and the CRASH_EVENT has not yet been queued)
            elif not snake_crashing:
                snake_crashing = True # the snake is now crashing
                pygame.time.set_timer(CRASH_EVENT, CRASH_DELAY) # activate the CRASH_EVENT in CRASH_DELAY ms
        
        # delay the mainloop by TICKRATE
        clock.tick(TICKRATE)
    
    agent.close()
    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

