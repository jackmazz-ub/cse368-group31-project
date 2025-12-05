import pygame
import random
import sys
import os
import time
from enum import IntEnum

from agent import Agent, agent_ptr
from apple import Apple, apple_ptr
from gameboard import Gameboard, gameboard_ptr
from snake import Directions, Snake, snake_ptr

"""
=====================================================================================================
| CONSTANTS |
=============
"""

WINDOW_TITLE = "Q-Snake"
WINDOW_X = 0
WINDOW_Y = 0
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
VSYNC = True

V_SPAWN_PCT = 0.500 # percent of display to spawn vertically
H_SPAWN_PCT = 0.325 # percent of display to spawn horizontally
FOOTER_PCT = 0.1 # percent of display to use for the footer

TICKRATE = 10 # display refresh rate
CRASH_DELAY = 50 # ms until snake crashes (gives player time to react)

CRASH_EVENT = pygame.USEREVENT # sent when snake crashes
TIMER_EVENT = pygame.USEREVENT+1 # send when a second elapses

"""
=====================================================================================================
| HELPER CLASSES |
==================
"""

class Gamemodes(IntEnum):
    MANUAL = 0
    AUTO = 1

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
running = True

game_over = False
gamemode = Gamemodes.AUTO
time = 0

crash_pending = False

"""
=====================================================================================================
| KEY EVENT HANDLERS |
======================
"""

def on_keyup(event):
    global running
    
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
        running = False
        
    # grow snake on 'G' key-release (use only for development purposes)
    # elif event.key == pygame.K_g:
    #     snake_ptr.value.grow(5)

def on_keydown(event):
    if gamemode == Gamemodes.MANUAL:
        # change snake direction to North on 'W' or 'Up-Arrow' key-press
        if event.key == pygame.K_UP or event.key == pygame.K_w:
            snake_ptr.value.turn(Directions.NORTH)
            
        # change snake direction to West on 'A' or 'Left-Arrow' key-press
        elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
            snake_ptr.value.turn(Directions.WEST)
            
        # change snake direction to South on 'S' or 'Down-Arrow' key-press
        elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
            snake_ptr.value.turn(Directions.SOUTH)
            
        # change snake direction to East on 'D' or 'Right-Arrow' key-press
        elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
            snake_ptr.value.turn(Directions.EAST)

"""
=====================================================================================================
| INITIALIZATION |
==================
"""

def restart_game(mode=None):
    global game_over
    global gamemode
    global time
    
    if mode is None:
        mode = gamemode
    
    # if the provided gamemode is the same as the current one, cancel the restart
    elif mode == gamemode:
        return
        
    game_over = False
    gamemode = mode
    time = 0
    
    snake_ptr.value.place()
    apple_ptr.value.place()
    
    print("\n")

"""
=====================================================================================================
| MAIN |
========
"""

def main(argv):
    global screen
    global display_info
    global running
    
    global game_over
    global gamemode
    global time
    
    global crash_pending
    
    # initialize game components
    gameboard_ptr.value = Gameboard()
    snake_ptr.value = Snake()
    apple_ptr.value = Apple()
    
    # initialize autonomous player
    agent_ptr.value = Agent()
    
    # initialize the display
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (WINDOW_X, WINDOW_Y)
    pygame.init()
    display_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,
        vsync=VSYNC,
    )
    pygame.display.set_caption(WINDOW_TITLE)
    
    pygame.time.set_timer(TIMER_EVENT, 1000) # send TIMER_EVENT after 1 second
    clock = pygame.time.Clock()
    
    # mainloop
    while running:
        gameboard_ptr.value.draw(
            screen, 
            gamemode_titles[gamemode],
            snake_ptr.value.length, 
            time,
        )
        
        # activate events
        for event in pygame.event.get():
            if event.type == CRASH_EVENT and crash_pending:
                game_over = True
                snake_ptr.value.crash()
                if gamemode == Gamemodes.AUTO:
                    restart_game()
                
            elif event.type == TIMER_EVENT and not game_over:
                time += 1
            elif event.type == pygame.KEYUP:
                on_keyup(event)
            elif event.type == pygame.KEYDOWN:
                on_keydown(event)
            elif event.type == pygame.QUIT:
                running = False
        
        if not game_over:
            if gamemode == Gamemodes.MANUAL:
                snake_ptr.value.move()
            elif gamemode == Gamemodes.AUTO:
                agent_ptr.value.move()
            
            # if the snake moved successfully, cancel any crash events
            if not snake_ptr.value.crashing:
                crash_pending = False
                if not apple_ptr.value.placed:
                    game_over = True
            
            # if the snake hit a wall (and the CRASH_EVENT has not yet been queued)
            elif not crash_pending:
                pygame.time.set_timer(CRASH_EVENT, CRASH_DELAY) # activate the CRASH_EVENT in CRASH_DELAY ms
                crash_pending = True
        
        # delay the mainloop by TICKRATE
        clock.tick(TICKRATE)
    
    agent_ptr.value.close()
    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

