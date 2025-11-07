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

V_SPAWN_PCT = 0.5
H_SPAWN_PCT = 0.325
FOOTER_PCT = 0.1

TICKRATE = 60
CRASH_DELAY = 50

CRASH_EVENT = pygame.USEREVENT

SNAKE_INIT_LENGTH = 4

screen = None
display_info = None

gameboard = None
grid_rows = None
grid_cols = None

snake = None
snake_direc = None
snake_crashing = None

active = None
game_over = None

def init_gameboard():
    global gameboard
    global grid_rows
    global grid_cols
    
    grid_rows = int(display_info.current_h//CELL_HEIGHT * (1-FOOTER_PCT)) - 2
    grid_cols = display_info.current_w//CELL_WIDTH - 2
    
    gameboard = Gameboard(grid_rows, grid_cols)

def init_snake():
    global snake
    global snake_crashing
    
    if snake is not None:
        snake.destroy()
    
    length = SNAKE_INIT_LENGTH
    row = random.randint(length, grid_rows-length-1)
    col = random.randint(length, grid_cols-length-1)
    
    v_spawn = int(grid_rows * V_SPAWN_PCT)
    h_spawn = int(grid_cols * H_SPAWN_PCT)
    
    direc = None
    if col < h_spawn:
        direc = Directions.EAST
    elif col > grid_cols - h_spawn:
        direc = Directions.WEST
    elif row < v_spawn:
        direc = Directions.SOUTH
    elif row > grid_rows - v_spawn:
        direc = Directions.NORTH
    else:
        direc = random.randint(1, 2)
        if (random.random() <= 0.5):
            direc *= -1
    
    snake = Snake(gameboard, length, row, col, direc)
    snake_crashing = False

def on_keyup(event):
    global active
    global game_over
    
    if event.key == pygame.K_r:
        game_over = False
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
        
    elif event.key == pygame.K_g: # dev purposes
        snake.grow(10)

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
    
    active = True
    game_over = False
    while active:
        gameboard.draw(screen)
        
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
        
        if not game_over:
            if snake.move(snake_direc):
                snake_crashing = False
                snake_direc = None
            elif not snake_crashing:
                snake_crashing = True
                pygame.time.set_timer(CRASH_EVENT, CRASH_DELAY)
        
        clock.tick(TICKRATE)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

