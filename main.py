import pygame
import random
import sys
import os

from gameboard import Gameboard, Markers
from snake import Snake, Directions

SCREEN_X = 0
SCREEN_Y = 0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FULLSCREEN = False
VSYNC = True

V_SPAWN_PCT = 0.5
H_SPAWN_PCT = 0.325

TICKRATE = 60
CRASH_DELAY = 50
SNAKE_MOVE_DELAY = 100  # milliseconds between snake moves (lower = faster)

CRASH_EVENT = pygame.USEREVENT
MOVE_EVENT = pygame.USEREVENT + 1

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
start_time = None
score = None

def init_gameboard():
    global gameboard
    global grid_rows
    global grid_cols

    # Fixed grid size for consistency across all devices
    grid_rows = 45
    grid_cols = 90

    gameboard = Gameboard(grid_rows, grid_cols)

def spawn_apple():
    """Spawn an apple at a random empty location on the gameboard"""
    empty_cells = []

    # Find all empty cells
    for row in range(grid_rows):
        for col in range(grid_cols):
            if gameboard.get_marker(row, col) == Markers.FLOOR:
                empty_cells.append((row, col))

    # Spawn apple at random empty cell
    if empty_cells:
        row, col = random.choice(empty_cells)
        gameboard.set_marker(row, col, Markers.APPLE)
        return True

    return False

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
    global start_time
    global score

    if event.key == pygame.K_r:
        game_over = False
        init_snake()
        start_time = pygame.time.get_ticks() / 1000.0  # Reset timer
        score = 0  # Reset score
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

def main():
    global screen
    global display_info

    global gameboard
    global snake_direc

    global active
    global game_over
    global start_time
    global score

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
    spawn_apple()  # Spawn initial apple
    clock = pygame.time.Clock()

    # Set up the movement timer
    pygame.time.set_timer(MOVE_EVENT, SNAKE_MOVE_DELAY)

    # Initialize timer and score
    start_time = pygame.time.get_ticks() / 1000.0
    score = 0

    active = True
    game_over = False
    while active:
        # Calculate elapsed time
        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - start_time if not game_over else elapsed_time

        gameboard.draw(screen, score, elapsed_time)

        for event in pygame.event.get():
            if event.type == CRASH_EVENT and snake_crashing:
                game_over = True
                snake.crash()
            elif event.type == MOVE_EVENT and not game_over:
                move_result = snake.move(snake_direc)
                if move_result:
                    snake_crashing = False
                    snake_direc = None

                    # Check if snake ate an apple
                    if move_result == "apple":
                        snake.grow(3)
                        score += 1  # Increment score
                        spawn_apple()  # Spawn new apple
                elif not snake_crashing:
                    snake_crashing = True
                    pygame.time.set_timer(CRASH_EVENT, CRASH_DELAY)
            elif event.type == pygame.KEYUP:
                on_keyup(event)
            elif event.type == pygame.KEYDOWN:
                on_keydown(event)
            elif event.type == pygame.QUIT:
                active = False

        clock.tick(TICKRATE)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())

