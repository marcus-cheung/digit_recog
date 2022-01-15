import pyglet
from pyglet import shapes
from pyglet.window import key
from pickle import load
import numpy as np
from classes import *

TIME_INCREMENT = 1 / 30

SCALE = 30
WIDTH = 28
HEIGHT = 28
WHITE = 255, 255, 255

mouse_pos = [0, 0]
pen_eraser = True
network = None
with open("hundred.pickle", "rb") as file:
    network = load(file)

class Canvas():
    def __init__(self) -> None:
        self.grid = [[0] * 28 for i in range(28)]
        self.result = None
        self.confidence = 0

    def clear(self):
        self.grid = [[0] * 28 for i in range(28)]


game_window = pyglet.window.Window(SCALE * WIDTH, SCALE * HEIGHT, caption='digit_recog')

canvas = Canvas()

def update(dt):
    pass 

@game_window.event
def on_draw():
    batch = pyglet.graphics.Batch()
    game_window.clear()
    blocks = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if canvas.grid[i][j] == 1:
                block = shapes.Rectangle(SCALE * (i + 0.05), SCALE * (j + 0.05), SCALE * 0.90, SCALE * 0.90,
                                         color=WHITE, batch=batch)
                blocks.append(block)
    batch.draw()
    if canvas.result is not None:
        result_text = pyglet.text.Label(f'Did you write a {str(canvas.result)}({canvas.confidence})?',
                                        font_name='Times New Roman',
                                        font_size=24,
                                        color = (255,0,0,255),
                                        x=WIDTH * SCALE // 2, y=HEIGHT * SCALE // 2,
                                        anchor_x='center', anchor_y='center')
        result_text.draw()


@game_window.event
def on_mouse_motion(x, y, dx, dy):
    mouse_pos[0] = x
    mouse_pos[1] = y

@game_window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    mouse_update(x,y)

@game_window.event
def on_mouse_press(x, y, button, modifiers):
    mouse_update(x,y)

def mouse_update(x,y):
    closest_point = get_closest_point(x, y)
    canvas.grid[closest_point[0]][closest_point[1]] = pen_eraser

@game_window.event
def on_key_release(symbol, modifiers): 
    global pen_eraser
    if symbol == key.Z:
        canvas.clear()
        canvas.result = None
    if symbol == key.ENTER or symbol == key.RETURN:
        canvas.result, canvas.confidence = network.run(list(np.ravel(np.array(canvas.grid), order = "C")))
    if symbol == key.SPACE:
        pen_eraser = 1 - pen_eraser
        

def get_closest_point(x, y):
    adjusted_position = x / SCALE - 0.5, y / SCALE - 0.5
    closest_point = [round(adjusted_position[i]) for i in range(2)]
    return closest_point


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update, TIME_INCREMENT)
    pyglet.app.run()