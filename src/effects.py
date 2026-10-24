import pygame
import math
import random
from src.utils import rotate_point

def draw_main_engine(screen, rocket):

    if not rocket.main_engine:
        return

    length = random.randint(35, 50)

    cx = rocket.x
    cy = rocket.y

    engine_y = rocket.image.get_height() // 2 - 5
    p1 = (cx - 8, cy + engine_y-18)
    p2 = (cx + 8, cy + 105-18)
    p3 = (cx, cy + 105 + length-18)
    
    p1 = rotate_point(cx, cy, *p1, rocket.angle)
    p2 = rotate_point(cx, cy, *p2, rocket.angle)
    p3 = rotate_point(cx, cy, *p3, rocket.angle)

    pygame.draw.polygon(
        screen,
        (255, 80, 0),
        [p1, p2, p3]
    )

    p1 = (cx - 4, cy + 105)
    p2 = (cx + 4, cy + 105)
    p3 = (cx, cy + 105 + length - 10)

    p1 = rotate_point(cx, cy, *p1, rocket.angle)
    p2 = rotate_point(cx, cy, *p2, rocket.angle)
    p3 = rotate_point(cx, cy, *p3, rocket.angle)

    pygame.draw.polygon(
        screen,
        (255, 220, 0),
        [p1, p2, p3]
    )

def draw_left_thruster(screen, rocket):

    if not rocket.left_thruster:
        return

    length = random.randint(18, 28)

    cx = rocket.x
    cy = rocket.y

    p1 = (cx - 15, cy - 85+50)
    p2 = (cx - 15 - length, cy - 78+50)
    p3 = (cx - 15 - length, cy - 92+50)

    p1 = rotate_point(cx, cy, *p1, rocket.angle)
    p2 = rotate_point(cx, cy, *p2, rocket.angle)
    p3 = rotate_point(cx, cy, *p3, rocket.angle)

    pygame.draw.polygon(
        screen,
        (220,220,220),
        [p1,p2,p3]
    )


def draw_right_thruster(screen, rocket):

    if not rocket.right_thruster:
        return

    length = random.randint(18,28)

    cx = rocket.x
    cy = rocket.y

    p1 = (cx + 15, cy - 85+50)
    p2 = (cx + 15 + length, cy - 78+50)
    p3 = (cx + 15 + length, cy - 92+50)

    p1 = rotate_point(cx, cy, *p1, rocket.angle)
    p2 = rotate_point(cx, cy, *p2, rocket.angle)
    p3 = rotate_point(cx, cy, *p3, rocket.angle)

    pygame.draw.polygon(
        screen,
        (220,220,220),
        [p1,p2,p3]
    )