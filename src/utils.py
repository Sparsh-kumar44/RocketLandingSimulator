import math

def rotate_point(cx, cy, x, y, angle):

    dx = x - cx
    dy = y - cy

    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)

    return (
        cx + rx,
        cy + ry
    )