import math
from src.settings import *
from src.utils import rotate_point


def update(rocket):

    if rocket.simulation_over:
        return

    rocket.ax = 0
    rocket.ay = GRAVITY

    if rocket.fuel <= 0:

        rocket.fuel = 0

        rocket.main_engine = False
        rocket.left_thruster = False
        rocket.right_thruster = False

    if rocket.main_engine:

        thrust_x = MAIN_THRUST * math.sin(rocket.angle)
        thrust_y = MAIN_THRUST * math.cos(rocket.angle)
        rocket.fuel -= MAIN_ENGINE_FUEL
        rocket.ax += thrust_x
        rocket.ay -= thrust_y


    rocket.angular_acceleration = 0

    if rocket.left_thruster and rocket.fuel > 0:

        rocket.angular_acceleration = ROTATION_SPEED
        rocket.fuel -= RCS_FUEL

    if rocket.right_thruster and rocket.fuel > 0:

        rocket.angular_acceleration = -ROTATION_SPEED
        rocket.fuel -= RCS_FUEL

    rocket.angular_velocity += rocket.angular_acceleration

    rocket.angular_velocity *= ROTATION_DAMPING

    rocket.angle += rocket.angular_velocity


    if rocket.angle > ANGLE_CLAMP:
        rocket.angle = ANGLE_CLAMP
        rocket.angular_velocity = 0

    if rocket.angle < -ANGLE_CLAMP:
        rocket.angle = -ANGLE_CLAMP
        rocket.angular_velocity = 0

    rocket.vx += rocket.ax
    rocket.vy += rocket.ay
    rocket.vx *= 0.995
    rocket.x += rocket.vx
    rocket.y += rocket.vy

    half_width = rocket.image.get_width() / 2

    if rocket.x < half_width:
        rocket.x = half_width
        rocket.vx = 0

    if rocket.x > WIDTH - half_width:
        rocket.x = WIDTH - half_width
        rocket.vx = 0

    ground_y = HEIGHT - GROUND_HEIGHT
    rocket_bottom = rocket.y + ROCKET_HEIGHT / 2

    if rocket_bottom >= ground_y:
        rocket.impact_vx = rocket.vx
        rocket.impact_vy = rocket.vy
        rocket.impact_angle = rocket.angle
        rocket.y = ground_y - ROCKET_HEIGHT / 2

        left_leg = rotate_point(
            rocket.x,
            rocket.y,
            rocket.x + LEFT_LEG[0],
            rocket.y + LEFT_LEG[1],
            rocket.angle
        )

        right_leg = rotate_point(
            rocket.x,
            rocket.y,
            rocket.x + RIGHT_LEG[0],
            rocket.y + RIGHT_LEG[1],
            rocket.angle
        )

        left_x = left_leg[0]
        right_x = right_leg[0]

        good_vertical = abs(rocket.vy) <= MAX_LANDING_VSPEED
        good_horizontal = abs(rocket.vx) <= MAX_LANDING_HSPEED
        good_angle = abs(rocket.angle) <= MAX_LANDING_ANGLE

        on_pad = (
            left_x >= PAD_LEFT and
            right_x <= PAD_RIGHT
        )

        if (
            good_vertical and
            good_horizontal and
            good_angle and
            on_pad
        ):
            rocket.landed = True
        else:
            rocket.crashed = True

        rocket.simulation_over = True

        rocket.vx = 0
        rocket.vy = 0
        rocket.ax = 0
        rocket.ay = 0

        rocket.angular_velocity = 0
        rocket.angular_acceleration = 0