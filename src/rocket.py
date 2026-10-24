import pygame
from src.settings import *
import random
import math
from pathlib import Path
class Rocket:

    def __init__(self):

        self.image = pygame.image.load("assets/rocket.png")
        original_width = self.image.get_width()
        original_height = self.image.get_height()

        scale = ROCKET_HEIGHT / original_height
        new_width = int(original_width * scale)

        self.image = pygame.transform.smoothscale(
            self.image,
            (new_width, ROCKET_HEIGHT)
        )

        self.x = random.uniform(
            PAD_X - 50,
            PAD_X + PAD_WIDTH + 50
        )

        self.y = random.uniform(
            80,
            220
        )


        self.vx = random.uniform(
            -1.0,
            1.0
        )

        self.vy = random.uniform(
            -1.0,
            5.0
        )

        self.angle = math.radians(
            random.uniform(-15, 15)
        )

        self.angular_velocity = random.uniform(
            -0.03,
            0.03
)
        self.ax = 0.0
        self.ay = 0.0
        self.main_engine = False
        self.left_thruster = False
        self.right_thruster = False
        self.landed = False
        self.crashed = False
        self.simulation_over = False
        self.impact_vx = 0
        self.impact_vy = 0
        self.impact_angle = 0
        self.fuel = MAX_FUEL