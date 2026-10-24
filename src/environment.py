import math

import numpy as np

from src.rocket import Rocket
from src.physics import update
from src.settings import *


SAFE_BRAKE = 0.12
SAFE_TOUCHDOWN_SPEED = 1.0
MIN_SPAWN_ALTITUDE = 20.0

SHAPING_SCALE = 10.0
TIME_COST = 0.02
FUEL_COST = 0.12  #
LANDING_FUEL_BONUS = 40.0  

LANDING_REWARD = 100.0
CRASH_PENALTY = 100.0


class RocketEnv:
    def __init__(self, difficulty=1.0):
        self.difficulty = difficulty
        self.rocket = self._spawn_rocket()
        self.previous_shaping = self._shaping()

    def _spawn_rocket(self):
        """Create a Rocket, then pull its random start state toward an easy one.

        difficulty 1.0 keeps Rocket's own spawn ranges unchanged. Lower values
        start it closer to the pad centre, nearer the ground, slower, more upright
        and with less spin; training starts easy and works up to 1.0.
        """
        rocket = Rocket()
        d = max(0.0, min(1.0, self.difficulty))
        if d >= 1.0:
            return rocket

        pad_center = PAD_X + PAD_WIDTH / 2
        rocket.x = pad_center + (rocket.x - pad_center) * d
        rocket.vx *= d
        rocket.vy *= d
        rocket.angle *= d
        rocket.angular_velocity *= d

        ground_center_y = HEIGHT - GROUND_HEIGHT - ROCKET_HEIGHT / 2
        altitude = (ground_center_y - rocket.y) * d
        rocket.y = ground_center_y - max(MIN_SPAWN_ALTITUDE, altitude)
        return rocket

    def get_altitude(self):
        ground_y = HEIGHT - GROUND_HEIGHT
        return ground_y - (self.rocket.y + ROCKET_HEIGHT / 2)

    def reset(self, difficulty=None):
        if difficulty is not None:
            self.difficulty = difficulty
        self.rocket = self._spawn_rocket()
        self.previous_shaping = self._shaping()
        return self.get_state()

    def apply_action(self, action):
        self.rocket.main_engine = False
        self.rocket.left_thruster = False
        self.rocket.right_thruster = False

        if action == 1:
            self.rocket.main_engine = True
        elif action == 2:
            self.rocket.left_thruster = True
        elif action == 3:
            self.rocket.right_thruster = True
        elif action == 4:
            self.rocket.main_engine = True
            self.rocket.left_thruster = True
        elif action == 5:
            self.rocket.main_engine = True
            self.rocket.right_thruster = True

    def step(self, action):
        self.apply_action(action)
        fuel_before = max(0.0, self.rocket.fuel)
        update(self.rocket)
        fuel_used = max(0.0, fuel_before - max(0.0, self.rocket.fuel))
        reward = self.calculate_reward(fuel_used)
        done = self.rocket.simulation_over
        return self.get_state(), reward, done

    def get_state(self):
        pad_center = PAD_X + PAD_WIDTH / 2
        return np.array(
            [
                (self.rocket.x - pad_center) / (PAD_WIDTH / 2),
                self.get_altitude() / 400.0,
                self.rocket.vx / 2.0,
                self.rocket.vy / 5.0,
                self.rocket.angle / math.radians(20),
                self.rocket.angular_velocity / math.radians(1.0),
                max(0.0, self.rocket.fuel) / MAX_FUEL,
            ],
            dtype=np.float32,
        )

    def _shaping(self):

        rocket = self.rocket
        altitude = max(0.0, self.get_altitude())
        pad_center = PAD_X + PAD_WIDTH / 2

        safe_vy = SAFE_TOUCHDOWN_SPEED + math.sqrt(2.0 * SAFE_BRAKE * altitude)
        overspeed = max(0.0, rocket.vy - safe_vy)
        climbing = max(0.0, -rocket.vy)

        near_ground = max(0.0, min(1.0, (200.0 - altitude) / 200.0))
        attitude_weight = 0.5 + 0.5 * near_ground

        cost = (
            1.0 * altitude / 100.0
            + 2.0 * abs(rocket.x - pad_center) / (PAD_WIDTH / 2)
            + 2.0 * overspeed
            + 0.5 * climbing
            + attitude_weight * 0.5 * abs(rocket.vx) / MAX_LANDING_HSPEED
            + attitude_weight * 0.5 * abs(rocket.angle) / MAX_LANDING_ANGLE
            + 0.3 * abs(rocket.angular_velocity) / 0.02
        )
        return -cost

    def calculate_reward(self, fuel_used=0.0):
        rocket = self.rocket
        pad_center = PAD_X + PAD_WIDTH / 2
        fuel_ratio = max(0.0, min(1.0, rocket.fuel / MAX_FUEL))

        if rocket.landed:
            vertical_quality = 1.0 - min(abs(rocket.impact_vy) / MAX_LANDING_VSPEED, 1.0)
            angle_quality = 1.0 - min(abs(rocket.impact_angle) / MAX_LANDING_ANGLE, 1.0)
            center_quality = max(
                0.0, 1.0 - abs(rocket.x - pad_center) / (PAD_WIDTH / 2)
            )
            return (
                LANDING_REWARD
                + LANDING_FUEL_BONUS * fuel_ratio
                + 5.0 * (vertical_quality + angle_quality + center_quality)
            )

        if rocket.crashed:
         
            vertical_excess = max(0.0, abs(rocket.impact_vy) / MAX_LANDING_VSPEED - 1.0)
            horizontal_excess = max(0.0, abs(rocket.impact_vx) / MAX_LANDING_HSPEED - 1.0)
            angle_excess = max(0.0, abs(rocket.impact_angle) / MAX_LANDING_ANGLE - 1.0)
            position_excess = max(
                0.0, abs(rocket.x - pad_center) / (PAD_WIDTH / 2) - 1.0
            )
            severity = (
                10.0 * min(vertical_excess, 4.0)
                + 5.0 * min(horizontal_excess, 4.0)
                + 5.0 * min(angle_excess, 4.0)
                + 10.0 * min(position_excess, 2.0)
            )
            return -CRASH_PENALTY - severity

        shaping = self._shaping()
        reward = SHAPING_SCALE * (shaping - self.previous_shaping)
        self.previous_shaping = shaping

        reward -= TIME_COST
        reward -= FUEL_COST * fuel_used
        return reward

    def render(self, screen, clock):
        from src.renderer import draw_scene

        draw_scene(screen, self.rocket, clock)
