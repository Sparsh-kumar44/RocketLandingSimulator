import numpy as np

from src.rocket import Rocket
from src.physics import update
from src.settings import *


class RocketEnv:
    def __init__(self):
        self.rocket = Rocket()
        self.previous_potential = 0.0
        self.previous_altitude = max(0.0, self.get_altitude())

    def get_altitude(self):
        ground_y = HEIGHT - GROUND_HEIGHT
        return ground_y - (self.rocket.y + ROCKET_HEIGHT / 2)

    def reset(self):
        self.rocket = Rocket()
        self.previous_altitude = max(0.0, self.get_altitude())
        self.previous_potential = self._potential()
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
        fuel_before = max(0.0, float(self.rocket.fuel))
        update(self.rocket)
        fuel_used = min(fuel_before, max(0.0, fuel_before - self.rocket.fuel))
        reward = self.calculate_reward(fuel_used)
        done = self.rocket.simulation_over
        return self.get_state(), reward, done

    def get_state(self):
        return np.array(
            [
                self.rocket.x / WIDTH,
                self.rocket.y / HEIGHT,
                self.rocket.vx / 10,
                self.rocket.vy / 10,
                self.rocket.angle / MAX_ANGLE,
                self.rocket.angular_velocity / 0.1,
                max(0.0, self.rocket.fuel) / MAX_FUEL,
            ],
            dtype=np.float32,
        )

    def _potential(self):
        """Higher potential means a state that is easier to land from safely."""
        rocket = self.rocket
        altitude = max(0.0, self.get_altitude())
        pad_center = PAD_X + PAD_WIDTH / 2

        distance_cost = min(abs(rocket.x - pad_center) / max(PAD_WIDTH / 2, 1), 2.0)
        horizontal_speed_cost = min(abs(rocket.vx) / max(MAX_LANDING_HSPEED, 0.1), 3.0)
        angle_cost = min(abs(rocket.angle) / max(MAX_LANDING_ANGLE, 0.01), 3.0)
        angular_speed_cost = min(abs(rocket.angular_velocity) / 0.04, 3.0)

        # Do not target near-zero vertical speed near the ground: that can teach
        # the agent to hover, spend all its fuel, and then fall.
        altitude_fraction = min(altitude / 350.0, 1.0)
        target_vy = 0.8 + 1.5 * altitude_fraction
        vertical_speed_cost = min(abs(rocket.vy - target_vy) / 3.0, 3.0)

        near_ground = max(0.0, min(1.0, (180.0 - altitude) / 180.0))
        vertical_weight = 0.35 + 0.85 * near_ground

        cost = (
            1.0 * distance_cost
            + 0.75 * horizontal_speed_cost
            + 0.70 * angle_cost
            + 0.20 * angular_speed_cost
            + vertical_weight * vertical_speed_cost
        )
        return -cost

    def calculate_reward(self, fuel_used=0.0):
        rocket = self.rocket
        pad_center = PAD_X + PAD_WIDTH / 2
        fuel_ratio = max(0.0, min(1.0, rocket.fuel / MAX_FUEL))

        # Successful landing is the main objective; remaining fuel is a bonus,
        # not a reason to avoid using the engine when braking is necessary.
        if rocket.landed:
            vertical_quality = 1.0 - min(abs(rocket.impact_vy) / MAX_LANDING_VSPEED, 1.0)
            horizontal_quality = 1.0 - min(abs(rocket.impact_vx) / MAX_LANDING_HSPEED, 1.0)
            angle_quality = 1.0 - min(abs(rocket.impact_angle) / MAX_LANDING_ANGLE, 1.0)
            center_quality = max(
                0.0,
                1.0 - abs(rocket.x - pad_center) / max(PAD_WIDTH / 2, 1),
            )
            return float(
                300.0
                + 30.0 * fuel_ratio
                + 6.0 * (vertical_quality + horizontal_quality + angle_quality)
                + 8.0 * center_quality
            )

        if rocket.crashed:
            vertical_excess = max(0.0, abs(rocket.impact_vy) / max(MAX_LANDING_VSPEED, 0.1) - 1.0)
            horizontal_excess = max(0.0, abs(rocket.impact_vx) / max(MAX_LANDING_HSPEED, 0.1) - 1.0)
            angle_excess = max(0.0, abs(rocket.impact_angle) / max(MAX_LANDING_ANGLE, 0.01) - 1.0)
            position_excess = max(
                0.0,
                abs(rocket.x - pad_center) / max(PAD_WIDTH / 2, 1) - 1.0,
            )

            # Much stronger penalty for hard impacts, particularly vertical speed.
            severity = (
                30.0 * min(vertical_excess, 5.0)
                + 14.0 * min(horizontal_excess, 5.0)
                + 12.0 * min(angle_excess, 5.0)
                + 18.0 * min(position_excess, 5.0)
            )
            fuel_waste_penalty = 12.0 * (1.0 - fuel_ratio)
            return float(-200.0 - severity - fuel_waste_penalty)

        current_altitude = max(0.0, self.get_altitude())
        current_potential = self._potential()
        altitude_progress = self.previous_altitude - current_altitude

        # Dense feedback for improving trajectory and making controlled progress
        # downwards. Progress is clipped so a dangerously fast fall is not rewarded
        # without limit; vertical-speed shaping penalizes excessive descent speed.
        reward = 2.0 * (current_potential - self.previous_potential)
        reward += 0.012 * float(np.clip(altitude_progress, -2.0, 2.0))
        reward -= 0.015  # small time cost, to discourage hovering indefinitely

        # Fuel was previously almost free (-0.004 per frame for the main engine).
        # Charge for actual fuel consumed and a small additional cost for engine use.
        reward -= 0.15 * fuel_used
        if rocket.main_engine and fuel_used > 0.0:
            reward -= 0.025
        if rocket.left_thruster and fuel_used > 0.0:
            reward -= 0.004
        if rocket.right_thruster and fuel_used > 0.0:
            reward -= 0.004

        self.previous_potential = current_potential
        self.previous_altitude = current_altitude
        return float(reward)

    def render(self, screen, clock):
        from src.renderer import draw_scene

        draw_scene(screen, self.rocket, clock)
