import pygame
import math

from src.settings import *
from src.utils import rotate_point
from src.effects import (
    draw_main_engine,
    draw_left_thruster,
    draw_right_thruster
)


def draw_scene(screen, rocket, clock):

    font = pygame.font.SysFont("Consolas", 22)
    title_font = pygame.font.SysFont("Consolas", 36, bold=True)

    screen.fill(SKY_COLOR)

    ground_y = HEIGHT - GROUND_HEIGHT

    pygame.draw.line(
        screen,
        HORIZON_COLOR,
        (0, ground_y),
        (WIDTH, ground_y),
        2
    )


    pygame.draw.rect(
        screen,
        GROUND_COLOR,
        (0, ground_y, WIDTH, GROUND_HEIGHT)
    )


    pygame.draw.rect(
        screen,
        PAD_COLOR,
        (PAD_X, PAD_Y, PAD_WIDTH, PAD_HEIGHT)
    )

    rotated = pygame.transform.rotate(
        rocket.image,
        -math.degrees(rocket.angle)
    )

    rect = rotated.get_rect(
        center=(rocket.x, rocket.y)
    )

    screen.blit(rotated, rect)

    draw_main_engine(screen, rocket)
    draw_left_thruster(screen, rocket)
    draw_right_thruster(screen, rocket)

    hud = pygame.Surface((255, 175))
    hud.set_alpha(170)
    hud.fill(HUD_BG)

    screen.blit(hud, (15, 15))

    altitude = max(
        0,
        ground_y - (rocket.y + ROCKET_HEIGHT / 2)
    )

    lines = [
        f"Altitude : {altitude:.1f}",
        f"V Speed  : {rocket.vy:.2f}",
        f"H Speed  : {rocket.vx:.2f}",
        f"Angle    : {math.degrees(rocket.angle):.1f}°",
        f"Fuel     : {rocket.fuel / MAX_FUEL * 100:.1f}%",
        f"FPS      : {int(clock.get_fps())}"
    ]

    for i, line in enumerate(lines):

        img = font.render(
            line,
            True,
            HUD_TEXT
        )

        screen.blit(
            img,
            (30, 30 + i * 24)
        )

    if rocket.simulation_over:

        panel = pygame.Surface((420, 190))
        panel.set_alpha(210)
        panel.fill((35, 35, 35))

        panel_rect = panel.get_rect(
            center=(WIDTH // 2, HEIGHT // 2)
        )

        screen.blit(panel, panel_rect)

        if rocket.landed:

            title = title_font.render(
                "SUCCESSFUL LANDING",
                True,
                (0, 220, 0)
            )

        else:

            title = title_font.render(
                "CRASH",
                True,
                (230, 50, 50)
            )

        screen.blit(
            title,
            (
                WIDTH // 2 - title.get_width() // 2,
                HEIGHT // 2 - 70
            )
        )

        info_font = pygame.font.SysFont(
            "Consolas",
            22
        )

        stats = [

            f"Vertical Speed   : {abs(rocket.impact_vy):.2f}",

            f"Horizontal Speed : {abs(rocket.impact_vx):.2f}",

            f"Angle            : {abs(math.degrees(rocket.impact_angle)):.1f}°"

        ]

        for i, stat in enumerate(stats):

            img = info_font.render(
                stat,
                True,
                (255, 255, 255)
            )

            screen.blit(
                img,
                (
                    WIDTH // 2 - img.get_width() // 2,
                    HEIGHT // 2 - 10 + i * 30
                )
            )

        restart = font.render(
            "Press R to Restart",
            True,
            (200, 200, 200)
        )

        screen.blit(
            restart,
            (
                WIDTH // 2 - restart.get_width() // 2,
                HEIGHT // 2 + 95
            )
        )
