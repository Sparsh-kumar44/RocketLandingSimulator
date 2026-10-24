import pygame

from src.settings import WIDTH, HEIGHT, FPS
from src.rocket import Rocket
from src.renderer import draw_scene
from src.physics import update

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rocket Landing Simulator")

clock = pygame.time.Clock()

rocket = Rocket()

running = True

while running:

    clock.tick(FPS)
    for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r and rocket.simulation_over:
                    rocket = Rocket()
    keys = pygame.key.get_pressed()
    if not rocket.simulation_over:
        rocket.main_engine = keys[pygame.K_UP]
        rocket.left_thruster = keys[pygame.K_LEFT]
        rocket.right_thruster = keys[pygame.K_RIGHT]
    else:
        rocket.main_engine = False
        rocket.left_thruster = False
        rocket.right_thruster = False
    update(rocket)
    draw_scene(screen, rocket, clock)

    pygame.display.flip()

pygame.quit()