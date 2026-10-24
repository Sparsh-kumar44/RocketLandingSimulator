import pygame
import torch

from src.environment import RocketEnv
from src.agent import Agent
from src.settings import WIDTH, HEIGHT, FPS


def play():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rocket Landing AI")

    clock = pygame.time.Clock()

    env = RocketEnv()


    agent = Agent()

    agent.load("rocket_model.pth")


    agent.epsilon = 0

    state = env.reset()

    running = True

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

    
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_r
            ):
                state = env.reset()


        action = agent.select_action(state)

        state, reward, done = env.step(action)


        env.render(screen, clock)

        pygame.display.flip()


    pygame.quit()


if __name__ == "__main__":

    play()