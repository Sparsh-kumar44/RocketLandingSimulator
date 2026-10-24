from collections import deque

import pygame

from src.environment import RocketEnv
from src.agent import Agent


EPISODES = 10000
MAX_STEPS = 1000
MEMORY_WARMUP = 1000
TRAIN_EVERY = 4


def train():
    pygame.init()
    pygame.display.set_mode((1, 1))

    env = RocketEnv()
    agent = Agent()
    recent_landings = deque(maxlen=100)

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            steps += 1

            # Treat the time limit as terminal so the network does not learn
            # to bootstrap from a state that will be discarded on reset.
            timed_out = steps >= MAX_STEPS and not done
            terminal = done or timed_out
            agent.remember(state, action, reward, next_state, terminal)

            if len(agent.memory) >= MEMORY_WARMUP and steps % TRAIN_EVERY == 0:
                agent.train()

            state = next_state
            total_reward += reward
            done = terminal

        landed = bool(env.rocket.landed)
        recent_landings.append(1 if landed else 0)
        agent.decay_epsilon()

        landing_rate = 100.0 * sum(recent_landings) / len(recent_landings)
        outcome = "LANDED" if landed else ("TIMEOUT" if steps >= MAX_STEPS else "CRASH")
        print(
            f"Episode {episode + 1:5d} | {outcome:7s} | "
            f"Reward: {total_reward:8.2f} | Steps: {steps:4d} | "
            f"Landing rate (last {len(recent_landings):3d}): {landing_rate:5.1f}% | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        if (episode + 1) % 100 == 0:
            agent.save("rocket_model.pth")

    agent.save("rocket_model.pth")
    pygame.quit()


if __name__ == "__main__":
    train()
