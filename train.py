from collections import deque

import pygame

from src.environment import RocketEnv
from src.agent import Agent


EPISODES = 3000
MAX_STEPS = 1000
MEMORY_WARMUP = 1000
TRAIN_EVERY = 2
MODEL_PATH = "rocket_model.pth"

START_DIFFICULTY = 0.0
DIFFICULTY_STEP = 0.1
PROMOTE_WINDOW = 50
PROMOTE_RATE = 0.6

EVAL_EVERY = 25
EVAL_EPISODES = 50


def evaluate(agent, episodes=EVAL_EPISODES):
    env = RocketEnv(difficulty=1.0)
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    landings = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            state, _, done = env.step(agent.select_action(state))
            steps += 1
        landings += int(env.rocket.landed)
    agent.epsilon = saved_epsilon
    return landings / episodes


def train(episodes=EPISODES, save_path=MODEL_PATH):
    pygame.init()
    pygame.display.set_mode((1, 1))

    env = RocketEnv()
    agent = Agent()
    level_landings = deque(maxlen=PROMOTE_WINDOW)
    difficulty = START_DIFFICULTY
    best_full_rate = -1.0

    for episode in range(episodes):
        state = env.reset(difficulty)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            steps += 1

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= MEMORY_WARMUP and steps % TRAIN_EVERY == 0:
                agent.train()

            state = next_state
            total_reward += reward

        landed = bool(env.rocket.landed)
        level_landings.append(1 if landed else 0)
        agent.decay_epsilon()

        level_count = len(level_landings)
        level_rate = sum(level_landings) / level_count
        outcome = "LANDED" if landed else ("TIMEOUT" if not done else "CRASH")
        print(
            f"Episode {episode + 1:5d} | {outcome:7s} | "
            f"Reward: {total_reward:8.2f} | Steps: {steps:4d} | "
            f"Level rate (last {level_count:2d}): {100 * level_rate:5.1f}% | "
            f"Difficulty: {difficulty:.1f} | Epsilon: {agent.epsilon:.3f}"
        )

        if (
            difficulty < 1.0
            and level_count == PROMOTE_WINDOW
            and level_rate >= PROMOTE_RATE
        ):
            difficulty = min(1.0, round(difficulty +
             DIFFICULTY_STEP, 2))
            level_landings.clear()
            print(f"--- Difficulty raised to {difficulty:.1f} ---")

        if difficulty == 1.0 and (episode + 1) % EVAL_EVERY == 0:
            eval_rate = evaluate(agent)
            message = f"--- Test flights: {100 * eval_rate:.0f}% landed"
            if eval_rate > best_full_rate:
                best_full_rate = eval_rate
                agent.save(save_path)
                message += " (new best, saved)"
            print(message + f" | best so far: {100 * best_full_rate:.0f}% ---")
        elif difficulty < 1.0 and best_full_rate < 0 and (episode + 1) % 100 == 0:
            agent.save(save_path)

    if best_full_rate < 0:
        agent.save(save_path)
    print(f"Done. Best model landed {100 * max(best_full_rate, 0):.0f}% of test flights.")
    pygame.quit()


if __name__ == "__main__":
    train()
