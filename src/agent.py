import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import DQN
from src.memory import ReplayMemory


class Agent:
    def __init__(self):
        self.gamma = 0.99

        # Keep exploration alive for longer; early policies are usually poor.
        self.epsilon = 0.99
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.batch_size = 128
        self.tau = 0.005
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(50000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(6)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN: policy network selects the next action; target network
        # evaluates it. This tends to reduce overestimation of Q-values.
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.mul_(1.0 - self.tau).add_(policy_param, alpha=self.tau)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
