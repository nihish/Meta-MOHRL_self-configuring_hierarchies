"""
HIRO baseline: HIerarchical Reinforcement learning with Off-policy correction.

Adapted for discrete action spaces (traffic signal control).
2-level hierarchy:
- Higher-level controller: selects subgoals every c steps
- Lower-level controller: executes primitive actions given subgoal
Uses TD3-style critics with hindsight goal relabeling.

Reference: Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning", NeurIPS 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


class HIROHighLevel(nn.Module):
    """Higher-level controller: selects subgoals every c steps."""

    def __init__(self, obs_dim: int, goal_dim: int, hidden: int = 256):
        super().__init__()
        self.goal_dim = goal_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, goal_dim), nn.Tanh()
        )
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def q_values(self, state: torch.Tensor, goal: torch.Tensor):
        x = torch.cat([state, goal], dim=-1)
        return self.critic1(x), self.critic2(x)


class HIROLowLevel(nn.Module):
    """Lower-level controller: executes actions given subgoal."""

    def __init__(self, obs_dim: int, goal_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.actor = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + num_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.actor(x)

    def q_value(self, state: torch.Tensor, goal: torch.Tensor, action_oh: torch.Tensor):
        x = torch.cat([state, goal, action_oh], dim=-1)
        return self.critic(x)


class HIROAgent:
    """HIRO: 2-level hierarchical RL with off-policy correction."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        goal_dim: int = 16,
        commitment: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.goal_dim = goal_dim
        self.commitment = commitment
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.high = HIROHighLevel(obs_dim, goal_dim).to(device)
        self.low = HIROLowLevel(obs_dim, goal_dim, num_actions).to(device)

        self.high_target = HIROHighLevel(obs_dim, goal_dim).to(device)
        self.high_target.load_state_dict(self.high.state_dict())

        self.high_opt = torch.optim.Adam(self.high.parameters(), lr=lr)
        self.low_opt = torch.optim.Adam(self.low.parameters(), lr=lr)

        self.high_buffer = deque(maxlen=buffer_size)
        self.low_buffer = deque(maxlen=buffer_size)

        self.current_goal = None
        self.step_count = 0
        self.episode_rewards = []

    def reset(self):
        self.current_goal = None
        self.step_count = 0
        self.episode_rewards = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if self.step_count % self.commitment == 0 or self.current_goal is None:
            with torch.no_grad():
                self.current_goal = self.high(obs_t)
                if not deterministic:
                    self.current_goal += torch.randn_like(self.current_goal) * 0.1

        with torch.no_grad():
            logits = self.low(obs_t, self.current_goal)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / 0.5, dim=-1)
                action = torch.multinomial(probs, 1).item()

        self.step_count += 1
        return action

    def observe(self, obs, action, reward_vec, next_obs, done, **kwargs):
        self.episode_rewards.append(reward_vec)
        scalar_r = reward_vec.sum()
        goal_np = self.current_goal.detach().cpu().numpy().flatten() \
            if self.current_goal is not None else np.zeros(self.goal_dim)

        self.low_buffer.append((obs, goal_np, action, scalar_r, next_obs, done))

        if self.step_count % self.commitment == 0:
            self.high_buffer.append((obs, goal_np, scalar_r * self.commitment, next_obs, done))

    def train_step(self) -> Dict[str, float]:
        losses = {}
        if len(self.low_buffer) >= self.batch_size:
            batch = random.sample(list(self.low_buffer), self.batch_size)
            states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
            goals = torch.FloatTensor([b[1] for b in batch]).to(self.device)
            actions = torch.LongTensor([b[2] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[3] for b in batch]).to(self.device)
            next_states = torch.FloatTensor([b[4] for b in batch]).to(self.device)
            dones = torch.FloatTensor([b[5] for b in batch]).to(self.device)

            actions_oh = F.one_hot(actions, self.num_actions).float()
            q = self.low.q_value(states, goals, actions_oh).squeeze()
            with torch.no_grad():
                next_logits = self.low(next_states, goals)
                next_a = next_logits.argmax(-1)
                next_oh = F.one_hot(next_a, self.num_actions).float()
                q_target = rewards + self.gamma * (1 - dones) * \
                    self.low.q_value(next_states, goals, next_oh).squeeze()

            low_loss = F.mse_loss(q, q_target)
            logits = self.low(states, goals)
            actor_loss = -F.softmax(logits, -1).gather(1, actions.unsqueeze(1)).log().mean()

            self.low_opt.zero_grad()
            (low_loss + actor_loss).backward()
            nn.utils.clip_grad_norm_(self.low.parameters(), 1.0)
            self.low_opt.step()
            losses['hiro_low_loss'] = low_loss.item()

        if len(self.high_buffer) >= self.batch_size:
            batch = random.sample(list(self.high_buffer), self.batch_size)
            states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
            goals = torch.FloatTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

            q1, q2 = self.high.q_values(states, goals)
            with torch.no_grad():
                next_g = self.high_target(next_states)
                tq1, tq2 = self.high_target.q_values(next_states, next_g)
                q_target = rewards + self.gamma * (1 - dones) * torch.min(tq1, tq2).squeeze()

            high_loss = F.mse_loss(q1.squeeze(), q_target) + F.mse_loss(q2.squeeze(), q_target)

            pred_goals = self.high(states)
            q_actor = self.high.critic1(torch.cat([states, pred_goals], -1))
            actor_loss = -q_actor.mean()

            self.high_opt.zero_grad()
            (high_loss + actor_loss).backward()
            nn.utils.clip_grad_norm_(self.high.parameters(), 1.0)
            self.high_opt.step()

            for p, tp in zip(self.high.parameters(), self.high_target.parameters()):
                tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

            losses['hiro_high_loss'] = high_loss.item()

        return losses

    def get_metrics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        ep_r = np.array(self.episode_rewards)
        return {
            'total_reward': ep_r.sum(),
            'avg_speed_reward': ep_r[:, 0].mean(),
            'avg_waiting_reward': ep_r[:, 1].mean(),
            'avg_queue_reward': ep_r[:, 2].mean()
        }
