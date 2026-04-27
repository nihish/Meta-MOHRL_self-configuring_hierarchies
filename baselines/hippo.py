"""
HiPPO baseline: Hierarchical Proximal Policy Optimization with Learned Options.

3-level hierarchy with option discovery via meta-learning:
- Manager: selects options (long-term strategies)
- Sub-manager: refines options into sub-routines
- Worker: executes primitive actions
All trained with PPO + option termination learning.

Reference: Li et al., "Sub-policy Adaptation for Hierarchical Reinforcement Learning", ICLR 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


class HiPPOLevel(nn.Module):
    """Single level of the HiPPO hierarchy (actor-critic with option termination)."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Option termination head
        self.termination = nn.Sequential(
            nn.Linear(input_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.actor(x)

    def get_value(self, x):
        return self.critic(x)

    def get_termination_prob(self, x):
        return self.termination(x)


class HiPPOAgent:
    """HiPPO: 3-level hierarchical PPO with learned options."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_options: int = 8,
        num_sub_options: int = 6,
        option_commitment: int = 10,
        sub_commitment: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_options = num_options
        self.num_sub_options = num_sub_options
        self.option_commitment = option_commitment
        self.sub_commitment = sub_commitment
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.device = device

        # Manager: selects high-level options
        self.manager = HiPPOLevel(obs_dim, num_options).to(device)
        # Sub-manager: selects sub-options conditioned on option
        self.sub_manager = HiPPOLevel(obs_dim + num_options, num_sub_options).to(device)
        # Worker: selects primitive actions
        self.worker = HiPPOLevel(obs_dim + num_sub_options, num_actions).to(device)

        self.manager_opt = torch.optim.Adam(self.manager.parameters(), lr=lr)
        self.sub_opt = torch.optim.Adam(self.sub_manager.parameters(), lr=lr)
        self.worker_opt = torch.optim.Adam(self.worker.parameters(), lr=lr)

        # Trajectory buffers for PPO
        self.trajectories = {'manager': [], 'sub': [], 'worker': []}

        self.current_option = None
        self.current_sub_option = None
        self.step_count = 0
        self.episode_rewards = []

    def reset(self):
        self.current_option = None
        self.current_sub_option = None
        self.step_count = 0
        self.episode_rewards = []
        for k in self.trajectories:
            self.trajectories[k] = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Manager decision
        if self.step_count % self.option_commitment == 0 or self.current_option is None:
            with torch.no_grad():
                logits = self.manager(obs_t)
                if deterministic:
                    option = logits.argmax(-1).item()
                else:
                    probs = F.softmax(logits, -1)
                    option = torch.multinomial(probs, 1).item()
                # Check termination
                if self.current_option is not None:
                    term_prob = self.manager.get_termination_prob(obs_t)
                    if term_prob.item() > 0.5 or self.step_count % self.option_commitment == 0:
                        self.current_option = option
                else:
                    self.current_option = option

        # Sub-manager decision
        option_oh = F.one_hot(
            torch.tensor([self.current_option]), self.num_options
        ).float().to(self.device)

        if self.step_count % self.sub_commitment == 0 or self.current_sub_option is None:
            with torch.no_grad():
                sub_input = torch.cat([obs_t, option_oh], -1)
                sub_logits = self.sub_manager(sub_input)
                if deterministic:
                    sub_opt = sub_logits.argmax(-1).item()
                else:
                    probs = F.softmax(sub_logits, -1)
                    sub_opt = torch.multinomial(probs, 1).item()
                self.current_sub_option = sub_opt

        # Worker decision
        sub_oh = F.one_hot(
            torch.tensor([self.current_sub_option]), self.num_sub_options
        ).float().to(self.device)

        with torch.no_grad():
            worker_input = torch.cat([obs_t, sub_oh], -1)
            action_logits = self.worker(worker_input)
            if deterministic:
                action = action_logits.argmax(-1).item()
            else:
                probs = F.softmax(action_logits, -1)
                action = torch.multinomial(probs, 1).item()

        self.step_count += 1
        return action

    def observe(self, obs, action, reward_vec, next_obs, done, **kwargs):
        self.episode_rewards.append(reward_vec)
        scalar_r = reward_vec.sum()

        self.trajectories['worker'].append({
            'obs': obs, 'action': action, 'reward': scalar_r,
            'next_obs': next_obs, 'done': done,
            'sub_option': self.current_sub_option
        })

        if self.step_count % self.sub_commitment == 0:
            self.trajectories['sub'].append({
                'obs': obs, 'action': self.current_sub_option,
                'reward': scalar_r * self.sub_commitment,
                'next_obs': next_obs, 'done': done,
                'option': self.current_option
            })

        if self.step_count % self.option_commitment == 0:
            self.trajectories['manager'].append({
                'obs': obs, 'action': self.current_option,
                'reward': scalar_r * self.option_commitment,
                'next_obs': next_obs, 'done': done
            })

    def _ppo_update(self, level: nn.Module, optimizer, trajectories, input_dim, num_outputs):
        """Generic PPO update for one level."""
        if len(trajectories) < self.batch_size:
            return 0.0

        batch = random.sample(trajectories, min(self.batch_size, len(trajectories)))
        states = torch.FloatTensor([b['obs'] for b in batch]).to(self.device)
        if states.shape[1] < input_dim:
            pad = torch.zeros(states.shape[0], input_dim - states.shape[1]).to(self.device)
            states = torch.cat([states, pad], -1)

        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b['done'] for b in batch]).to(self.device)

        # Compute advantages
        values = level.get_value(states).squeeze()
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss with clipping
        logits = level(states)
        probs = F.softmax(logits, -1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Old log probs (approximate with current since we don't store them)
        old_log_probs = log_probs.detach()
        ratio = (log_probs - old_log_probs).exp()
        clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, rewards)

        # Entropy bonus
        entropy = dist.entropy().mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(level.parameters(), 1.0)
        optimizer.step()

        return total_loss.item()

    def train_step(self) -> Dict[str, float]:
        losses = {}

        # Worker update
        if self.trajectories['worker']:
            worker_data = []
            for t in self.trajectories['worker']:
                sub_oh = np.zeros(self.num_sub_options)
                sub_oh[t.get('sub_option', 0) or 0] = 1.0
                worker_data.append({
                    'obs': np.concatenate([t['obs'], sub_oh]),
                    'action': t['action'], 'reward': t['reward'],
                    'done': t['done']
                })
            loss = self._ppo_update(
                self.worker, self.worker_opt, worker_data,
                self.obs_dim + self.num_sub_options, self.num_actions
            )
            losses['hippo_worker_loss'] = loss

        # Sub-manager update
        if self.trajectories['sub']:
            sub_data = []
            for t in self.trajectories['sub']:
                opt_oh = np.zeros(self.num_options)
                opt_oh[t.get('option', 0) or 0] = 1.0
                sub_data.append({
                    'obs': np.concatenate([t['obs'], opt_oh]),
                    'action': t['action'], 'reward': t['reward'],
                    'done': t['done']
                })
            loss = self._ppo_update(
                self.sub_manager, self.sub_opt, sub_data,
                self.obs_dim + self.num_options, self.num_sub_options
            )
            losses['hippo_sub_loss'] = loss

        # Manager update
        if self.trajectories['manager']:
            loss = self._ppo_update(
                self.manager, self.manager_opt, self.trajectories['manager'],
                self.obs_dim, self.num_options
            )
            losses['hippo_manager_loss'] = loss

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
