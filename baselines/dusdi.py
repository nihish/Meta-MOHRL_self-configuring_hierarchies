"""
DUSDi baseline: Disentangled Unsupervised Skill Discovery for Efficient HRL.

Re-implemented faithfully from the official NeurIPS 2024 codebase:
  https://github.com/JiahengHu/DUSDi

Core algorithm:
  1. DIAYN-based skill discriminator partitioned by state dimensions
  2. Intrinsic reward: r_intr = log p(z|s,s') - log(1/K) per partition
  3. Skills are one-hot vectors per partition channel
  4. Actor-critic conditioned on obs + skill
  5. Two-phase training: intrinsic-only → mixed (intrinsic + extrinsic)

Adapted for single-agent SUMO-RL traffic signal control.

Reference:
  Hu et al., "DUSDi: Disentangled Unsupervised Skill Discovery for
  Efficient Hierarchical Reinforcement Learning", NeurIPS 2024.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import random


# ═══════════════ Partitioned DIAYN Discriminator ═══════════════
class PartitionedDIAYN(nn.Module):
    """
    Partitioned discriminator: one classifier per state-partition channel.
    Each channel predicts the skill index from a partition of (obs, next_obs).

    Official DUSDi uses PARTED_DIAYN which has one discriminator per
    state-factor partition, predicting z from the corresponding state slice.
    """

    def __init__(self, obs_dim: int, skill_dim: int, num_channels: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.skill_dim = skill_dim
        self.num_channels = num_channels

        # Partition obs into `num_channels` roughly equal slices
        self.partition_sizes = self._compute_partitions(obs_dim, num_channels)

        # One discriminator per partition channel
        self.discriminators = nn.ModuleList()
        for p_size in self.partition_sizes:
            # Input: concat of (obs_partition, next_obs_partition) for transition
            self.discriminators.append(nn.Sequential(
                nn.Linear(p_size * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, skill_dim),
            ))

    def _compute_partitions(self, obs_dim: int, num_channels: int) -> List[int]:
        """Partition obs_dim into num_channels roughly equal parts."""
        base = obs_dim // num_channels
        remainder = obs_dim % num_channels
        sizes = []
        for i in range(num_channels):
            sizes.append(base + (1 if i < remainder else 0))
        return sizes

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Returns logits [batch, num_channels, skill_dim] — one prediction per channel.
        """
        batch_size = obs.shape[0]
        all_logits = []
        offset = 0
        for i, (p_size, disc) in enumerate(zip(self.partition_sizes,
                                                 self.discriminators)):
            obs_part = obs[:, offset:offset + p_size]
            next_obs_part = next_obs[:, offset:offset + p_size]
            inp = torch.cat([obs_part, next_obs_part], dim=-1)
            logits = disc(inp)  # [batch, skill_dim]
            all_logits.append(logits)
            offset += p_size

        return torch.stack(all_logits, dim=1)  # [batch, num_channels, skill_dim]


# ═══════════════ Actor-Critic Networks ═══════════════
class DUSDiActor(nn.Module):
    """Actor network conditioned on obs + flattened skill vector."""

    def __init__(self, obs_dim: int, action_dim: int, skill_input_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + skill_input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, skill], dim=-1)
        return self.net(x)


class DUSDiCritic(nn.Module):
    """
    Separated critic: one Q-head per skill channel + optional external reward head.
    Following official DUSDi's SepCritic.
    """

    def __init__(self, obs_dim: int, action_dim: int, skill_input_dim: int,
                 num_channels: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + skill_input_dim + action_dim
        self.num_channels = num_channels

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Per-channel Q heads (one for intrinsic reward per channel)
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_channels)
        ])
        # External reward head
        self.ext_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action_oh: torch.Tensor,
                skill: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = torch.cat([obs, action_oh, skill], dim=-1)
        h = self.trunk(x)
        channel_qs = [head(h) for head in self.q_heads]
        ext_q = self.ext_head(h)
        return channel_qs, ext_q


# ═══════════════ DUSDi Agent ═══════════════
class DUSDiAgent:
    """
    DUSDi: Disentangled Unsupervised Skill Discovery (NeurIPS 2024).

    Faithful to official implementation:
      - Partitioned DIAYN discriminator
      - Intrinsic reward = log p(z|s,s') - log(1/K)
      - One-hot skill per channel (partition)
      - Two-phase: skill discovery then task optimization
      - Soft actor-critic style training
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        skill_dim: int = 5,
        num_channels: int = 3,
        update_skill_every_step: int = 15,
        diayn_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        buffer_size: int = 200_000,
        intrinsic_phase_frac: float = 0.2,
        device: str = "cpu",
        **kwargs,  # Accept and ignore extra kwargs from run_experiment
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.skill_dim = skill_dim
        self.num_channels = num_channels
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.intrinsic_phase_frac = intrinsic_phase_frac

        # Skill representation size: num_channels * skill_dim (flattened one-hot)
        self.skill_input_dim = num_channels * skill_dim

        # ── Networks ──
        self.diayn = PartitionedDIAYN(
            obs_dim, skill_dim, num_channels
        ).to(device)

        self.actor = DUSDiActor(
            obs_dim, num_actions, self.skill_input_dim
        ).to(device)

        self.critic = DUSDiCritic(
            obs_dim, num_actions, self.skill_input_dim, num_channels
        ).to(device)

        self.critic_target = DUSDiCritic(
            obs_dim, num_actions, self.skill_input_dim, num_channels
        ).to(device)
        # Init target = critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimizers ──
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.diayn_criterion = nn.CrossEntropyLoss()

        # ── Replay buffer ──
        self.buffer = deque(maxlen=buffer_size)

        # ── State ──
        self.current_skill = None  # [num_channels] int indices
        self.current_skill_flat = None  # [skill_input_dim] float one-hot
        self.step_count = 0
        self.total_episodes = 0
        self.max_episodes = 4000  # Set from outside
        self.episode_rewards = []

    def _sample_skill(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random skill: one index per channel → flattened one-hot."""
        indices = np.random.randint(0, self.skill_dim, size=self.num_channels)
        flat = np.zeros(self.skill_input_dim, dtype=np.float32)
        for c in range(self.num_channels):
            flat[c * self.skill_dim + indices[c]] = 1.0
        return indices, flat

    def reset(self):
        """Reset for new episode. Sample initial skill."""
        self.step_count = 0
        self.episode_rewards = []
        self.current_skill, self.current_skill_flat = self._sample_skill()
        self.total_episodes += 1

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action given obs + current skill."""
        # Skill update schedule: re-sample at fixed intervals
        if (self.step_count > 0 and
                self.step_count % self.update_skill_every_step == 0):
            self.current_skill, self.current_skill_flat = self._sample_skill()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        skill_t = torch.FloatTensor(self.current_skill_flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_t, skill_t)
            if deterministic:
                action = logits.argmax(-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

        self.step_count += 1
        return action

    def observe(self, obs, action, reward_vec, next_obs, done, **kwargs):
        """Store transition with skill info."""
        self.episode_rewards.append(reward_vec)
        self.buffer.append({
            'obs': obs.copy() if isinstance(obs, np.ndarray) else np.array(obs),
            'action': action,
            'reward_vec': reward_vec.copy() if isinstance(reward_vec, np.ndarray) else np.array(reward_vec),
            'next_obs': next_obs.copy() if isinstance(next_obs, np.ndarray) else np.array(next_obs),
            'done': float(done),
            'skill_indices': self.current_skill.copy(),
            'skill_flat': self.current_skill_flat.copy(),
        })

    def _is_intrinsic_phase(self) -> bool:
        """First intrinsic_phase_frac of training uses intrinsic reward only."""
        return self.total_episodes < self.max_episodes * self.intrinsic_phase_frac

    def _compute_intrinsic_reward(self, obs: torch.Tensor,
                                   next_obs: torch.Tensor,
                                   skill_indices: torch.Tensor) -> torch.Tensor:
        """
        r_intr = sum_c [ log p(z_c | s, s') - log(1/K) ]
        Following official DUSDi compute_intr_reward.
        """
        with torch.no_grad():
            # [batch, num_channels, skill_dim]
            d_pred = self.diayn(obs, next_obs)
            d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)

            # Gather log-prob for correct skill per channel
            batch_size = obs.shape[0]
            reward = torch.zeros(batch_size, device=self.device)
            for c in range(self.num_channels):
                z_c = skill_indices[:, c].long()  # [batch]
                log_prob = d_pred_log_softmax[:, c, :]  # [batch, skill_dim]
                r_c = log_prob[torch.arange(batch_size), z_c] - math.log(
                    1.0 / self.skill_dim)
                reward += r_c

        return reward * self.diayn_scale

    def train_step(self) -> Dict[str, float]:
        """One training step: update DIAYN, critic, actor."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = random.sample(list(self.buffer), self.batch_size)

        obs = torch.FloatTensor(np.array([b['obs'] for b in batch])).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        reward_vecs = torch.FloatTensor(
            np.array([b['reward_vec'] for b in batch])
        ).to(self.device)
        next_obs = torch.FloatTensor(
            np.array([b['next_obs'] for b in batch])
        ).to(self.device)
        dones = torch.FloatTensor([b['done'] for b in batch]).to(self.device)
        skill_indices = torch.LongTensor(
            np.array([b['skill_indices'] for b in batch])
        ).to(self.device)
        skill_flat = torch.FloatTensor(
            np.array([b['skill_flat'] for b in batch])
        ).to(self.device)

        extr_reward = reward_vecs.sum(dim=-1)  # scalar external reward
        losses = {}

        # ── 1. Update DIAYN discriminator ──
        d_pred = self.diayn(obs, next_obs)  # [batch, channels, skill_dim]
        diayn_loss = 0.0
        diayn_acc = 0.0
        for c in range(self.num_channels):
            z_c = skill_indices[:, c].long()
            pred_c = d_pred[:, c, :]
            diayn_loss += self.diayn_criterion(pred_c, z_c)
            # accuracy
            pred_idx = pred_c.argmax(dim=-1)
            diayn_acc += (pred_idx == z_c).float().mean().item()

        diayn_loss = diayn_loss  # already summed over channels
        self.diayn_opt.zero_grad()
        diayn_loss.backward()
        nn.utils.clip_grad_norm_(self.diayn.parameters(), 1.0)
        self.diayn_opt.step()
        losses['diayn_loss'] = diayn_loss.item()
        losses['diayn_acc'] = diayn_acc / self.num_channels

        # ── 2. Compute reward for critic ──
        intr_reward = self._compute_intrinsic_reward(obs, next_obs, skill_indices)

        if self._is_intrinsic_phase():
            reward = intr_reward
        else:
            # Mixed phase: external + attenuated intrinsic
            reward = extr_reward + 0.1 * intr_reward

        # ── 3. Update critic ──
        action_oh = F.one_hot(actions, self.num_actions).float()
        channel_qs, ext_q = self.critic(obs, action_oh, skill_flat)
        q_pred = ext_q.squeeze(-1)
        for cq in channel_qs:
            q_pred = q_pred + cq.squeeze(-1)

        with torch.no_grad():
            # Target: use greedy next action
            next_logits = self.actor(next_obs, skill_flat)
            next_action = next_logits.argmax(dim=-1)
            next_action_oh = F.one_hot(next_action, self.num_actions).float()
            next_channel_qs, next_ext_q = self.critic_target(
                next_obs, next_action_oh, skill_flat)
            q_next = next_ext_q.squeeze(-1)
            for cq in next_channel_qs:
                q_next = q_next + cq.squeeze(-1)
            q_target = reward + self.gamma * (1.0 - dones) * q_next

        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        losses['critic_loss'] = critic_loss.item()

        # ── 4. Update actor ──
        logits = self.actor(obs, skill_flat)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled_actions = dist.sample()
        sampled_oh = F.one_hot(sampled_actions, self.num_actions).float()
        channel_qs_a, ext_q_a = self.critic(obs, sampled_oh, skill_flat)
        q_val = ext_q_a.squeeze(-1)
        for cq in channel_qs_a:
            q_val = q_val + cq.squeeze(-1)

        actor_loss = -(dist.log_prob(sampled_actions) * q_val.detach()).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        losses['actor_loss'] = actor_loss.item()

        # ── 5. Soft update target critic ──
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return losses

    def get_metrics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        ep_r = np.array(self.episode_rewards)
        return {
            'total_reward': float(ep_r.sum()),
            'avg_speed_reward': float(ep_r[:, 0].mean()),
            'avg_waiting_reward': float(ep_r[:, 1].mean()),
            'avg_queue_reward': float(ep_r[:, 2].mean()),
        }
