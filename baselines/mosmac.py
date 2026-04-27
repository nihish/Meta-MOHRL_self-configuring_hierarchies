"""
MOSMAC baseline: Multi-Objective Sequential Multi-Agent Cooperative RL.

Adapted from the official AAMAS 2025 codebase:
  https://github.com/smu-ncc/MOSMAC

Core algorithm (adapted for single-agent multi-objective traffic control):
  1. GRU-based recurrent agent for temporal reasoning
  2. Multi-objective Q-decomposition with per-objective Q-heads
  3. Epsilon-greedy exploration with linear annealing
  4. Sequential objective handling: objective weight shifting over training
  5. Target network with periodic hard updates

The original MOSMAC is a MARL benchmark on StarCraft II. We adapt its
core algorithmic ideas — RNN agents + multi-objective Q-decomposition +
sequential task allocation — for our single-agent SUMO-RL environment.

Reference:
  Geng et al., "MOSMAC: A Multi-agent Reinforcement Learning Benchmark
  on Sequential Multi-Objective Tasks", AAMAS 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import math


# ═══════════════ RNN Agent (from MOSMAC rnn_agent.py) ═══════════════
class MOSMACRNNAgent(nn.Module):
    """
    GRU-based recurrent agent — faithful to MOSMAC's rnn_agent.py.
    Architecture: obs → FC → ReLU → GRU → FC → Q-values per objective
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64,
                 num_objectives: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_objectives = num_objectives

        # Input embedding
        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        # Recurrent layer (GRU as in official MOSMAC)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # Per-objective Q-value heads
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_actions)
            for _ in range(num_objectives)
        ])

        # Combined Q-head (for action selection)
        self.q_combined = nn.Linear(hidden_dim, num_actions)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            q_combined: [batch, num_actions] — for action selection
            q_objectives: list of [batch, num_actions] — per-objective Q-values
            hidden_out: [batch, hidden_dim] — new hidden state
        """
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden)

        q_objectives = [head(h) for head in self.q_heads]
        q_combined = self.q_combined(h)

        return q_combined, q_objectives, h


# ═══════════════ Episode Buffer (for RNN training) ═══════════════
class EpisodeBuffer:
    """Stores complete episodes for RNN-compatible batch training."""

    def __init__(self, max_episodes: int = 5000):
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []

    def add_step(self, obs, action, reward_vec, next_obs, done, hidden):
        self.current_episode.append({
            'obs': obs.copy() if isinstance(obs, np.ndarray) else np.array(obs),
            'action': action,
            'reward_vec': reward_vec.copy() if isinstance(reward_vec, np.ndarray) else np.array(reward_vec),
            'next_obs': next_obs.copy() if isinstance(next_obs, np.ndarray) else np.array(next_obs),
            'done': float(done),
            'hidden': hidden.detach().cpu().numpy(),
        })

    def end_episode(self):
        if self.current_episode:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []

    def sample_transitions(self, batch_size: int) -> list:
        """Sample random transitions across episodes (simpler than full-ep)."""
        all_transitions = []
        for ep in self.episodes:
            all_transitions.extend(ep)
        if len(all_transitions) < batch_size:
            return all_transitions
        return random.sample(all_transitions, batch_size)

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)


# ═══════════════ MOSMAC Agent ═══════════════
class MOSMACAgent:
    """
    MOSMAC-adapted agent for single-agent multi-objective traffic control.

    Key features from official MOSMAC:
    - GRU-based temporal reasoning (RNN agent)
    - Multi-objective Q-decomposition (separate Q per objective)
    - Epsilon-greedy with linear annealing
    - Sequential objective handling via shifting preference weights
    - Target network with hard updates
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        num_objectives: int = 3,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal_episodes: int = 2000,
        target_update_interval: int = 200,
        device: str = "cpu",
        **kwargs,  # Accept and ignore extra kwargs
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal_episodes = epsilon_anneal_episodes
        self.target_update_interval = target_update_interval

        # ── Networks ──
        self.agent = MOSMACRNNAgent(
            obs_dim, num_actions, hidden_dim, num_objectives
        ).to(device)

        self.target_agent = MOSMACRNNAgent(
            obs_dim, num_actions, hidden_dim, num_objectives
        ).to(device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        # ── Optimizer ──
        self.optimizer = torch.optim.RMSprop(
            self.agent.parameters(), lr=lr, alpha=0.99, eps=1e-5
        )

        # ── Buffer ──
        self.buffer = EpisodeBuffer(max_episodes=buffer_size)

        # ── State ──
        self.hidden_state = None
        self.step_count = 0
        self.total_episodes = 0
        self.max_episodes = 4000
        self.episode_rewards = []
        self.train_step_count = 0

        # ── Sequential objective weights ──
        # MOSMAC concept: shift focus across objectives during training
        # Early: focus on throughput (speed), Later: balance all
        self._objective_weights = np.array([1.0, 1.0, 1.0])

    def _get_epsilon(self) -> float:
        """Linear epsilon annealing."""
        frac = min(1.0, self.total_episodes / self.epsilon_anneal_episodes)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _update_objective_weights(self):
        """
        Sequential objective handling from MOSMAC:
        Phase 1 (0-25%): Focus on speed (throughput)
        Phase 2 (25-50%): Add waiting time
        Phase 3 (50-100%): Balance all objectives
        """
        progress = self.total_episodes / max(1, self.max_episodes)
        if progress < 0.25:
            self._objective_weights = np.array([2.0, 0.5, 0.5])
        elif progress < 0.5:
            self._objective_weights = np.array([1.5, 1.0, 0.5])
        else:
            self._objective_weights = np.array([1.0, 1.0, 1.0])

    def reset(self):
        """Reset for new episode."""
        # End previous episode in buffer
        self.buffer.end_episode()

        self.hidden_state = self.agent.init_hidden(1).to(self.device)
        self.step_count = 0
        self.episode_rewards = []
        self.total_episodes += 1
        self._update_objective_weights()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection using RNN agent."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_combined, q_objectives, self.hidden_state = self.agent(
                obs_t, self.hidden_state
            )

            # Weighted sum of per-objective Q-values for action selection
            q_weighted = torch.zeros_like(q_combined)
            for i, (q_obj, w) in enumerate(zip(q_objectives,
                                                  self._objective_weights)):
                q_weighted += w * q_obj
            # Blend with combined Q
            q_final = 0.5 * q_combined + 0.5 * q_weighted

        eps = self._get_epsilon() if not deterministic else 0.0
        if random.random() < eps:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = q_final.argmax(dim=-1).item()

        self.step_count += 1
        return action

    def observe(self, obs, action, reward_vec, next_obs, done, **kwargs):
        """Store transition."""
        self.episode_rewards.append(reward_vec)
        self.buffer.add_step(
            obs, action, reward_vec, next_obs, done,
            self.hidden_state.squeeze(0)
        )

    def train_step(self) -> Dict[str, float]:
        """Train on a batch of transitions."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample_transitions(self.batch_size)

        obs = torch.FloatTensor(np.array([b['obs'] for b in batch])).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        reward_vecs = torch.FloatTensor(
            np.array([b['reward_vec'] for b in batch])
        ).to(self.device)
        next_obs = torch.FloatTensor(
            np.array([b['next_obs'] for b in batch])
        ).to(self.device)
        dones = torch.FloatTensor([b['done'] for b in batch]).to(self.device)
        hiddens = torch.FloatTensor(
            np.array([b['hidden'] for b in batch])
        ).to(self.device)

        losses = {}

        # Forward pass with stored hidden states
        q_combined, q_objectives, _ = self.agent(obs, hiddens)

        # Target computation (no gradient)
        with torch.no_grad():
            target_hidden = self.target_agent.init_hidden(
                self.batch_size).to(self.device)
            t_q_combined, t_q_objectives, _ = self.target_agent(
                next_obs, target_hidden)

        # ── Per-objective Q-learning losses ──
        total_loss = torch.tensor(0.0, device=self.device)
        weights_t = torch.FloatTensor(self._objective_weights).to(self.device)

        for i in range(self.num_objectives):
            q_i = q_objectives[i]  # [batch, num_actions]
            q_taken = q_i.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                t_q_i = t_q_objectives[i]
                max_next_q = t_q_i.max(dim=-1)[0]
                target = reward_vecs[:, i] + self.gamma * (1 - dones) * max_next_q

            obj_loss = F.mse_loss(q_taken, target)
            total_loss += weights_t[i] * obj_loss
            losses[f'mosmac_q{i}_loss'] = obj_loss.item()

        # ── Combined Q loss ──
        q_comb_taken = q_combined.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        weighted_reward = (reward_vecs * weights_t.unsqueeze(0)).sum(dim=-1)

        with torch.no_grad():
            max_next_q_comb = t_q_combined.max(dim=-1)[0]
            target_comb = weighted_reward + self.gamma * (1 - dones) * max_next_q_comb

        comb_loss = F.mse_loss(q_comb_taken, target_comb)
        total_loss += comb_loss
        losses['mosmac_combined_loss'] = comb_loss.item()

        # ── Backprop ──
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 10.0)
        self.optimizer.step()

        self.train_step_count += 1

        # ── Hard target update ──
        if self.train_step_count % self.target_update_interval == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

        losses['mosmac_total_loss'] = total_loss.item()
        losses['mosmac_epsilon'] = self._get_epsilon()
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
