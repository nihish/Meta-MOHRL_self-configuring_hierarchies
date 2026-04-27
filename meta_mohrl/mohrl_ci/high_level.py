"""
High-level policy for MOHRL-ci (Level 1 - Strategic).

π¹(g | s̃_t): Selects long-horizon strategic goals.
Operates every T¹ environment steps.
Objective: Maximize network-wide average speed.
Reward: r¹_t = ⟨ω¹, R_t⟩ + β¹² · V²(s_{t+1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from meta_mohrl.core.networks import ActorNetwork, CriticNetwork, ValueNetwork


class HighLevelPolicy(nn.Module):
    """Strategic goal-setting policy (Level 1).

    Selects abstract subgoals for the mid-level policy.
    Reconditioned by bottom-up feedback: g_{h,t+1} ~ π_h(g | s̃, g_{h-1}, f_t)
    """

    def __init__(
        self,
        augmented_state_dim: int,
        num_goals: int = 8,
        num_objectives: int = 3,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        goal_dim: int = 16,
        lr: float = 5e-4,
        device: str = "cpu"
    ):
        super().__init__()
        self.num_goals = num_goals
        self.goal_dim = goal_dim
        self.device = device

        # Actor: π¹(g | s̃_t)
        self.actor = ActorNetwork(
            augmented_state_dim, num_goals, hidden_dim, lstm_hidden
        ).to(device)

        # Critic: Q¹(s̃, g) → R^K
        self.critic = CriticNetwork(
            augmented_state_dim, num_goals, num_objectives, hidden_dim
        ).to(device)

        # Target critic for stable TD learning
        self.critic_target = CriticNetwork(
            augmented_state_dim, num_goals, num_objectives, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Value network for bidirectional feedback to Level 2
        self.value_net = ValueNetwork(augmented_state_dim, hidden_dim).to(device)

        # Goal embedding: maps discrete goal index to embedding vector
        self.goal_embedding = nn.Embedding(num_goals, goal_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.value_net.parameters()),
            lr=lr * 2
        )

        self.lstm_state = None

    def select_goal(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a strategic goal.

        Args:
            state: augmented state [1, state_dim]
            deterministic: greedy selection if True
        Returns:
            goal_idx: selected goal index [1]
            goal_emb: goal embedding [1, goal_dim]
            log_prob: log probability [1]
        """
        with torch.no_grad():
            goal_idx, log_prob, self.lstm_state = self.actor.get_action(
                state, self.lstm_state, deterministic
            )
        goal_emb = self.goal_embedding(goal_idx)
        return goal_idx, goal_emb, log_prob

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get V¹(s̃) for upper feedback signal."""
        with torch.no_grad():
            return self.value_net(state)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        omega: torch.Tensor,
        gamma: float = 0.99,
        tau: float = 0.005,
        grad_clip: float = 1.0
    ) -> Dict[str, float]:
        """Update high-level policy via actor-critic.

        Actor: ∇_θ J_h = E[∇ log π_h(g|s̃) · Q_h(s̃, g)]
        Critic: L_Q = Σ_k (Q_target^k - Q^k)²

        Returns dict of losses for logging.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # One-hot encode actions for critic
        actions_onehot = F.one_hot(actions, self.num_goals).float()

        # --- Critic Update ---
        with torch.no_grad():
            # Target: best next action under current policy
            next_logits, _ = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions = next_probs.argmax(dim=-1)
            next_actions_oh = F.one_hot(next_actions, self.num_goals).float()
            target_q = self.critic_target(next_states, next_actions_oh)
            q_target = rewards + gamma * (1 - dones.unsqueeze(-1)) * target_q

        q_current = self.critic(states, actions_onehot)
        critic_loss = F.mse_loss(q_current, q_target)

        # Value network loss
        with torch.no_grad():
            scalarized_q = (q_current * omega.unsqueeze(0)).sum(dim=-1, keepdim=True)
        value_pred = self.value_net(states)
        value_loss = F.mse_loss(value_pred, scalarized_q.detach())

        total_critic_loss = critic_loss + 0.5 * value_loss
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        self.critic_optimizer.step()

        # --- Actor Update ---
        logits, _ = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        with torch.no_grad():
            q_values = self.critic(states, actions_onehot)
            advantages = (q_values * omega.unsqueeze(0)).sum(dim=-1)

        actor_loss = -(log_probs * advantages).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        self.actor_optimizer.step()

        # --- Soft update target critic ---
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {
            'high_actor_loss': actor_loss.item(),
            'high_critic_loss': critic_loss.item(),
            'high_value_loss': value_loss.item(),
            'high_entropy': entropy.item()
        }

    def reset_lstm(self):
        """Reset LSTM state for new episode."""
        self.lstm_state = None
