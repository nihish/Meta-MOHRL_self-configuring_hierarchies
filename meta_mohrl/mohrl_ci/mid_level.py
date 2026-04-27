"""
Mid-level policy for MOHRL-ci (Level 2 - Tactical).

π²(m | s̃_t, g_t): Translates strategic goals into intermediate subgoals.
Operates every T² environment steps.
Objective: Minimize intersection waiting time.
Reward: r²_t = ⟨ω², R_t⟩ + β²³ · V³(s_{t+1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from meta_mohrl.core.networks import ActorNetwork, CriticNetwork, ValueNetwork


class MidLevelPolicy(nn.Module):
    """Tactical subgoal selection policy (Level 2).

    Refines high-level strategic goals into actionable intermediate subgoals
    for the low-level policy.
    """

    def __init__(
        self,
        augmented_state_dim: int,
        goal_input_dim: int = 16,
        num_subgoals: int = 6,
        num_objectives: int = 3,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        subgoal_dim: int = 16,
        lr: float = 5e-4,
        device: str = "cpu"
    ):
        super().__init__()
        self.num_subgoals = num_subgoals
        self.subgoal_dim = subgoal_dim
        self.device = device

        # Input includes augmented state + high-level goal
        total_input = augmented_state_dim + goal_input_dim

        # Actor: π²(m | s̃_t, g_t)
        self.actor = ActorNetwork(
            total_input, num_subgoals, hidden_dim, lstm_hidden
        ).to(device)

        # Critic: Q²(s̃, m) → R^K
        self.critic = CriticNetwork(
            total_input, num_subgoals, num_objectives, hidden_dim
        ).to(device)

        # Target critic
        self.critic_target = CriticNetwork(
            total_input, num_subgoals, num_objectives, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Value network V²(s̃) for bidirectional feedback
        self.value_net = ValueNetwork(total_input, hidden_dim).to(device)

        # Subgoal embedding
        self.subgoal_embedding = nn.Embedding(num_subgoals, subgoal_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.value_net.parameters()),
            lr=lr * 2
        )

        self.lstm_state = None

    def select_subgoal(
        self,
        state: torch.Tensor,
        high_goal: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a tactical subgoal conditioned on high-level goal.

        Args:
            state: augmented state [1, state_dim]
            high_goal: high-level goal embedding [1, goal_dim]
            deterministic: greedy if True
        Returns:
            subgoal_idx, subgoal_emb, log_prob
        """
        combined = torch.cat([state, high_goal], dim=-1)
        with torch.no_grad():
            sg_idx, log_prob, self.lstm_state = self.actor.get_action(
                combined, self.lstm_state, deterministic
            )
        sg_emb = self.subgoal_embedding(sg_idx)
        return sg_idx, sg_emb, log_prob

    def get_value(self, state: torch.Tensor, high_goal: torch.Tensor) -> torch.Tensor:
        """Get V²(s̃) for feedback signal."""
        combined = torch.cat([state, high_goal], dim=-1)
        with torch.no_grad():
            return self.value_net(combined)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        omega: torch.Tensor,
        gamma: float = 0.99,
        tau: float = 0.005,
        grad_clip: float = 1.0
    ) -> Dict[str, float]:
        """Update mid-level policy via actor-critic."""
        states = batch['states']       # includes goal concatenation
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        actions_onehot = F.one_hot(actions, self.num_subgoals).float()

        # Critic update
        with torch.no_grad():
            next_logits, _ = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions = next_probs.argmax(dim=-1)
            next_ah = F.one_hot(next_actions, self.num_subgoals).float()
            target_q = self.critic_target(next_states, next_ah)
            q_target = rewards + gamma * (1 - dones.unsqueeze(-1)) * target_q

        q_current = self.critic(states, actions_onehot)
        critic_loss = F.mse_loss(q_current, q_target)

        with torch.no_grad():
            scal_q = (q_current * omega.unsqueeze(0)).sum(-1, keepdim=True)
        value_pred = self.value_net(states)
        value_loss = F.mse_loss(value_pred, scal_q.detach())

        total_loss = critic_loss + 0.5 * value_loss
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        self.critic_optimizer.step()

        # Actor update
        logits, _ = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        with torch.no_grad():
            q_vals = self.critic(states, actions_onehot)
            advantages = (q_vals * omega.unsqueeze(0)).sum(-1)

        actor_loss = -(log_probs * advantages).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        self.actor_optimizer.step()

        # Soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {
            'mid_actor_loss': actor_loss.item(),
            'mid_critic_loss': critic_loss.item(),
            'mid_value_loss': value_loss.item(),
            'mid_entropy': entropy.item()
        }

    def reset_lstm(self):
        self.lstm_state = None
