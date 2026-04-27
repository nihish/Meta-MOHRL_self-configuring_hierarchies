"""
Low-level policy for MOHRL-ci (Level 3 - Operational).

π³(a | s̃_t, g_t, m_t): Selects primitive environment actions.
Operates every step (T³ = 1).
Objective: Minimize queue length.
Reward: r³_t = ⟨ω³, R_t⟩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

from meta_mohrl.core.networks import ActorNetwork, CriticNetwork, ValueNetwork


class LowLevelPolicy(nn.Module):
    """Primitive action selection policy (Level 3).

    Selects traffic signal phases conditioned on augmented state,
    high-level goal, and mid-level subgoal.
    """

    def __init__(
        self,
        augmented_state_dim: int,
        goal_dim: int = 16,
        subgoal_dim: int = 16,
        num_actions: int = 4,
        num_objectives: int = 3,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        super().__init__()
        self.num_actions = num_actions
        self.device = device

        # Input: augmented state + high goal + mid subgoal
        total_input = augmented_state_dim + goal_dim + subgoal_dim

        # Actor: π³(a | s̃_t, g, m)
        self.actor = ActorNetwork(
            total_input, num_actions, hidden_dim, lstm_hidden
        ).to(device)

        # Critic: Q³(s̃, a) → R^K
        self.critic = CriticNetwork(
            total_input, num_actions, num_objectives, hidden_dim
        ).to(device)

        # Target critic
        self.critic_target = CriticNetwork(
            total_input, num_actions, num_objectives, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Value network V³(s̃) for feedback to Level 2
        self.value_net = ValueNetwork(total_input, hidden_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        self.lstm_state = None

    def select_action(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        subgoal: torch.Tensor,
        deterministic: bool = False,
        epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a primitive action (traffic signal phase).

        Args:
            state: augmented state [1, state_dim]
            goal: high-level goal embedding [1, goal_dim]
            subgoal: mid-level subgoal embedding [1, subgoal_dim]
            deterministic: greedy if True
        Returns:
            action: selected action index [1]
            log_prob: log probability [1]
        """
        combined = torch.cat([state, goal, subgoal], dim=-1)
        with torch.no_grad():
            action, log_prob, self.lstm_state = self.actor.get_action(
                combined, self.lstm_state, deterministic
            )
            # Epsilon-greedy override to shatter deterministic traps
            if not deterministic and epsilon > 0.0:
                if np.random.rand() < epsilon:
                    action = torch.tensor([np.random.randint(self.num_actions)], device=self.device)
        return action, log_prob

    def get_value(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        subgoal: torch.Tensor
    ) -> torch.Tensor:
        """Get V³(s̃) for feedback to mid-level."""
        combined = torch.cat([state, goal, subgoal], dim=-1)
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
        """Update low-level policy."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        actions_onehot = F.one_hot(actions, self.num_actions).float()

        # Critic update
        with torch.no_grad():
            next_logits, _ = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions = next_probs.argmax(dim=-1)
            next_ah = F.one_hot(next_actions, self.num_actions).float()
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
        actor_loss = actor_loss - 0.05 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        self.actor_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {
            'low_actor_loss': actor_loss.item(),
            'low_critic_loss': critic_loss.item(),
            'low_value_loss': value_loss.item(),
            'low_entropy': entropy.item()
        }

    def reset_lstm(self):
        self.lstm_state = None
