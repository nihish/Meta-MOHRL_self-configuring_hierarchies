"""
Bidirectional feedback module for MOHRL-ci.

Implements bottom-up feedback propagation:
- f_t = φ(s̃_t, g_h, a_t, r_t)  — feedback encoder
- g_{h,t+1} ~ π_h(g | s̃_{t+1}, g_{h-1}, f_t)  — reconditioned goal

And top-down subgoal signal flow through augmented state conditioning.
From Section IV-B of MOHRL-ci paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class BidirectionalFeedback(nn.Module):
    """Bidirectional feedback module coupling top-down goals with bottom-up outcomes.

    Bottom-up: Low-level execution quality → feedback vector → high-level goal revision
    Top-down:  High-level subgoal signal → conditions mid/low-level policies via augmented state
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        reward_dim: int = 3,
        feedback_dim: int = 64
    ):
        super().__init__()
        self.feedback_dim = feedback_dim

        # φ(s̃, g, a, r) → f_t
        total_in = state_dim + goal_dim + action_dim + reward_dim
        self.encoder = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feedback_dim),
            nn.Tanh()
        )

        # Relevance context aggregator: f_o(s_t, {s_i, h_i})
        # Simplified as attention-weighted average
        self.query_proj = nn.Linear(state_dim, 64)
        self.key_proj = nn.Linear(state_dim + feedback_dim, 64)
        self.value_proj = nn.Linear(state_dim + feedback_dim, feedback_dim)

    def compute_feedback(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action_onehot: torch.Tensor,
        reward: torch.Tensor
    ) -> torch.Tensor:
        """Compute feedback vector from execution outcome.

        f_t = φ(s̃_t, g_h, a_t, r_t)

        Args:
            state: augmented state [batch, state_dim]
            goal: current subgoal [batch, goal_dim]
            action_onehot: one-hot action [batch, action_dim]
            reward: reward vector [batch, reward_dim]
        Returns:
            feedback: [batch, feedback_dim]
        """
        x = torch.cat([state, goal, action_onehot, reward], dim=-1)
        return self.encoder(x)

    def compute_relevant_context(
        self,
        current_state: torch.Tensor,
        history_states: torch.Tensor,
        history_feedback: torch.Tensor
    ) -> torch.Tensor:
        """Compute relevant context embedding CX_t^Rel via attention.

        CX_t^Rel = f_o(s_t, {s_i, h_i}_{i=1}^t)

        Args:
            current_state: [batch, state_dim]
            history_states: [batch, T, state_dim]
            history_feedback: [batch, T, feedback_dim]
        Returns:
            relevant_context: [batch, feedback_dim]
        """
        if history_states.size(1) == 0:
            return torch.zeros(
                current_state.size(0), self.feedback_dim,
                device=current_state.device
            )

        query = self.query_proj(current_state).unsqueeze(1)        # [B, 1, 64]
        kv_input = torch.cat([history_states, history_feedback], dim=-1)
        keys = self.key_proj(kv_input)                              # [B, T, 64]
        values = self.value_proj(kv_input)                          # [B, T, fd]

        # Scaled dot-product attention
        attn = torch.bmm(query, keys.transpose(1, 2)) / 8.0       # [B, 1, T]
        attn = torch.softmax(attn, dim=-1)
        context = torch.bmm(attn, values).squeeze(1)               # [B, fd]

        return context
