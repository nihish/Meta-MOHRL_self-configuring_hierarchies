"""
Task Encoder for Meta-MOHRL.

f_enc(d; φ) → z ∈ ℝ⁶⁴

Maps task descriptor d ∈ ℝ⁵ to embedding z ∈ ℝ⁶⁴.
Architecture: ℝ⁵ → ℝ¹²⁸ →(ReLU)→ ℝ¹²⁸ →(ReLU)→ ℝ⁶⁴

Task descriptor d = [n/n_max, H_est/H_max, K/K_max, μ̄_{s₀}, σ̄²_{s₀}]
From Section 4.2 (Eq. 8) of Meta-MOHRL paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TaskEncoder(nn.Module):
    """Encodes observable task properties into a fixed-dimensional embedding.

    The task descriptor d captures:
    - n/n_max: normalized agent count
    - H_est/H_max: normalized estimated horizon
    - K/K_max: normalized number of objectives
    - μ̄(s₀): mean of initial state distribution
    - σ̄²(s₀): variance of initial state distribution
    """

    def __init__(
        self,
        descriptor_dim: int = 5,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.embedding_dim = embedding_dim

        # ℝ⁵ → ℝ¹²⁸ → ReLU → ℝ¹²⁸ → ReLU → ℝ⁶⁴
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            descriptor: task descriptor [batch, 5]
        Returns:
            embedding: task embedding z [batch, 64]
        """
        return self.encoder(descriptor)

    @staticmethod
    def compute_descriptor(
        num_agents: int,
        horizon_estimate: int,
        num_objectives: int,
        initial_states: np.ndarray,
        n_max: int = 40,
        H_max: int = 4000,
        K_max: int = 5
    ) -> np.ndarray:
        """Compute task descriptor from environment metadata.

        Args:
            num_agents: number of agents in the environment
            horizon_estimate: estimated episode length in steps
            num_objectives: number of reward objectives
            initial_states: array of initial state samples [m, obs_dim]
            n_max, H_max, K_max: normalization constants
        Returns:
            descriptor: [5] normalized task descriptor
        """
        # Safely compute stats (handle empty arrays and zero-dim)
        if initial_states is not None and initial_states.size > 0:
            s_mean = float(np.nanmean(initial_states))
            s_var = float(np.nanvar(initial_states))
        else:
            s_mean = 0.0
            s_var = 0.0

        # Replace any residual NaN with 0
        if np.isnan(s_mean):
            s_mean = 0.0
        if np.isnan(s_var):
            s_var = 0.0

        d = np.array([
            num_agents / n_max,
            horizon_estimate / H_max,
            num_objectives / K_max,
            s_mean,
            s_var
        ], dtype=np.float32)

        # Replace any NaN and clip to [0, 1]
        d = np.nan_to_num(d, nan=0.0)
        d = np.clip(d, 0.0, 1.0)
        return d
