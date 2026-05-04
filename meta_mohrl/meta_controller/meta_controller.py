"""
Meta-Controller for Meta-MOHRL.

f_meta(z; ψ) → c = (T¹, T², γ¹, γ², γ³, β¹², β²³)

Maps task embedding z to the 7-component configuration vector that
parameterizes the MOHRL-ci hierarchy for one episode.

Architecture: 2-layer MLP (128 hidden, ReLU) with three output heads:
1. Commitment intervals: Gumbel-softmax over {1, 2, 5, 10, 20, 50}
2. Discount factors: γ^l = 0.90 + 0.099·σ(f_γ(z))
3. Feedback gains: β^{l,l+1} = σ(f_β(z))

From Section 4.3 (Eqs. 9-13) of Meta-MOHRL paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from meta_mohrl.meta_controller.gumbel import GumbelSoftmaxSelector
from meta_mohrl.meta_controller.task_encoder import TaskEncoder
from meta_mohrl.config import MetaConfig


class MetaController(nn.Module):
    """Meta-controller that produces hierarchical configuration from task context.

    Operates once per episode at the outer-loop timescale.
    The MOHRL-ci hierarchy then runs for the full episode under this configuration.
    """

    def __init__(self, config: MetaConfig):
        super().__init__()
        self.config = config

        # Task encoder: d → z
        self.task_encoder = TaskEncoder(
            config.task_descriptor_dim,
            config.embedding_dim,
            config.encoder_hidden
        )

        # Shared backbone: z → hidden representation
        self.backbone = nn.Sequential(
            nn.Linear(config.embedding_dim, config.meta_hidden),
            nn.ReLU(),
            nn.Linear(config.meta_hidden, config.meta_hidden),
            nn.ReLU()
        )

        # --- Output Head 1: Commitment Intervals (Gumbel-softmax) ---
        self.T1_selector = GumbelSoftmaxSelector(
            config.meta_hidden,
            config.T1_candidates,
            config.gumbel_tau_start,
            config.gumbel_tau_min,
            config.gumbel_decay_rate
        )
        self.T2_selector = GumbelSoftmaxSelector(
            config.meta_hidden,
            config.T2_candidates,
            config.gumbel_tau_start,
            config.gumbel_tau_min,
            config.gumbel_decay_rate
        )

        # --- Output Head 2: Discount Factors ---
        # γ^l = 0.90 + 0.099 · σ(f_γ(z))  →  γ ∈ (0.90, 0.999)
        self.gamma1_head = nn.Linear(config.meta_hidden, 1)
        self.gamma2_head = nn.Linear(config.meta_hidden, 1)
        self.gamma3_head = nn.Linear(config.meta_hidden, 1)

        # --- Output Head 3: Feedback Gains ---
        # β^{l,l+1} = σ(f_β(z))  →  β ∈ (0, 1)
        self.beta12_head = nn.Linear(config.meta_hidden, 1)
        self.beta23_head = nn.Linear(config.meta_hidden, 1)

    def forward(
        self,
        descriptor: torch.Tensor,
        hard: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Produce configuration vector from task descriptor.

        Args:
            descriptor: task descriptor d [batch, 5]
            hard: if True, use hard categorical selection (test time)
        Returns:
            config: dict with T1, T2, gamma1, gamma2, gamma3, beta12, beta23
        """
        # Task encoding
        z = self.task_encoder(descriptor)  # [batch, 64]

        # Shared backbone
        h = self.backbone(z)  # [batch, 128]

        # Commitment intervals via Gumbel-softmax
        T1, T1_probs = self.T1_selector(h, hard=hard)
        T2, T2_probs = self.T2_selector(h, hard=hard)

        # Discount factors: γ = 0.90 + 0.099 · σ(·)
        gamma1 = 0.90 + 0.099 * torch.sigmoid(self.gamma1_head(h)).squeeze(-1)
        gamma2 = 0.90 + 0.099 * torch.sigmoid(self.gamma2_head(h)).squeeze(-1)
        gamma3 = 0.90 + 0.099 * torch.sigmoid(self.gamma3_head(h)).squeeze(-1)

        # Feedback gains: β = σ(·) ∈ (0, 1)
        beta12 = torch.sigmoid(self.beta12_head(h)).squeeze(-1)
        beta23 = torch.sigmoid(self.beta23_head(h)).squeeze(-1)

        return {
            'T1': T1, 'T2': T2,
            'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3,
            'beta12': beta12, 'beta23': beta23,
            'T1_probs': T1_probs, 'T2_probs': T2_probs,
            'z': z  # task embedding for logging
        }

    def get_config_dict(
        self,
        descriptor: torch.Tensor
    ) -> Dict[str, float]:
        """Get configuration as plain Python dict (for MOHRL-ci agent).

        Uses hard selection (test-time behavior).
        """
        with torch.no_grad():
            config = self.forward(descriptor, hard=True)

        def _safe(val, default):
            """Extract scalar, replacing NaN with default."""
            v = val.item()
            if v != v:  # NaN check
                return default
            return v

        return {
            'T1': max(1, int(_safe(config['T1'], 10))),
            'T2': max(1, int(_safe(config['T2'], 5))),
            'gamma1': _safe(config['gamma1'], 0.99),
            'gamma2': _safe(config['gamma2'], 0.99),
            'gamma3': _safe(config['gamma3'], 0.99),
            'beta12': _safe(config['beta12'], 0.5),
            'beta23': _safe(config['beta23'], 0.5),
        }

    def step_annealing(self):
        """Advance Gumbel-softmax temperature annealing."""
        self.T1_selector.step_annealing()
        self.T2_selector.step_annealing()
