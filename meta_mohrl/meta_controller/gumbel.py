"""
Gumbel-Softmax selector for differentiable discrete commitment interval selection.

From Eq. (10) in Meta-MOHRL paper:
p̂_k = exp((f_T(z) + g_k) / τ) / Σ exp((f_T(z) + g_k') / τ)

Where g_k ~ Gumbel(0,1), τ anneals from 1.0 to 0.5 over 500 iterations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GumbelSoftmaxSelector(nn.Module):
    """Differentiable discrete selection via Gumbel-Softmax.

    Uses straight-through estimator: hard argmax in forward pass,
    soft probabilities in backward pass for gradient flow.
    """

    def __init__(
        self,
        input_dim: int,
        candidates: List[int],
        tau_start: float = 5.0,
        tau_min: float = 0.5,
        decay_rate: float = 0.9
    ):
        super().__init__()
        self.candidates = candidates
        self.num_candidates = len(candidates)
        self.candidates_tensor = None  # lazy init on device

        # Learnable logit head
        self.logit_head = nn.Linear(input_dim, self.num_candidates)

        # Temperature annealing
        self.tau = tau_start
        self.tau_min = tau_min
        self.decay_rate = decay_rate
        self._current_step = 0

    def forward(
        self,
        z: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a commitment interval.

        Args:
            z: task embedding from meta-controller backbone [batch, input_dim]
            hard: if True, use hard argmax (test time)
        Returns:
            selected_T: selected interval value [batch] (continuous relaxation or hard)
            probs: selection probabilities [batch, num_candidates]
        """
        # Lazy init candidates tensor
        if self.candidates_tensor is None or self.candidates_tensor.device != z.device:
            self.candidates_tensor = torch.FloatTensor(self.candidates).to(z.device)

        logits = self.logit_head(z)  # [batch, num_candidates]
        tau = self.get_temperature()

        if hard or not self.training:
            # Hard selection: argmax
            idx = logits.argmax(dim=-1)  # [batch]
            selected_T = self.candidates_tensor[idx]
            probs = F.softmax(logits, dim=-1)
        else:
            # Gumbel-softmax with straight-through
            gumbel_probs = F.gumbel_softmax(logits, tau=tau, hard=False)
            # Continuous relaxation: weighted sum of candidates
            selected_T = (gumbel_probs * self.candidates_tensor.unsqueeze(0)).sum(dim=-1)
            probs = gumbel_probs

            # Straight-through for environment stepping
            # (hard value in forward, soft gradient in backward)
            hard_idx = probs.argmax(dim=-1)
            hard_T = self.candidates_tensor[hard_idx]
            selected_T = hard_T + (selected_T - selected_T.detach())

        return selected_T, probs

    def get_temperature(self) -> float:
        """Get current annealing temperature."""
        return max(self.tau_min, self.tau)

    def step_annealing(self):
        """Advance annealing schedule by one step."""
        self.tau *= self.decay_rate
        self._current_step += 1

    def get_hard_selection(self, z: torch.Tensor) -> int:
        """Get hard integer selection (for environment use)."""
        logits = self.logit_head(z)
        idx = logits.argmax(dim=-1).item()
        return self.candidates[idx]
