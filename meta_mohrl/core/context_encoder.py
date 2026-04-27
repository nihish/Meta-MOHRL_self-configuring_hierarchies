"""
Context encoder and memory module for MOHRL-ci.

Implements context-aware real-time adaptability:
- ContextEncoder: Encodes environmental features into context vector cx_t
- MemoryModule: External memory M_t with episodic information
- Augmented state construction: s̃_t = [s_t || cx_t || M_t]

From Section IV-A of MOHRL-ci paper.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from collections import deque


class ContextEncoder(nn.Module):
    """Encodes environmental context features into a compact vector.
    
    cx_t encodes both static and dynamic descriptors from:
    - Environment (density, phase, neighbors)
    - System feedback (recent rewards, goal achievement)
    - Agent history (recent states, actions)
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int = 64,
        history_len: int = 10
    ):
        super().__init__()
        self.context_dim = context_dim
        self.history_len = history_len

        # Encode current observation features
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, context_dim),
            nn.ReLU()
        )

        # Encode recent history via 1D convolution
        self.history_encoder = nn.Sequential(
            nn.Conv1d(obs_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Fuse current + historical context
        self.fusion = nn.Sequential(
            nn.Linear(context_dim + 64, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU()
        )

    def forward(
        self,
        obs: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            obs: current observation [batch, obs_dim]
            history: recent observations [batch, history_len, obs_dim]
        Returns:
            context: context vector [batch, context_dim]
        """
        current_ctx = self.obs_encoder(obs)  # [batch, context_dim]

        if history is not None and history.size(1) > 0:
            # [batch, obs_dim, history_len] for Conv1d
            hist_input = history.transpose(1, 2)
            hist_ctx = self.history_encoder(hist_input).squeeze(-1)  # [batch, 64]
        else:
            hist_ctx = torch.zeros(
                obs.size(0), 64, device=obs.device
            )

        combined = torch.cat([current_ctx, hist_ctx], dim=-1)
        return self.fusion(combined)


class MemoryModule:
    """External episodic memory M_t for long-horizon reasoning.
    
    Stores experience tuples: M_t = {(s_i, a_i, r_i, cx_i, h_i)}
    With priority-based filtering and aging mechanism.
    
    From MOHRL-ci Eq. (5): M_t ← M_{t-1} ∪ {(s_t, a_t, r_t, cx_t, h_t)}
    """

    def __init__(
        self,
        capacity: int = 500,
        obs_dim: int = 48,
        context_dim: int = 64,
        memory_dim: int = 64,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.memory_dim = memory_dim
        self.device = device

        # Memory tracks
        self.states: deque = deque(maxlen=capacity)
        self.contexts: deque = deque(maxlen=capacity)
        self.rewards: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

        # Memory encoder: compresses recent memory into fixed-size vector
        self.memory_encoder = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, memory_dim)
        ).to(device)

    def add(
        self,
        state: np.ndarray,
        context: np.ndarray,
        reward: float,
        priority: float = 1.0
    ):
        """Add a new tuple to memory."""
        self.states.append(state)
        self.contexts.append(context)
        self.rewards.append(reward)
        self.priorities.append(priority)

    def get_memory_embedding(
        self,
        current_state: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """Retrieve a compressed memory embedding.
        
        Uses k most recent/relevant entries to build memory representation.
        
        Args:
            current_state: current observation [batch, obs_dim]
            k: number of memory entries to use
        Returns:
            memory_emb: memory embedding [batch, memory_dim]
        """
        if len(self.states) == 0:
            batch_size = current_state.size(0)
            return torch.zeros(
                batch_size, self.memory_dim,
                device=self.device
            )

        # Get k most recent entries
        k = min(k, len(self.states))
        recent_states = list(self.states)[-k:]
        recent_contexts = list(self.contexts)[-k:]

        # Stack and encode
        states_t = torch.FloatTensor(np.array(recent_states)).to(self.device)
        contexts_t = torch.FloatTensor(np.array(recent_contexts)).to(self.device)

        combined = torch.cat([states_t, contexts_t], dim=-1)  # [k, obs+ctx]
        encoded = self.memory_encoder(combined)  # [k, memory_dim]

        # Average over memory entries
        memory_emb = encoded.mean(dim=0, keepdim=True)  # [1, memory_dim]

        # Expand to batch size
        batch_size = current_state.size(0)
        return memory_emb.expand(batch_size, -1)

    def reset(self):
        """Reset memory for new episode."""
        self.states.clear()
        self.contexts.clear()
        self.rewards.clear()
        self.priorities.clear()


def build_augmented_state(
    obs: torch.Tensor,
    context: torch.Tensor,
    memory: torch.Tensor,
    goal: Optional[torch.Tensor] = None,
    feedback: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Construct augmented state s̃_t = [s_t || cx_t || M_t || g || f_t].
    
    From Eq. (3) in Meta-MOHRL paper:
    s^l_t = [s_t || h^l_{t-1} || g^{l+1}_t]
    
    Extended with context and memory from MOHRL-ci:
    s̃_t = (s_t, cx_t, M_t)
    
    Args:
        obs: raw observation [batch, obs_dim]
        context: context vector [batch, context_dim]
        memory: memory embedding [batch, memory_dim]
        goal: subgoal from higher level [batch, goal_dim] (optional)
        feedback: feedback from lower level [batch, feedback_dim] (optional)
    Returns:
        augmented_state: [batch, total_dim]
    """
    components = [obs, context, memory]
    if goal is not None:
        components.append(goal)
    if feedback is not None:
        components.append(feedback)
    return torch.cat(components, dim=-1)
