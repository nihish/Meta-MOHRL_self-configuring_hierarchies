"""
Hierarchical experience replay buffer for MOHRL-ci.
Separate buffers D_h, D_{h-1}, D_l for each hierarchical level.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Single-level experience replay buffer with priority support."""

    def __init__(self, capacity: int = 100_000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: np.ndarray,      # multi-objective reward vector
        next_state: np.ndarray,
        done: bool,
        goal: Optional[np.ndarray] = None,
        feedback: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None
    ):
        """Store a transition."""
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            'feedback': feedback,
            'omega': omega
        }
        self.buffer.append(transition)
        # Default priority = max existing + 1 (for prioritized replay)
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_prio)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly."""
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        return self._collate(batch)

    def sample_prioritized(
        self, batch_size: int, alpha: float = 0.6
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling."""
        batch_size = min(batch_size, len(self.buffer))
        priorities = np.array(list(self.priorities))
        probs = priorities ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-0.4)
        weights /= weights.max()

        return self._collate(batch), indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + 1e-6

    def _collate(self, batch: list) -> Dict[str, torch.Tensor]:
        """Collate batch into tensors."""
        result = {}
        result['states'] = torch.FloatTensor(
            np.array([t['state'] for t in batch])
        ).to(self.device)
        result['actions'] = torch.LongTensor(
            np.array([t['action'] for t in batch])
        ).to(self.device)
        result['rewards'] = torch.FloatTensor(
            np.array([t['reward'] for t in batch])
        ).to(self.device)
        result['next_states'] = torch.FloatTensor(
            np.array([t['next_state'] for t in batch])
        ).to(self.device)
        result['dones'] = torch.FloatTensor(
            np.array([t['done'] for t in batch])
        ).to(self.device)

        # Optional fields
        if batch[0]['goal'] is not None:
            result['goals'] = torch.FloatTensor(
                np.array([t['goal'] for t in batch])
            ).to(self.device)
        if batch[0]['feedback'] is not None:
            result['feedbacks'] = torch.FloatTensor(
                np.array([t['feedback'] for t in batch])
            ).to(self.device)
        if batch[0]['omega'] is not None:
            result['omegas'] = torch.FloatTensor(
                np.array([t['omega'] for t in batch])
            ).to(self.device)

        return result

    def __len__(self) -> int:
        return len(self.buffer)


class HierarchicalReplayBuffer:
    """Manages separate replay buffers for each hierarchical level.
    
    D_h (high-level), D_{h-1} (mid-level), D_l (low-level)
    Each stores transitions at its respective timescale.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        device: str = "cpu"
    ):
        self.buffers = {
            'high': ReplayBuffer(capacity, device),
            'mid': ReplayBuffer(capacity, device),
            'low': ReplayBuffer(capacity, device),
        }

    def push(self, level: str, **kwargs):
        """Push transition to specified level buffer."""
        assert level in self.buffers, f"Unknown level: {level}"
        self.buffers[level].push(**kwargs)

    def sample(self, level: str, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample from specified level buffer."""
        return self.buffers[level].sample(batch_size)

    def __len__(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def level_size(self, level: str) -> int:
        return len(self.buffers[level])
