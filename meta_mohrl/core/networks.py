"""
Neural network architectures for Meta-MOHRL.
- ActorNetwork: Policy network with LSTM context, outputs discrete action logits
- CriticNetwork: Multi-objective Q-value estimation (K-dimensional output)
- LSTMEncoder: Per-level LSTM cell for temporal context encoding
- FeedbackEncoder: φ(s̃, g, a, r) → feedback vector f_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """Per-level LSTM cell for temporal context encoding.
    
    Produces hidden state h^l_t from raw state, capturing temporal dependencies.
    From Eq. (3) in Meta-MOHRL paper: s^l_t = [s_t || h^l_{t-1} || g^{l+1}_t]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: input tensor [batch, input_dim]
            hx: (h, c) tuple of previous hidden/cell states
        Returns:
            h: hidden state [batch, hidden_dim]
            (h, c): updated hidden/cell states
        """
        batch_size = x.size(0)
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h, c = hx

        h_new, c_new = self.lstm_cell(x, (h, c))
        h_new = self.layer_norm(h_new)
        return h_new, (h_new, c_new)


class ActorNetwork(nn.Module):
    """Hierarchical actor (policy) network.
    
    Takes augmented state s̃_t = [s_t || h^l || g^{l+1}] and outputs
    action/subgoal logits for discrete selection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        lstm_hidden: int = 128
    ):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(input_dim, lstm_hidden)
        # MLP on top of LSTM output + goal conditioning
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        state: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            state: augmented state [batch, input_dim]
            lstm_state: previous LSTM hidden state
        Returns:
            logits: action/subgoal logits [batch, output_dim]
            lstm_state: updated LSTM state
        """
        h, lstm_state = self.lstm_encoder(state, lstm_state)
        logits = self.net(h)
        return logits, lstm_state

    def get_action(
        self,
        state: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from policy.
        
        Returns:
            action: selected action [batch]
            log_prob: log probability of action [batch]
            lstm_state: updated LSTM state
        """
        logits, lstm_state = self.forward(state, lstm_state)
        # Defensive: clamp logits and replace NaN to prevent softmax failure
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = logits.clamp(-20.0, 20.0)
        probs = F.softmax(logits, dim=-1)
        # Extra safety: ensure valid probability distribution
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, lstm_state


class CriticNetwork(nn.Module):
    """Multi-objective critic network.
    
    Estimates Q-values for each objective: Q(s̃, a) → R^K
    Supports preference-conditioned value: Q(s, a, ω) = Σ ωᵢ Qᵢ(s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_objectives: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_objectives = num_objectives

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Per-objective Q-value heads
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_objectives)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: augmented state [batch, state_dim]
            action: one-hot action [batch, action_dim]
        Returns:
            q_values: per-objective Q [batch, num_objectives]
        """
        x = torch.cat([state, action], dim=-1)
        features = self.shared(x)
        q_values = torch.cat([head(features) for head in self.q_heads], dim=-1)
        return q_values

    def scalarized_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        omega: torch.Tensor
    ) -> torch.Tensor:
        """Preference-weighted Q-value: Q(s,a,ω) = ⟨ω, Q(s,a)⟩
        
        Args:
            omega: preference vector [batch, num_objectives] or [num_objectives]
        Returns:
            scalar Q-value [batch]
        """
        q_vec = self.forward(state, action)  # [batch, K]
        if omega.dim() == 1:
            omega = omega.unsqueeze(0).expand_as(q_vec)
        return (q_vec * omega).sum(dim=-1)


class ValueNetwork(nn.Module):
    """State value network V(s̃) for each level.
    
    Used in bidirectional feedback: r^l = ⟨ω^l, R⟩ + β^{l,l+1} V^{l+1}(s')
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns V(s̃) [batch, 1]"""
        return self.net(state)


class FeedbackEncoder(nn.Module):
    """Feedback encoder φ for bidirectional information flow.
    
    Computes f_t = φ(s̃_t, g_h, a_t, r_t) capturing execution quality.
    From Eq. (6) in MOHRL-ci paper.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        reward_dim: int,
        output_dim: int = 64
    ):
        super().__init__()
        total_input = state_dim + goal_dim + action_dim + reward_dim
        self.net = nn.Sequential(
            nn.Linear(total_input, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.Tanh()  # bounded feedback
        )

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: augmented state [batch, state_dim]
            goal: current subgoal [batch, goal_dim]
            action: executed action (one-hot) [batch, action_dim]
            reward: reward vector [batch, reward_dim]
        Returns:
            feedback: feedback vector [batch, output_dim]
        """
        x = torch.cat([state, goal, action, reward], dim=-1)
        return self.net(x)
