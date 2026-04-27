"""
MOHRL-ci Agent: Full 3-level hierarchical RL agent (Inner Loop).

Integrates:
- High-level (L1): Strategic goals every T¹ steps → maximize speed
- Mid-level (L2): Tactical subgoals every T² steps → minimize waiting
- Low-level (L3): Primitive actions every step → minimize queue
- Context-augmented states, bidirectional feedback, Pareto optimization

Implements Algorithm 2 from the MOHRL-ci paper.
Accepts configuration vector c = (T¹, T², γ¹, γ², γ³, β¹², β²³) from meta-controller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List

from meta_mohrl.config import MOHRLConfig
from meta_mohrl.mohrl_ci.high_level import HighLevelPolicy
from meta_mohrl.mohrl_ci.mid_level import MidLevelPolicy
from meta_mohrl.mohrl_ci.low_level import LowLevelPolicy
from meta_mohrl.mohrl_ci.feedback import BidirectionalFeedback
from meta_mohrl.core.context_encoder import ContextEncoder, MemoryModule, build_augmented_state
from meta_mohrl.core.replay_buffer import HierarchicalReplayBuffer
from meta_mohrl.core.pareto import ParetoFront


class MOHRLciAgent:
    """Multi-Objective Hierarchical RL with Contextual Intelligence.

    Three-level hierarchy with bidirectional feedback and Pareto optimization.
    Configuration parameters can be overridden by the meta-controller.
    """

    def __init__(self, obs_dim: int, num_actions: int, config: MOHRLConfig):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = config.device

        # --- Configuration (may be overridden by meta-controller) ---
        self.T1 = config.T1_default
        self.T2 = config.T2_default
        self.gamma1 = config.gamma1_default
        self.gamma2 = config.gamma2_default
        self.gamma3 = config.gamma3_default
        self.beta12 = config.beta12_default
        self.beta23 = config.beta23_default

        # --- Context & Memory ---
        self.context_encoder = ContextEncoder(
            obs_dim, config.context_dim, config.context_window
        ).to(self.device)
        self.memory = MemoryModule(
            config.memory_capacity, obs_dim, config.context_dim,
            config.memory_dim, self.device
        )

        # Augmented state dim = obs + context + memory
        self.aug_state_dim = obs_dim + config.context_dim + config.memory_dim

        # --- Three-Level Hierarchy ---
        self.high_level = HighLevelPolicy(
            self.aug_state_dim,
            num_goals=8,
            num_objectives=config.num_objectives,
            hidden_dim=config.actor_hidden,
            lstm_hidden=config.lstm_hidden,
            goal_dim=config.goal_dim,
            lr=config.actor_lr,
            device=self.device
        )

        self.mid_level = MidLevelPolicy(
            self.aug_state_dim,
            goal_input_dim=config.goal_dim,
            num_subgoals=6,
            num_objectives=config.num_objectives,
            hidden_dim=config.actor_hidden,
            lstm_hidden=config.lstm_hidden,
            subgoal_dim=config.goal_dim,
            lr=config.actor_lr,
            device=self.device
        )

        self.low_level = LowLevelPolicy(
            self.aug_state_dim,
            goal_dim=config.goal_dim,
            subgoal_dim=config.goal_dim,
            num_actions=num_actions,
            num_objectives=config.num_objectives,
            hidden_dim=config.actor_hidden,
            lstm_hidden=config.lstm_hidden,
            lr=config.critic_lr,
            device=self.device
        )

        # --- Bidirectional Feedback ---
        self.feedback = BidirectionalFeedback(
            self.aug_state_dim,
            config.goal_dim,
            num_actions,
            config.num_objectives,
            feedback_dim=64
        ).to(self.device)

        # --- Replay Buffers ---
        self.replay_buffer = HierarchicalReplayBuffer(
            config.replay_buffer_size, self.device
        )

        # --- Pareto Fronts (one per level) ---
        self.pareto_fronts = {
            'high': ParetoFront(config.num_objectives),
            'mid': ParetoFront(config.num_objectives),
            'low': ParetoFront(config.num_objectives),
        }

        # --- Episode state ---
        self.current_goal_idx = None
        self.current_goal_emb = None
        self.current_subgoal_idx = None
        self.current_subgoal_emb = None
        self.step_counter = 0
        self.obs_history = []
        self.feedback_history = []
        self.episode_rewards = []

    def set_configuration(self, config_vector: Dict[str, float]):
        """Set hierarchical configuration from meta-controller.

        config_vector: {T1, T2, gamma1, gamma2, gamma3, beta12, beta23}
        """
        self.T1 = int(config_vector.get('T1', self.T1))
        self.T2 = int(config_vector.get('T2', self.T2))
        self.gamma1 = config_vector.get('gamma1', self.gamma1)
        self.gamma2 = config_vector.get('gamma2', self.gamma2)
        self.gamma3 = config_vector.get('gamma3', self.gamma3)
        self.beta12 = config_vector.get('beta12', self.beta12)
        self.beta23 = config_vector.get('beta23', self.beta23)

    def reset(self):
        """Reset episode state."""
        self.current_goal_idx = None
        self.current_goal_emb = None
        self.current_subgoal_idx = None
        self.current_subgoal_emb = None
        self.step_counter = 0
        self.obs_history = []
        self.feedback_history = []
        self.episode_rewards = []
        self.memory.reset()
        self.high_level.reset_lstm()
        self.mid_level.reset_lstm()
        self.low_level.reset_lstm()

    def _build_augmented_state(self, obs: np.ndarray) -> torch.Tensor:
        """Build s̃_t = [s_t || cx_t || M_t]."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Build history tensor for context encoding
        if len(self.obs_history) > 0:
            hist = np.array(self.obs_history[-self.config.context_window:])
            hist_t = torch.FloatTensor(hist).unsqueeze(0).to(self.device)
        else:
            hist_t = None

        # Context encoding
        context = self.context_encoder(obs_t, hist_t)

        # Memory embedding
        mem_emb = self.memory.get_memory_embedding(obs_t)

        # Build augmented state
        aug_state = build_augmented_state(obs_t, context, mem_emb)
        return aug_state

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        epsilon: float = 0.0
    ) -> int:
        """Select action using 3-level hierarchy.

        At step t:
        1. If t % T1 == 0: high-level selects new goal
        2. If t % T2 == 0: mid-level selects new subgoal (conditioned on goal)
        3. Every step: low-level selects primitive action

        Args:
            obs: raw observation from environment
            deterministic: greedy action selection
        Returns:
            action: primitive action index
        """
        aug_state = self._build_augmented_state(obs)

        # --- High-level decision (every T1 steps) ---
        if self.step_counter % self.T1 == 0 or self.current_goal_emb is None:
            goal_idx, goal_emb, _ = self.high_level.select_goal(
                aug_state, deterministic
            )
            self.current_goal_idx = goal_idx
            self.current_goal_emb = goal_emb

        # --- Mid-level decision (every T2 steps) ---
        if self.step_counter % self.T2 == 0 or self.current_subgoal_emb is None:
            sg_idx, sg_emb, _ = self.mid_level.select_subgoal(
                aug_state, self.current_goal_emb, deterministic
            )
            self.current_subgoal_idx = sg_idx
            self.current_subgoal_emb = sg_emb

        # --- Low-level decision (every step) ---
        action, log_prob = self.low_level.select_action(
            aug_state, self.current_goal_emb, self.current_subgoal_emb,
            deterministic, epsilon
        )

        self.step_counter += 1
        self.obs_history.append(obs.copy())

        return action.item()

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward_vector: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        omega: Optional[np.ndarray] = None
    ):
        """Process transition and compute hierarchical rewards.

        Reward decomposition with bidirectional feedback:
        r³ = ⟨ω³, R⟩
        r² = ⟨ω², R⟩ + β²³ · V³(s')
        r¹ = ⟨ω¹, R⟩ + β¹² · V²(s')
        """
        if omega is None:
            omega = np.array(self.config.default_omega)

        aug_state = self._build_augmented_state(obs)
        aug_next = self._build_augmented_state(next_obs)

        reward_t = torch.FloatTensor(reward_vector).unsqueeze(0).to(self.device)
        omega_t = torch.FloatTensor(omega).to(self.device)

        # --- Compute bidirectional feedback ---
        action_onehot = np.zeros(self.num_actions)
        action_onehot[action] = 1.0
        action_oh_t = torch.FloatTensor(action_onehot).unsqueeze(0).to(self.device)

        feedback_vec = self.feedback.compute_feedback(
            aug_state,
            self.current_goal_emb if self.current_goal_emb is not None
                else torch.zeros(1, self.config.goal_dim, device=self.device),
            action_oh_t,
            reward_t
        )
        self.feedback_history.append(feedback_vec.detach().cpu().numpy().flatten())

        # --- Compute per-level rewards (Eqs. 4-6 from Meta-MOHRL) ---
        # r³ = ⟨ω³, R⟩ (low-level: just scalarized reward)
        r3 = (omega_t * reward_t.squeeze()).sum().item()

        # r² = ⟨ω², R⟩ + β²³ · V³(s')  (mid-level with bottom-up feedback)
        v3_next = self.low_level.get_value(
            aug_next,
            self.current_goal_emb if self.current_goal_emb is not None
                else torch.zeros(1, self.config.goal_dim, device=self.device),
            self.current_subgoal_emb if self.current_subgoal_emb is not None
                else torch.zeros(1, self.config.goal_dim, device=self.device)
        ).item()
        r2 = (omega_t * reward_t.squeeze()).sum().item() + self.beta23 * v3_next

        # r¹ = ⟨ω¹, R⟩ + β¹² · V²(s') (high-level with bottom-up feedback)
        v2_next = self.mid_level.get_value(
            aug_next,
            self.current_goal_emb if self.current_goal_emb is not None
                else torch.zeros(1, self.config.goal_dim, device=self.device)
        ).item()
        r1 = (omega_t * reward_t.squeeze()).sum().item() + self.beta12 * v2_next

        # --- Store in replay buffers ---
        aug_s_np = aug_state.detach().cpu().numpy().flatten()
        aug_ns_np = aug_next.detach().cpu().numpy().flatten()

        # Low-level: store every step
        low_state = np.concatenate([
            aug_s_np,
            self.current_goal_emb.detach().cpu().numpy().flatten()
                if self.current_goal_emb is not None
                else np.zeros(self.config.goal_dim),
            self.current_subgoal_emb.detach().cpu().numpy().flatten()
                if self.current_subgoal_emb is not None
                else np.zeros(self.config.goal_dim),
        ])
        low_next = np.concatenate([
            aug_ns_np,
            self.current_goal_emb.detach().cpu().numpy().flatten()
                if self.current_goal_emb is not None
                else np.zeros(self.config.goal_dim),
            self.current_subgoal_emb.detach().cpu().numpy().flatten()
                if self.current_subgoal_emb is not None
                else np.zeros(self.config.goal_dim),
        ])
        self.replay_buffer.push(
            'low', state=low_state, action=action,
            reward=reward_vector * np.array([0.0, 0.0, 1.0]),  # queue objective
            next_state=low_next, done=done, omega=omega
        )

        # Mid-level: store every T2 steps
        if self.step_counter % self.T2 == 0:
            mid_state = np.concatenate([
                aug_s_np,
                self.current_goal_emb.detach().cpu().numpy().flatten()
                    if self.current_goal_emb is not None
                    else np.zeros(self.config.goal_dim)
            ])
            mid_next = np.concatenate([
                aug_ns_np,
                self.current_goal_emb.detach().cpu().numpy().flatten()
                    if self.current_goal_emb is not None
                    else np.zeros(self.config.goal_dim)
            ])
            self.replay_buffer.push(
                'mid',
                state=mid_state,
                action=self.current_subgoal_idx.item()
                    if self.current_subgoal_idx is not None else 0,
                reward=reward_vector * np.array([0.0, 1.0, 0.0]),
                next_state=mid_next, done=done, omega=omega
            )

        # High-level: store every T1 steps
        if self.step_counter % self.T1 == 0:
            self.replay_buffer.push(
                'high', state=aug_s_np,
                action=self.current_goal_idx.item()
                    if self.current_goal_idx is not None else 0,
                reward=reward_vector * np.array([1.0, 0.0, 0.0]),
                next_state=aug_ns_np, done=done, omega=omega
            )

        # Update memory
        context_np = aug_state[:, self.obs_dim:self.obs_dim + self.config.context_dim]\
            .detach().cpu().numpy().flatten()
        total_reward = reward_vector.sum()
        self.memory.add(obs, context_np, total_reward, abs(total_reward) + 0.1)

        self.episode_rewards.append(reward_vector)

    def train_step(self) -> Dict[str, float]:
        """Perform one training step for all three levels.

        Returns dict of losses for logging.
        """
        losses = {}
        batch_size = self.config.batch_size

        # Sample preference vectors from Pareto fronts
        omega_high = torch.FloatTensor(
            self.pareto_fronts['high'].sample_preference()
        ).to(self.device)
        omega_mid = torch.FloatTensor(
            self.pareto_fronts['mid'].sample_preference()
        ).to(self.device)
        omega_low = torch.FloatTensor(
            self.pareto_fronts['low'].sample_preference()
        ).to(self.device)

        # --- Train high-level ---
        if self.replay_buffer.level_size('high') >= batch_size:
            batch = self.replay_buffer.sample('high', batch_size)
            high_losses = self.high_level.update(
                batch, omega_high, self.gamma1,
                self.config.target_update_tau, self.config.grad_clip
            )
            losses.update(high_losses)

        # --- Train mid-level ---
        if self.replay_buffer.level_size('mid') >= batch_size:
            batch = self.replay_buffer.sample('mid', batch_size)
            mid_losses = self.mid_level.update(
                batch, omega_mid, self.gamma2,
                self.config.target_update_tau, self.config.grad_clip
            )
            losses.update(mid_losses)

        # --- Train low-level ---
        if self.replay_buffer.level_size('low') >= batch_size:
            batch = self.replay_buffer.sample('low', batch_size)
            low_losses = self.low_level.update(
                batch, omega_low, self.gamma3,
                self.config.target_update_tau, self.config.grad_clip
            )
            losses.update(low_losses)

        return losses

    def update_pareto_fronts(self):
        """Update Pareto fronts with episode results."""
        if len(self.episode_rewards) == 0:
            return

        # Aggregate episode rewards
        ep_rewards = np.array(self.episode_rewards)
        avg_rewards = ep_rewards.mean(axis=0)

        for level in ['high', 'mid', 'low']:
            omega = self.pareto_fronts[level].sample_preference()
            self.pareto_fronts[level].add_solution(avg_rewards, omega)

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters (for meta-gradient computation)."""
        params = []
        params.extend(self.high_level.parameters())
        params.extend(self.mid_level.parameters())
        params.extend(self.low_level.parameters())
        params.extend(self.feedback.parameters())
        params.extend(self.context_encoder.parameters())
        return params

    def compute_inner_objective(self) -> torch.Tensor:
        """Compute J_inner for meta-gradient.

        J_MOHRL-ci(θ) = Σ_l E_{ω~P̂^l}[E_{π^l}[Σ γ^t r^l_t]]
        """
        if len(self.episode_rewards) == 0:
            return torch.tensor(0.0, device=self.device)

        ep_rewards = np.array(self.episode_rewards)

        # Compute discounted returns per level using their gamma
        total = 0.0
        for k in range(self.config.num_objectives):
            rewards_k = ep_rewards[:, k]
            gammas = [self.gamma1, self.gamma2, self.gamma3]
            gamma_k = gammas[k]
            discounted = sum(
                gamma_k ** t * r for t, r in enumerate(rewards_k)
            )
            total += discounted / len(rewards_k)

        return torch.tensor(total, device=self.device, requires_grad=False)

    def get_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        for level in ['high', 'mid', 'low']:
            pf = self.pareto_fronts[level]
            metrics[f'{level}_hypervolume'] = pf.compute_hypervolume()
            metrics[f'{level}_pareto_size'] = pf.cardinality()

        if len(self.episode_rewards) > 0:
            ep_r = np.array(self.episode_rewards)
            metrics['avg_speed_reward'] = ep_r[:, 0].mean()
            metrics['avg_waiting_reward'] = ep_r[:, 1].mean()
            metrics['avg_queue_reward'] = ep_r[:, 2].mean()
            metrics['total_reward'] = ep_r.sum()

        return metrics
