"""
Configuration dataclasses for Meta-MOHRL framework.
All hyperparameters from both papers consolidated here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class MOHRLConfig:
    """MOHRL-ci inner loop configuration (from MOHRL-ci paper)."""
    # --- Hierarchy Structure ---
    num_levels: int = 3
    # Default commitment intervals (overridden by meta-controller)
    T1_default: int = 10          # high-level: every T1 env steps
    T2_default: int = 5           # mid-level: every T2 env steps
    T3: int = 1                   # low-level: every step (fixed)

    # --- Network Architecture ---
    obs_dim: int = 48             # observation dim (set by environment)
    context_dim: int = 64         # context encoder output dim
    memory_dim: int = 64          # memory embedding dim
    lstm_hidden: int = 128        # per-level LSTM hidden state dim
    goal_dim: int = 16            # subgoal embedding dimension
    actor_hidden: int = 256       # actor MLP hidden units
    critic_hidden: int = 256      # critic MLP hidden units

    # --- Multi-Objective ---
    num_objectives: int = 3       # K = 3  (speed, waiting, queue)
    # Default preference vectors (uniform)
    default_omega: List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])

    # --- Discount Factors (overridden by meta-controller) ---
    gamma1_default: float = 0.99
    gamma2_default: float = 0.99
    gamma3_default: float = 0.99

    # --- Feedback Gains (overridden by meta-controller) ---
    beta12_default: float = 0.5   # high ← mid
    beta23_default: float = 0.5   # mid  ← low

    # --- Learning ---
    actor_lr: float = 3e-4        # high-level LR (stabilized for 4k episodes)
    critic_lr: float = 3e-4       # low-level LR / critic LR (stabilized for 4k episodes)
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    target_update_tau: float = 0.005  # soft target network update
    grad_clip: float = 1.0

    # --- Action Space ---
    num_actions: int = 4          # 4 signal phases (set by environment)

    # --- Context / Memory ---
    memory_capacity: int = 500    # max tuples in external memory M
    context_window: int = 10      # recent steps for context encoding

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MetaConfig:
    """Meta-controller outer loop configuration (from Meta-MOHRL paper)."""
    # --- Task Encoder ---
    task_descriptor_dim: int = 5  # d = [n/n_max, H/H_max, K/K_max, mu, sigma^2]
    embedding_dim: int = 64       # z dimension (d_z)
    encoder_hidden: int = 128     # encoder MLP hidden
    n_max: int = 40               # max agent count for normalization
    H_max: int = 4000             # max horizon for normalization
    K_max: int = 5                # max objectives for normalization

    # --- Meta-Controller ---
    meta_hidden: int = 128        # meta-controller MLP hidden
    commitment_candidates: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50]
    )
    gamma_range: Tuple[float, float] = (0.90, 0.999)  # γ ∈ (0.90, 0.999)

    # --- Gumbel-Softmax ---
    gumbel_tau_start: float = 1.0
    gumbel_tau_min: float = 0.5
    gumbel_anneal_steps: int = 500  # anneal over 500 meta-iterations

    # --- Bilevel Optimization ---
    inner_steps: int = 5          # K = 5 inner gradient steps
    alpha_in: float = 3e-4        # inner learning rate
    alpha_out: float = 1e-3       # outer (meta) learning rate
    soft_update_lambda: float = 0.1  # θ₀ soft-update weight

    # --- Training ---
    num_meta_iterations: int = 50   # prototype: 50 (paper: 500)
    meta_grad_clip: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EnvConfig:
    """SUMO-RL environment configuration."""
    # --- SUMO Paths ---
    sumo_home: str = r"C:\Program Files (x86)\Eclipse\Sumo"
    net_file: str = ""            # set at runtime
    route_file: str = ""          # set at runtime
    use_gui: bool = False
    num_seconds: int = 3600       # 1 hour simulation
    delta_time: int = 5           # seconds between agent decisions
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 60

    # --- Multi-Agent ---
    single_agent: bool = False    # multi-agent by default
    num_agents: int = 4           # 2x2 grid → 4 traffic lights

    # --- Vehicle Types ---
    vehicle_types: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus"]
    )
    vehicle_ratios: List[float] = field(
        default_factory=lambda: [0.7, 0.2, 0.1]  # 70% cars, 20% trucks, 10% buses
    )


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    # --- General ---
    seed: int = 42
    num_episodes: int = 200       # episodes per inner-loop training
    max_steps_per_episode: int = 3600  # 1 hour at 1s/step
    eval_interval: int = 10       # evaluate every N episodes
    save_interval: int = 50       # save model every N episodes

    # --- Logging ---
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    results_dir: str = "results"

    # --- Pareto ---
    num_preference_samples: int = 20  # ω vectors for Pareto front
    pareto_archive_size: int = 100    # max solutions in archive

    # --- Ablation ---
    ablation_mode: str = "full"  # "full", "no_meta", "no_feedback", "no_context", "no_pareto"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
