# Meta-MOHRL: Self-Configuring Hierarchies with Meta-Adaptive Hierarchical Reinforcement Learning

> **NeurIPS 2026 Submission**

Meta-MOHRL is a meta-adaptive hierarchical reinforcement learning framework that learns temporal commitment intervals, per-level discount factors, and inter-level feedback gains as a function of task context via bilevel optimization. It wraps a three-level multi-objective cooperative hierarchy with a lightweight meta-controller that executes once per episode.

## Repository Structure

```
Meta-MOHRL-v2/
├── reproduce.py                  # One-command reproduction & claim verification
├── train.py                      # Meta-MOHRL bilevel training (Algorithm 1)
├── train_baselines.py            # Baseline training (HIRO, HiPPO, DUSDi, MOSMAC)
├── run_experiment.py             # Full experiment: all algorithms + evaluation
├── evaluate.py                   # Post-training evaluation utilities
├── plot_results.py               # Version 3 figure generation
├── plot_seeds_and_pareto.py      # 5-seed convergence & Pareto plots
├── requirements.txt              # Python dependencies
│
├── meta_mohrl/                   # Core framework
│   ├── config.py                 # All hyperparameters
│   ├── core/                     # Networks, replay buffer, Pareto utilities
│   ├── meta_controller/          # Bilevel meta-controller + Gumbel-softmax
│   ├── mohrl_ci/                 # 3-level hierarchy (high/mid/low + feedback)
│   └── environment/              # SUMO-RL wrapper + topology generation
│
├── baselines/                    # HIRO, HiPPO, DUSDi, MOSMAC implementations
│
├── sumo_configs/                 # SUMO network files
│   ├── 15_topologies/            # 15 randomized intersection layouts
│   ├── grid2x2.net.xml           # 2×2 grid network
│   └── single_intersection.*     # Single 4-way intersection
│
├── results/                      # 5-seed experiment records (seeds 42–46)
│   └── experiment_record_seed*.json
│
└── figures/                      # Publication figures (Version 3)
    ├── version_3_reward_overall_convergence.png
    ├── version_3_ablation_study.png
    ├── version_3_meta_learning_adaptation.png
    ├── version_3_config_gamma_adaptation.png
    ├── version_3_multi_agent_decentralization.png
    ├── version_3_hypervolume_optimization.png
    ├── version_3_radar_comparison.png
    ├── version_3_reward_distributions.png
    ├── meta_mohrl_5seeds_convergence.png
    └── meta_mohrl_pareto_improved.png
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install SUMO

Install [Eclipse SUMO](https://eclipse.dev/sumo/) and set the `SUMO_HOME` environment variable:

```bash
# Windows
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo

# Linux
export SUMO_HOME=/usr/share/sumo
```

### 3. Verify Manuscript Claims (no training needed)

```bash
python reproduce.py --verify
```

This checks all numerical claims (Q1–Q5) against the saved 5-seed results.

### 4. Regenerate Figures

```bash
python reproduce.py --plot-only
```

### 5. Full Training Reproduction

```bash
python reproduce.py                    # All 5 seeds (~15h per seed)
python reproduce.py --seeds 42         # Single seed (~15h)
```

## Environment

- **Simulation**: SUMO-RL, 4-way signalized intersection, 4 cooperative agents
- **Episode**: 3,600 simulation seconds (200 decision steps at Δt = 5s)
- **Training**: 4,000 episodes per algorithm, 15 meta-iterations
- **Topologies**: 15 randomized layouts (symmetric grids, asymmetric arterials, heterogeneous meshes, bottleneck grids)
- **Objectives**: Speed (Level 1), Waiting Time (Level 2), Queue Length (Level 3)
- **Vehicle Fleet**: Cars (70%), Trucks (20%), Buses (10%)

## Hardware

All experiments were conducted on a consumer laptop:
- **ASUS Zenbook Duo UX8406MA**
- **Intel Core Ultra 9 185H** (16 cores, 22 threads)
- **32 GB RAM**, Intel Arc integrated graphics
- No dedicated GPU required

## Hyperparameters

### Meta-Controller (Outer Loop)
| Parameter | Value |
|---|---|
| Meta learning rate | 5 × 10⁻⁴ |
| Meta iterations | 15 |
| Inner gradient steps (K) | 5 |
| Task descriptor dim | 5 |
| Embedding dim | 64 |
| MLP hidden | 128 |
| Gumbel temp (init → min) | 5.0 → 0.5 |
| T¹ bounds | [10, 50] steps |
| T² bounds | [3, 10] steps |
| γ bounds | (0.90, 0.995) |
| β bounds | (0.0, 1.0) |

### Inner Loop (3-Level Hierarchy)
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3 × 10⁻⁴ |
| Actor/Critic hidden | 256 |
| LSTM hidden | 128 |
| Replay buffer | 100,000 |
| Batch size | 64 |
| Critic Polyak τ | 0.005 |

## Baselines

| Method | Architecture | Reference |
|---|---|---|
| HIRO | 2-level hierarchy, fixed T | Nachum et al., NeurIPS 2018 |
| HiPPO | Multi-level PPO, fixed T | Zhang et al., Pattern Recognition 2025 |
| DUSDi | Unsupervised skill discovery | Hu et al., NeurIPS 2024 |
| MOSMAC | Flat multi-objective MARL | Geng et al., AAMAS 2025 |

## License

This repository is provided for anonymous peer review.
