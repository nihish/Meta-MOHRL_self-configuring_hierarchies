# Meta-MOHRL (Multi-Objective Hierarchical Reinforcement Learning)

This repository contains the clean implementation of the Meta-MOHRL algorithm for multi-objective traffic signal control, along with state-of-the-art baseline models (HIRO, HiPPO, DUSDi, MOSMAC). It includes the necessary environment configurations, the main training and evaluation scripts, and the official Version 4 experiment results.

## Setup Instructions

### 1. Requirements

First, ensure you have Python 3.9+ installed. Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### 2. Environment (SUMO)

The environment relies on Eclipse SUMO (Simulation of Urban MObility). 
1. Install SUMO from [the official website](https://eclipse.dev/sumo/).
2. Make sure you set the `SUMO_HOME` environment variable to your SUMO installation directory. For example, on Windows:
```bash
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```
*(You may need to update the `sumo_home` path in `meta_mohrl/config.py` depending on your installation location).*

## Running the Experiment

We provide two primary scripts for training:

### Option 1: Standard Training Loop (`train.py`)
This script runs the standard bilevel training loop (Algorithm 1) for the Meta-MOHRL framework.

```bash
python train.py --episodes 100 --meta-iters 50 --inner-steps 5 --net single --ablation full
```
**Arguments:**
- `--episodes`: Episodes per inner loop (default: 100).
- `--meta-iters`: Outer loop iterations (default: 50).
- `--inner-steps`: Inner gradient steps (default: 5).
- `--net`: Network topology (`single` or `grid2x2`).
- `--ablation`: Run different ablations (`full`, `no_meta`, `no_feedback`, `no_context`, `no_pareto`).
- `--use-gui`: Add this flag to visualize the SUMO simulation.

### Option 2: Full Experiment Runner (`run_experiment.py`)
This script runs an automated long-running experiment (e.g., 4,000 episodes) for Meta-MOHRL, its ablation variants, and all baseline algorithms. It saves a comprehensive JSON record and evaluates the learned policies on 15 randomized topologies.

```bash
python run_experiment.py
```

### Option 3: Train Baseline Algorithms (`train_baselines.py`)
This script allows you to train the individual baseline models (HIRO, HiPPO, DUSDi, MOSMAC) independently.

```bash
python train_baselines.py
```

## Included Results

The `version_4_results/` folder contains the final publication-quality graphs from the full 4,000-episode experiment. These plots showcase convergence metrics, Pareto fronts, hypervolume optimization, and radar comparisons across all baselines and variants.

## Hyperparameters and Environment Details

### Meta-Controller Hyperparameters
- Task Descriptor Dimension: 5
- Embedding Dimension: 64
- Meta MLP Hidden Units: 128
- Gumbel-Softmax Anneal Steps: 500
- Outer Learning Rate: 1e-3
- Inner Learning Rate: 3e-4
- Meta Gradient Clip: 1.0

### MOHRL-ci (Inner Loop) Hyperparameters
- Hierarchy Levels: 3 (High, Mid, Low)
- Actor/Critic Hidden Units: 256
- LSTM Hidden Units: 128
- Context / Memory Dim: 64
- Goal Dimension: 16
- Memory Capacity: 500 tuples
- Replay Buffer Size: 100,000
- Batch Size: 64
- Actor/Critic Learning Rate: 3e-4

### Environment Configuration
- Objectives: 3 (Speed, Waiting Time, Queue Length)
- Simulation Time: 3600 seconds
- Delta Time (Decision Interval): 5 seconds
- Minimum/Maximum Green Time: 5s / 60s
- Yellow Time: 2s
- Actions: 4 signal phases
- Vehicle Mix: 70% cars, 20% trucks, 10% buses
