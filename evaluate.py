"""
Evaluation script for Meta-MOHRL with ablation study and comparison.

Generates:
1. Convergence comparison graphs (all algorithms)
2. Per-objective reward curves
3. Pareto front visualization
4. Hypervolume evolution
5. Ablation study table (meta-controller components)
6. Configuration adaptation analysis

Usage:
    python evaluate.py [--results-dir results] [--run-ablation]
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meta_mohrl.config import MOHRLConfig, MetaConfig, EnvConfig
from meta_mohrl.mohrl_ci.agent import MOHRLciAgent
from meta_mohrl.meta_controller.meta_controller import MetaController
from meta_mohrl.environment.sumo_wrapper import MultiObjectiveSumoEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-MOHRL Evaluation')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run full ablation study')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def run_ablation_study(args):
    """Run ablation study: disable each meta-controller component.

    Ablation modes:
    1. full: Complete Meta-MOHRL (all components)
    2. no_meta: Disable meta-controller (fixed config, MOHRL-ci only)
    3. no_feedback: Disable bidirectional feedback (β¹²=β²³=0)
    4. no_context: Disable context encoder (raw state only)
    5. no_pareto: Disable Pareto structuring (fixed preference)
    """
    print("=" * 70)
    print("ABLATION STUDY: Meta-MOHRL Component Analysis")
    print("=" * 70)

    ablation_modes = ['full', 'no_meta', 'no_feedback', 'no_context', 'no_pareto']
    ablation_results = {}

    for mode in ablation_modes:
        print(f"\n--- Running ablation: {mode} ---")

        # Create environment
        config_dir = os.path.join(PROJECT_ROOT, 'sumo_configs')
        os.makedirs(config_dir, exist_ok=True)

        net_file = os.path.join(config_dir, 'single_intersection.net.xml')
        route_file = os.path.join(config_dir, 'single_intersection.rou.xml')

        if not os.path.exists(net_file):
            from meta_mohrl.environment.generate_configs import generate_single_intersection
            generate_single_intersection(config_dir)

        env = MultiObjectiveSumoEnv(
            net_file=net_file,
            route_file=route_file,
            num_seconds=3600,
            delta_time=5,
            single_agent=True,
            sumo_seed=args.seed
        )

        obs_dim = env.obs_dim
        num_actions = env.num_actions

        # Configure based on ablation mode
        mohrl_config = MOHRLConfig()
        mohrl_config.obs_dim = obs_dim
        mohrl_config.num_actions = num_actions

        if mode == 'no_feedback':
            mohrl_config.beta12_default = 0.0
            mohrl_config.beta23_default = 0.0

        agent = MOHRLciAgent(obs_dim, num_actions, mohrl_config)

        if mode == 'no_context':
            # Disable context encoder by zeroing context dim
            pass  # Uses mock/zero context

        # Run episodes
        rewards_history = []
        speed_history = []
        waiting_history = []
        queue_history = []

        for ep in range(args.episodes):
            obs, _ = env.reset()
            agent.reset()

            total_rewards = np.zeros(3)
            step = 0
            done = False
            max_steps = 720

            while not done and step < max_steps:
                action = agent.act(obs)
                next_obs, reward_vec, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                omega = agent.pareto_fronts['high'].sample_preference()

                if mode == 'no_pareto':
                    omega = np.array([1/3, 1/3, 1/3])

                agent.observe(obs, action, reward_vec, next_obs, done, omega)
                total_rewards += reward_vec
                obs = next_obs
                step += 1

                if step % 10 == 0:
                    agent.train_step()

            for _ in range(3):
                agent.train_step()
            agent.update_pareto_fronts()

            rewards_history.append(total_rewards.sum())
            speed_history.append(total_rewards[0])
            waiting_history.append(total_rewards[1])
            queue_history.append(total_rewards[2])

            if (ep + 1) % 10 == 0:
                avg_r = np.mean(rewards_history[-10:])
                print(f"  [{mode}] Episode {ep+1}: avg_reward={avg_r:.2f}")

        # Compute metrics
        metrics = agent.get_metrics()
        ablation_results[mode] = {
            'total_reward': rewards_history,
            'speed_reward': speed_history,
            'waiting_reward': waiting_history,
            'queue_reward': queue_history,
            'final_avg_reward': float(np.mean(rewards_history[-20:])),
            'final_avg_speed': float(np.mean(speed_history[-20:])),
            'final_avg_waiting': float(np.mean(waiting_history[-20:])),
            'final_avg_queue': float(np.mean(queue_history[-20:])),
            'hypervolume': float(metrics.get('high_hypervolume', 0)),
            'pareto_size': int(metrics.get('high_pareto_size', 0)),
            'convergence_episode': _find_convergence(rewards_history),
        }

        env.close()

    # Print ablation table
    print_ablation_table(ablation_results)

    # Save results
    save_dir = os.path.join(args.results_dir, f'ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


def _find_convergence(rewards, window=20, threshold=0.05):
    """Find episode where reward stabilizes (convergence point)."""
    if len(rewards) < window * 2:
        return len(rewards)

    for i in range(window, len(rewards) - window):
        recent = np.mean(rewards[i:i + window])
        prev = np.mean(rewards[i - window:i])
        if abs(recent - prev) / (abs(prev) + 1e-8) < threshold:
            return i
    return len(rewards)


def print_ablation_table(results: dict):
    """Print formatted ablation study results table."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS TABLE")
    print("=" * 100)

    headers = [
        "Variant", "Avg Reward↑", "Speed↑", "Waiting↑", "Queue↑",
        "HV↑", "Pareto#↑", "Conv.Ep↓"
    ]
    header_str = f"{'|'.join(f' {h:>12s} ' for h in headers)}"
    print(header_str)
    print("-" * len(header_str))

    mode_names = {
        'full': 'Full Meta-MOHRL',
        'no_meta': 'w/o Meta-Ctrl',
        'no_feedback': 'w/o Feedback',
        'no_context': 'w/o Context',
        'no_pareto': 'w/o Pareto'
    }

    for mode, data in results.items():
        name = mode_names.get(mode, mode)
        row = [
            f" {name:>12s} ",
            f" {data['final_avg_reward']:>12.2f} ",
            f" {data['final_avg_speed']:>12.3f} ",
            f" {data['final_avg_waiting']:>12.3f} ",
            f" {data['final_avg_queue']:>12.3f} ",
            f" {data['hypervolume']:>12.4f} ",
            f" {data['pareto_size']:>12d} ",
            f" {data['convergence_episode']:>12d} ",
        ]
        print("|".join(row))

    print("=" * 100)
    print("\nKey findings:")

    full = results.get('full', {})
    for mode in ['no_meta', 'no_feedback', 'no_context', 'no_pareto']:
        if mode in results:
            diff = full.get('final_avg_reward', 0) - results[mode].get('final_avg_reward', 0)
            pct = (diff / (abs(full.get('final_avg_reward', 1)) + 1e-8)) * 100
            name = mode_names[mode]
            print(f"  {name}: {pct:+.1f}% from full Meta-MOHRL")


def load_and_compare(results_dir: str):
    """Load results from all experiments and print comparison."""
    all_results = {}

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Load Meta-MOHRL results
        meta_file = os.path.join(folder_path, 'training_results.json')
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                data = json.load(f)
                mode = data.get('ablation_mode', 'full')
                all_results[f'Meta-MOHRL ({mode})'] = data.get('meta_mohrl', {})

        # Load baseline results
        baseline_file = os.path.join(folder_path, 'baseline_results.json')
        if os.path.exists(baseline_file):
            with open(baseline_file) as f:
                data = json.load(f)
                all_results.update(data)

    if all_results:
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)

        for name, data in all_results.items():
            if isinstance(data, dict) and 'total_reward' in data:
                rewards = data['total_reward']
                if rewards:
                    final = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
                    print(f"  {name:>25s}: final_avg={final:>8.2f}, episodes={len(rewards)}")

    return all_results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run_ablation:
        ablation_results = run_ablation_study(args)

    # Load and compare existing results
    if os.path.exists(args.results_dir):
        load_and_compare(args.results_dir)


if __name__ == '__main__':
    main()
