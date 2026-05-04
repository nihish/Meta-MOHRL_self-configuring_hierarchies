"""
Baseline training script for HIRO, HiPPO, and DUSDi.

Trains all three baselines on the same SUMO-RL environment
for fair comparison with Meta-MOHRL.

Usage:
    python train_baselines.py [--episodes 200] [--use-gui] [--algorithm all]
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meta_mohrl.config import EnvConfig
from meta_mohrl.environment.sumo_wrapper import MultiObjectiveSumoEnv
from baselines.hiro import HIROAgent
from baselines.hippo import HiPPOAgent
from baselines.dusdi import DUSDiAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=720)
    parser.add_argument('--use-gui', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['all', 'hiro', 'hippo', 'dusdi'])
    parser.add_argument('--save-dir', type=str, default='results')
    return parser.parse_args()


def create_env(args) -> MultiObjectiveSumoEnv:
    config_dir = os.path.join(PROJECT_ROOT, 'sumo_configs')
    os.makedirs(config_dir, exist_ok=True)

    net_file = os.path.join(config_dir, 'grid2x2.net.xml')
    route_file = os.path.join(config_dir, 'grid2x2.rou.xml')

    if not os.path.exists(net_file):
        from meta_mohrl.environment.generate_configs import generate_grid_2x2
        generate_grid_2x2(config_dir)

    return MultiObjectiveSumoEnv(
        net_file=net_file,
        route_file=route_file,
        num_seconds=3600,
        delta_time=5,
        use_gui=args.use_gui,
        single_agent=False,
        sumo_seed=args.seed,
    )


def train_baseline(agent, env, args, name: str) -> dict:
    """Train a single baseline agent."""
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")

    results = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for ep in range(args.episodes):
        start = time.time()
        obs_dict, _ = env.reset()
        agent.reset()

        total_rewards = np.zeros(3)
        step = 0
        done = False
        
        agents = list(obs_dict.keys()) if isinstance(obs_dict, dict) else [0]

        while not done and step < args.max_steps:
            action_dict = {}
            for a_id in agents:
                obs_a = obs_dict[a_id] if isinstance(obs_dict, dict) else obs_dict
                action_dict[a_id] = agent.act(obs_a)
                
            action_input = action_dict if isinstance(obs_dict, dict) else action_dict[0]
            next_obs_dict, reward_dict, terminated, truncated, info = env.step(action_input)
            
            if isinstance(terminated, dict):
                done = all(terminated.values()) or all(truncated.values())
            else:
                done = terminated or truncated

            step_reward_vec = np.zeros(3)
            for a_id in agents:
                n_obs = next_obs_dict[a_id] if isinstance(next_obs_dict, dict) else next_obs_dict
                r_vec = reward_dict[a_id] if isinstance(reward_dict, dict) else reward_dict
                act = action_dict[a_id]
                o = obs_dict[a_id] if isinstance(obs_dict, dict) else obs_dict

                agent.observe(o, act, r_vec, n_obs, done)
                step_reward_vec += r_vec

            total_rewards += step_reward_vec
            obs_dict = next_obs_dict
            step += 1

            if step % 10 == 0:
                agent.train_step()

        # End-of-episode training
        for _ in range(3):
            agent.train_step()

        ep_time = time.time() - start
        results['total_reward'].append(float(total_rewards.sum()))
        results['speed_reward'].append(float(total_rewards[0]))
        results['waiting_reward'].append(float(total_rewards[1]))
        results['queue_reward'].append(float(total_rewards[2]))
        results['steps'].append(step)
        results['time'].append(ep_time)

        metrics = agent.get_metrics()
        for k, v in metrics.items():
            results[k].append(float(v))

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(results['total_reward'][-10:])
            print(f"  Episode {ep+1}/{args.episodes}: "
                  f"reward={total_rewards.sum():.2f}, "
                  f"avg10={avg_r:.2f}, "
                  f"speed={total_rewards[0]:.3f}, "
                  f"wait={total_rewards[1]:.3f}, "
                  f"queue={total_rewards[2]:.3f}")

    return dict(results)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = create_env(args)
    obs_dim = env.obs_dim
    num_actions = env.num_actions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Environment: obs_dim={obs_dim}, actions={num_actions}")
    print(f"Device: {device}")

    all_results = {}
    algorithms = {}

    if args.algorithm in ['all', 'hiro']:
        algorithms['HIRO'] = HIROAgent(
            obs_dim, num_actions, goal_dim=16, commitment=10,
            lr=3e-4, gamma=0.99, device=device
        )

    if args.algorithm in ['all', 'hippo']:
        algorithms['HiPPO'] = HiPPOAgent(
            obs_dim, num_actions, num_options=8, num_sub_options=6,
            option_commitment=10, sub_commitment=5,
            lr=3e-4, gamma=0.99, device=device
        )

    if args.algorithm in ['all', 'dusdi']:
        algorithms['DUSDi'] = DUSDiAgent(
            obs_dim, num_actions, num_skills=8, num_sub_skills=6,
            skill_commitment=10, sub_commitment=5,
            lr=3e-4, gamma=0.99, device=device
        )

    for name, agent in algorithms.items():
        all_results[name] = train_baseline(agent, env, args, name)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'baselines_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    results_file = os.path.join(save_dir, 'baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll baseline results saved to {save_dir}")
    env.close()


if __name__ == '__main__':
    main()
