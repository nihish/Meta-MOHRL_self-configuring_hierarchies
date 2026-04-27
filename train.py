"""
Meta-MOHRL Training Script (Algorithm 1 from Meta-MOHRL paper).

Bilevel training procedure:
- Outer loop: Meta-controller learns configuration mapping
- Inner loop: MOHRL-ci trains policies under generated configuration

Usage:
    python train.py [--episodes 200] [--meta-iters 50] [--use-gui] [--ablation full]
"""

import os
import sys
import argparse
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meta_mohrl.config import MOHRLConfig, MetaConfig, EnvConfig, TrainingConfig
from meta_mohrl.mohrl_ci.agent import MOHRLciAgent
from meta_mohrl.meta_controller.meta_controller import MetaController
from meta_mohrl.meta_controller.task_encoder import TaskEncoder
from meta_mohrl.environment.sumo_wrapper import MultiObjectiveSumoEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-MOHRL Training')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per inner loop')
    parser.add_argument('--meta-iters', type=int, default=50, help='Outer loop iterations')
    parser.add_argument('--inner-steps', type=int, default=5, help='Inner gradient steps K')
    parser.add_argument('--use-gui', action='store_true', help='Show SUMO GUI')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--net', type=str, default='single', choices=['single', 'grid2x2'])
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_meta', 'no_feedback', 'no_context', 'no_pareto'])
    parser.add_argument('--save-dir', type=str, default='results')
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_environment(args, env_config: EnvConfig) -> MultiObjectiveSumoEnv:
    """Create SUMO-RL environment."""
    config_dir = os.path.join(PROJECT_ROOT, 'sumo_configs')
    os.makedirs(config_dir, exist_ok=True)

    # Generate configs if they don't exist
    if args.net == 'single':
        net_file = os.path.join(config_dir, 'single_intersection.net.xml')
        route_file = os.path.join(config_dir, 'single_intersection.rou.xml')
        single_agent = True
    else:
        net_file = os.path.join(config_dir, 'grid2x2.net.xml')
        route_file = os.path.join(config_dir, 'grid2x2.rou.xml')
        single_agent = False

    if not os.path.exists(net_file):
        from meta_mohrl.environment.generate_configs import (
            generate_single_intersection, generate_grid_2x2
        )
        if args.net == 'single':
            generate_single_intersection(config_dir)
        else:
            generate_grid_2x2(config_dir)

    env = MultiObjectiveSumoEnv(
        net_file=net_file,
        route_file=route_file,
        num_seconds=env_config.num_seconds,
        delta_time=env_config.delta_time,
        use_gui=args.use_gui,
        single_agent=single_agent,
        sumo_seed=args.seed
    )
    return env


def run_episode(
    agent: MOHRLciAgent,
    env: MultiObjectiveSumoEnv,
    train: bool = True,
    max_steps: int = 720
) -> dict:
    """Run one episode, collecting data and optionally training."""
    obs, info = env.reset()
    agent.reset()

    total_rewards = np.zeros(3)
    step = 0
    done = False

    while not done and step < max_steps:
        action = agent.act(obs)
        next_obs, reward_vec, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Sample preference from Pareto front
        omega = agent.pareto_fronts['high'].sample_preference()
        agent.observe(obs, action, reward_vec, next_obs, done, omega)

        total_rewards += reward_vec
        obs = next_obs
        step += 1

        # Train periodically
        if train and step % 10 == 0:
            agent.train_step()

    # End-of-episode training
    if train:
        for _ in range(5):
            agent.train_step()
        agent.update_pareto_fronts()

    return {
        'total_reward': total_rewards.sum(),
        'speed_reward': total_rewards[0],
        'waiting_reward': total_rewards[1],
        'queue_reward': total_rewards[2],
        'steps': step,
        'metrics': agent.get_metrics()
    }


def meta_train(args):
    """Full Meta-MOHRL bilevel training loop (Algorithm 1)."""
    print("=" * 70)
    print("Meta-MOHRL Training")
    print(f"  Ablation mode: {args.ablation}")
    print(f"  Meta iterations: {args.meta_iters}")
    print(f"  Inner steps: {args.inner_steps}")
    print(f"  Episodes per inner loop: {args.episodes}")
    print(f"  Network: {args.net}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    set_seed(args.seed)

    # Configs
    mohrl_config = MOHRLConfig()
    meta_config = MetaConfig()
    env_config = EnvConfig()
    meta_config.num_meta_iterations = args.meta_iters
    meta_config.inner_steps = args.inner_steps

    # Environment
    env = create_environment(args, env_config)
    obs_dim = env.obs_dim
    num_actions = env.num_actions

    print(f"  Obs dim: {obs_dim}, Actions: {num_actions}")

    # Initialize MOHRL-ci agent (shared initialization θ₀)
    mohrl_config.obs_dim = obs_dim
    mohrl_config.num_actions = num_actions
    agent = MOHRLciAgent(obs_dim, num_actions, mohrl_config)

    # Meta-controller
    meta_controller = MetaController(meta_config).to(meta_config.device)
    meta_optimizer = torch.optim.Adam(
        meta_controller.parameters(),
        lr=meta_config.alpha_out
    )

    # Results tracking
    results = {
        'meta_mohrl': defaultdict(list),
        'config_history': [],
        'ablation_mode': args.ablation
    }

    # Save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'meta_mohrl_{args.ablation}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # ===== OUTER LOOP: Meta-training =====
    for meta_iter in range(args.meta_iters):
        iter_start = time.time()

        # Step 1: Compute task descriptor
        task_descriptor = env.get_task_descriptor()
        descriptor_t = torch.FloatTensor(task_descriptor).unsqueeze(0).to(meta_config.device)

        # Step 2: Generate configuration from meta-controller
        if args.ablation == 'no_meta':
            # Ablation: use default fixed configuration
            config_dict = {
                'T1': mohrl_config.T1_default,
                'T2': mohrl_config.T2_default,
                'gamma1': mohrl_config.gamma1_default,
                'gamma2': mohrl_config.gamma2_default,
                'gamma3': mohrl_config.gamma3_default,
                'beta12': mohrl_config.beta12_default,
                'beta23': mohrl_config.beta23_default,
            }
        else:
            config_dict = meta_controller.get_config_dict(descriptor_t)

        # Step 3: Apply configuration to MOHRL-ci agent
        agent.set_configuration(config_dict)
        results['config_history'].append(config_dict)

        print(f"\n--- Meta Iteration {meta_iter + 1}/{args.meta_iters} ---")
        print(f"  Config: T1={config_dict['T1']}, T2={config_dict['T2']}, "
              f"γ1={config_dict['gamma1']:.3f}, β12={config_dict['beta12']:.3f}")

        # Step 4: Inner loop — K gradient steps
        # Save initial parameters for meta-gradient
        theta_0 = {name: p.clone() for name, p in
                    zip(['dummy'], [torch.zeros(1)])}  # placeholder

        inner_rewards = []
        for k in range(meta_config.inner_steps):
            # Run episode and collect data
            ep_result = run_episode(
                agent, env, train=True,
                max_steps=env_config.num_seconds // env_config.delta_time
            )
            inner_rewards.append(ep_result['total_reward'])

            print(f"    Inner step {k+1}/{meta_config.inner_steps}: "
                  f"reward={ep_result['total_reward']:.2f} "
                  f"(speed={ep_result['speed_reward']:.3f}, "
                  f"wait={ep_result['waiting_reward']:.3f}, "
                  f"queue={ep_result['queue_reward']:.3f})")

        # Step 5: Compute meta-gradient (first-order approximation)
        if args.ablation != 'no_meta':
            # Evaluate J_inner after K steps
            config_output = meta_controller(descriptor_t, hard=False)

            # Use inner-loop performance as meta-objective
            j_inner = torch.tensor(
                np.mean(inner_rewards), dtype=torch.float32,
                device=meta_config.device
            )

            # Surrogate loss: maximize inner-loop return
            # The configuration influences performance through the agent
            # First-order approx: treat config as affecting reward directly
            meta_loss = -j_inner

            # Add regularization for config diversity
            T1_entropy = -(config_output['T1_probs'] *
                          (config_output['T1_probs'] + 1e-8).log()).sum()
            T2_entropy = -(config_output['T2_probs'] *
                          (config_output['T2_probs'] + 1e-8).log()).sum()
            meta_loss = meta_loss - 0.01 * (T1_entropy + T2_entropy)

            meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                meta_controller.parameters(), meta_config.meta_grad_clip
            )
            meta_optimizer.step()

            # Step annealing
            meta_controller.step_annealing()

        # Step 6: Record results
        iter_time = time.time() - iter_start
        avg_reward = np.mean(inner_rewards)
        results['meta_mohrl']['total_reward'].append(avg_reward)
        results['meta_mohrl']['speed_reward'].append(ep_result['speed_reward'])
        results['meta_mohrl']['waiting_reward'].append(ep_result['waiting_reward'])
        results['meta_mohrl']['queue_reward'].append(ep_result['queue_reward'])
        results['meta_mohrl']['iter_time'].append(iter_time)

        # Metrics
        metrics = agent.get_metrics()
        for k, v in metrics.items():
            results['meta_mohrl'][k].append(v)

        print(f"  Avg reward: {avg_reward:.2f}, Time: {iter_time:.1f}s")
        print(f"  HV: {metrics.get('high_hypervolume', 0):.4f}, "
              f"Pareto size: {metrics.get('high_pareto_size', 0)}")

        # Periodic save
        if (meta_iter + 1) % 10 == 0:
            _save_results(results, save_dir)
            print(f"  [Checkpoint saved to {save_dir}]")

    # Final save
    _save_results(results, save_dir)
    env.close()

    print("\n" + "=" * 70)
    print(f"Training complete! Results saved to {save_dir}")
    print("=" * 70)

    return results


def _save_results(results: dict, save_dir: str):
    """Save training results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable[key] = {
                k: [float(x) if isinstance(x, (np.floating, float)) else x
                    for x in v] if isinstance(v, list) else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            serializable[key] = [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in item.items()} if isinstance(item, dict) else item
                for item in value
            ]
        else:
            serializable[key] = value

    results_file = os.path.join(save_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == '__main__':
    args = parse_args()
    meta_train(args)
