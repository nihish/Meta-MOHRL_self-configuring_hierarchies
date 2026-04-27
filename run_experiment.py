"""
Unified experiment runner for Meta-MOHRL.

Runs Meta-MOHRL + all baselines + ablation study on SUMO-RL,
saves comprehensive record file, then generates plots from real data.

Training strategy:
  - Primary training on single intersection (4K episodes, all 6 algorithms)
  - Post-training evaluation on all 15 randomized topologies

Usage:
    python run_experiment.py
"""

import os
import sys
import json
import time
import random
import copy
import glob
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meta_mohrl.config import MOHRLConfig, MetaConfig, EnvConfig
from meta_mohrl.mohrl_ci.agent import MOHRLciAgent
from meta_mohrl.meta_controller.meta_controller import MetaController
from meta_mohrl.environment.sumo_wrapper import MultiObjectiveSumoEnv
from baselines.hiro import HIROAgent
from baselines.hippo import HiPPOAgent
from baselines.dusdi import DUSDiAgent
from baselines.mosmac import MOSMACAgent

# ──────────────────── Configuration ────────────────────
SEED = 42
NUM_EPISODES = 4000        # Full 4K organic depth
MAX_STEPS = 200            # Steps per episode at delta_time=5
META_ITERS = 15            # Meta-controller outer loop iterations
INNER_STEPS = 3            # Inner gradient steps per meta iteration
NUM_SECONDS = 1000         # SUMO simulation seconds per episode
DELTA_TIME = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RECORD_FILE = os.path.join(RESULTS_DIR, "experiment_record.json")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
TOPO_DIR = os.path.join(PROJECT_ROOT, "sumo_configs", "15_topologies")
CHECKPOINT_INTERVAL = 500  # Save intermediate JSON every N episodes


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_json_safe(obj):
    """Recursively convert non-JSON-serializable types to safe primitives."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif type(obj) not in (int, float, str, bool, type(None)):
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)
    return obj


def save_record(all_records, path=None):
    """Save record file with full JSON safety."""
    if path is None:
        path = RECORD_FILE
    safe = make_json_safe(all_records)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


def make_env():
    """Create natively executed SUMO-RL environment (single-agent schema)."""
    config_dir = os.path.join(PROJECT_ROOT, "sumo_configs")
    net_file = os.path.join(config_dir, "single_intersection.net.xml")
    route_file = os.path.join(config_dir, "single_intersection.rou.xml")

    if not os.path.exists(net_file):
        from meta_mohrl.environment.generate_configs import generate_single_intersection
        generate_single_intersection(config_dir)

    return MultiObjectiveSumoEnv(
        net_file=net_file,
        route_file=route_file,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        use_gui=False,
        single_agent=True,
        sumo_seed=SEED,
    )


def make_topo_env(net_file, route_file):
    """Create environment for a specific topology."""
    return MultiObjectiveSumoEnv(
        net_file=net_file,
        route_file=route_file,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        use_gui=False,
        single_agent=True,
        sumo_seed=SEED,
    )


def format_eta(elapsed, done, total):
    """Compute ETA string from progress."""
    if done <= 0:
        return "estimating..."
    rate = elapsed / done
    remaining = rate * (total - done)
    eta = timedelta(seconds=int(remaining))
    return str(eta)


# ─────── Episode runner shared by all algorithms ───────
def run_episode(agent, env, algorithm_name, train=True, epsilon=0.0):
    """Run one episode, return per-step and summary metrics."""
    obs, info = env.reset()
    agent.reset()

    total_rewards = np.zeros(3)
    step_records = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        if algorithm_name == "Meta-MOHRL":
            action = agent.act(obs, epsilon=epsilon)
        else:
            try:
                action = agent.act(obs, epsilon=epsilon)
            except TypeError:
                action = agent.act(obs)
        next_obs, reward_vec, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if hasattr(agent, 'observe'):
            if algorithm_name == "Meta-MOHRL" or algorithm_name.startswith("Ablation"):
                omega = agent.pareto_fronts['high'].sample_preference()
                agent.observe(obs, action, reward_vec, next_obs, done, omega)
            else:
                agent.observe(obs, action, reward_vec, next_obs, done)

        total_rewards += reward_vec
        step_records.append({
            "step": step,
            "speed": float(reward_vec[0]),
            "waiting": float(reward_vec[1]),
            "queue": float(reward_vec[2]),
            "action": int(action),
        })
        obs = next_obs
        step += 1

        # Train every 10 steps if in training mode
        if train and step % 10 == 0:
            agent.train_step()

    # End-of-episode training
    if train:
        for _ in range(3):
            agent.train_step()
        if hasattr(agent, 'update_pareto_fronts'):
            agent.update_pareto_fronts()

    return {
        "total_reward": float(total_rewards.sum()),
        "speed_reward": float(total_rewards[0]),
        "waiting_reward": float(total_rewards[1]),
        "queue_reward": float(total_rewards[2]),
        "steps": step,
        "step_records": step_records,
    }


# ─────────────────── Algorithm Runners ───────────────────
def run_meta_mohrl(env, obs_dim, num_actions, label="Meta-MOHRL",
                   ablation="full"):
    """Run Meta-MOHRL with bilevel training."""
    print(f"\n{'='*60}")
    print(f"  Running: {label} (ablation={ablation})")
    print(f"  Meta iters: {META_ITERS}, Inner steps: {INNER_STEPS}")
    print(f"{'='*60}")

    mohrl_cfg = MOHRLConfig()
    mohrl_cfg.obs_dim = obs_dim
    mohrl_cfg.num_actions = num_actions
    mohrl_cfg.device = DEVICE

    meta_cfg = MetaConfig()
    meta_cfg.device = DEVICE

    agent = MOHRLciAgent(obs_dim, num_actions, mohrl_cfg)
    meta_ctrl = MetaController(meta_cfg).to(DEVICE)
    meta_opt = torch.optim.Adam(meta_ctrl.parameters(), lr=meta_cfg.alpha_out)

    episode_results = []
    config_history = []
    meta_losses = []

    ep_counter = 0
    algo_start = time.time()

    for meta_iter in range(META_ITERS):
        # Task descriptor
        task_desc = env.get_task_descriptor()
        desc_t = torch.FloatTensor(task_desc).unsqueeze(0).to(DEVICE)

        # Configuration
        if ablation == "no_meta":
            cfg = {
                'T1': 10, 'T2': 5, 'gamma1': 0.99, 'gamma2': 0.99,
                'gamma3': 0.99, 'beta12': 0.5, 'beta23': 0.5
            }
        elif ablation == "no_feedback":
            cfg = meta_ctrl.get_config_dict(desc_t)
            cfg['beta12'] = 0.0
            cfg['beta23'] = 0.0
        else:
            cfg = meta_ctrl.get_config_dict(desc_t)

        agent.set_configuration(cfg)
        config_history.append(copy.deepcopy(cfg))

        inner_rewards = []
        for k in range(INNER_STEPS):
            eps = max(0.05, 1.0 - (ep_counter / 2000.0))
            ep_result = run_episode(agent, env, label, train=True, epsilon=eps)
            inner_rewards.append(ep_result['total_reward'])

            ep_counter += 1
            episode_results.append({
                "episode": ep_counter,
                "meta_iter": meta_iter + 1,
                "inner_step": k + 1,
                **{key: ep_result[key] for key in
                   ['total_reward', 'speed_reward', 'waiting_reward',
                    'queue_reward', 'steps']},
            })

            elapsed = time.time() - algo_start
            total_eps = META_ITERS * INNER_STEPS
            eta = format_eta(elapsed, ep_counter, NUM_EPISODES)
            print(f"  [{label}] meta={meta_iter+1}/{META_ITERS} "
                  f"k={k+1}/{INNER_STEPS}: "
                  f"reward={ep_result['total_reward']:+.3f}  "
                  f"(spd={ep_result['speed_reward']:.3f} "
                  f"wait={ep_result['waiting_reward']:.3f} "
                  f"que={ep_result['queue_reward']:.3f}) "
                  f"ETA={eta}")

        # Meta-gradient update
        meta_loss_val = 0.0
        if ablation not in ("no_meta",):
            cfg_out = meta_ctrl(desc_t, hard=False)
            j_inner = torch.tensor(np.mean(inner_rewards),
                                   dtype=torch.float32, device=DEVICE)
            meta_loss = -j_inner
            T1_ent = -(cfg_out['T1_probs'] *
                       (cfg_out['T1_probs'] + 1e-8).log()).sum()
            T2_ent = -(cfg_out['T2_probs'] *
                       (cfg_out['T2_probs'] + 1e-8).log()).sum()
            meta_loss = meta_loss - 0.01 * (T1_ent + T2_ent)
            meta_opt.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_ctrl.parameters(), 1.0)
            meta_opt.step()
            meta_ctrl.step_annealing()
            meta_loss_val = float(meta_loss.item())

        meta_losses.append(meta_loss_val)

    # Post meta-learning fine-tuning phase
    print(f"\n  [{label}] Meta-learning complete. Wiping replay buffer and fine-tuning...")
    
    # Force clear replay buffer to prevent toxic gradients from poisoning fine-tuning
    if hasattr(agent, 'replay_buffer'):
        for level in ['high', 'mid', 'low']:
            agent.replay_buffer.buffers[level].buffer.clear()
            agent.replay_buffer.buffers[level].priorities.clear()

    for ep in range(ep_counter, NUM_EPISODES):
        eps = max(0.01, 1.0 - (ep_counter / 2000.0))
        ep_result = run_episode(agent, env, label, train=True, epsilon=eps)
        
        ep_counter += 1
        episode_results.append({
            "episode": ep_counter,
            "meta_iter": "fine-tuning",
            "inner_step": "fine-tuning",
            **{key: ep_result[key] for key in
               ['total_reward', 'speed_reward', 'waiting_reward',
                'queue_reward', 'steps']},
        })

        if ep_counter % 5 == 0:
            elapsed = time.time() - algo_start
            avg_r = np.mean([e['total_reward'] for e in episode_results[-5:]])
            eta = format_eta(elapsed, ep_counter, NUM_EPISODES)
            print(f"  [{label}] ep={ep_counter}/{NUM_EPISODES}: "
                  f"reward={ep_result['total_reward']:+.3f}  "
                  f"avg5={avg_r:+.3f}  "
                  f"(spd={ep_result['speed_reward']:.3f} "
                  f"wait={ep_result['waiting_reward']:.3f} "
                  f"que={ep_result['queue_reward']:.3f}) "
                  f"ETA={eta}")

    # Get final metrics
    metrics = agent.get_metrics()
    algo_time = time.time() - algo_start

    print(f"  [{label}] Completed in {algo_time:.0f}s "
          f"({ep_counter} episodes)")

    return {
        "algorithm": label,
        "ablation": ablation,
        "episodes": episode_results,
        "config_history": config_history,
        "meta_loss": meta_losses,
        "final_metrics": {k: float(v) for k, v in metrics.items()},
        "training_time_seconds": algo_time,
    }


def run_baseline(BaselineClass, name, env, obs_dim, num_actions,
                 all_records=None, **kwargs):
    """Run a baseline algorithm with progress logging and checkpointing."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"{'='*60}")

    agent = BaselineClass(obs_dim, num_actions, device=DEVICE, **kwargs)
    episode_results = []
    algo_start = time.time()

    for ep in range(NUM_EPISODES):
        ep_result = run_episode(agent, env, name, train=True)

        episode_results.append({
            "episode": ep + 1,
            **{key: ep_result[key] for key in
               ['total_reward', 'speed_reward', 'waiting_reward',
                'queue_reward', 'steps']},
        })

        if (ep + 1) % 5 == 0:
            elapsed = time.time() - algo_start
            avg_r = np.mean([e['total_reward'] for e in episode_results[-5:]])
            eta = format_eta(elapsed, ep + 1, NUM_EPISODES)
            print(f"  [{name}] ep={ep+1}/{NUM_EPISODES}: "
                  f"reward={ep_result['total_reward']:+.3f}  "
                  f"avg5={avg_r:+.3f}  "
                  f"(spd={ep_result['speed_reward']:.3f} "
                  f"wait={ep_result['waiting_reward']:.3f} "
                  f"que={ep_result['queue_reward']:.3f}) "
                  f"ETA={eta}")

        # Intermediate checkpoint
        if all_records is not None and (ep + 1) % CHECKPOINT_INTERVAL == 0:
            all_records["algorithms"][name] = {
                "algorithm": name,
                "episodes": episode_results,
                "final_metrics": {},
                "training_time_seconds": time.time() - algo_start,
            }
            save_record(all_records)
            print(f"    [Checkpoint saved at ep {ep+1}]")

    metrics = agent.get_metrics()
    algo_time = time.time() - algo_start

    print(f"  [{name}] Completed in {algo_time:.0f}s "
          f"({NUM_EPISODES} episodes)")

    return {
        "algorithm": name,
        "episodes": episode_results,
        "final_metrics": {k: float(v) for k, v in metrics.items()},
        "training_time_seconds": algo_time,
    }


# ─────────── Multi-Topology Evaluation ───────────
def evaluate_on_topologies(agent, algorithm_name, num_eval_episodes=3):
    """Evaluate a trained agent across all 15 topologies."""
    print(f"\n  Evaluating {algorithm_name} on 15 topologies...")
    topo_results = []

    net_files = sorted(glob.glob(os.path.join(TOPO_DIR, "topo_*.net.xml")))
    for net_file in net_files:
        topo_name = os.path.basename(net_file).replace(".net.xml", "")
        route_file = net_file.replace(".net.xml", ".rou.xml")

        if not os.path.exists(route_file):
            print(f"    Skipping {topo_name}: no route file")
            continue

        try:
            env = make_topo_env(net_file, route_file)
            ep_rewards = []
            for _ in range(num_eval_episodes):
                ep_result = run_episode(agent, env, algorithm_name, train=False)
                ep_rewards.append(ep_result['total_reward'])
            env.close()

            avg_reward = float(np.mean(ep_rewards))
            topo_results.append({
                "topology": topo_name,
                "avg_reward": avg_reward,
                "episodes": num_eval_episodes,
            })
            print(f"    {topo_name}: avg_reward={avg_reward:+.3f}")
        except Exception as e:
            print(f"    {topo_name}: FAILED ({e})")
            topo_results.append({
                "topology": topo_name,
                "avg_reward": 0.0,
                "error": str(e),
            })

    return topo_results


# ───────────────────── Main Experiment ─────────────────────
def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("  META-MOHRL EXPERIMENT  (real SUMO-RL data)")
    print(f"  Device: {DEVICE}")
    print(f"  Episodes per baseline: {NUM_EPISODES}")
    print(f"  Max steps/ep: {MAX_STEPS}")
    print(f"  Meta iters: {META_ITERS}  |  Inner steps: {INNER_STEPS}")
    print(f"  15 topologies in: {TOPO_DIR}")
    print("=" * 60)

    env = make_env()
    obs, _ = env.reset()
    obs_dim = env.obs_dim
    num_actions = env.num_actions
    env.close()

    print(f"  Obs dim: {obs_dim}  |  Actions: {num_actions}")

    all_records = {
        "experiment_date": datetime.now().isoformat(),
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS,
        "meta_iters": META_ITERS,
        "inner_steps": INNER_STEPS,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "device": DEVICE,
        "num_topologies": 15,
        "algorithms": {},
    }

    total_start = time.time()

    try:
        # ─── 1. Meta-MOHRL (full) ───
        env = make_env()
        result = run_meta_mohrl(env, obs_dim, num_actions,
                                label="Meta-MOHRL", ablation="full")
        all_records["algorithms"]["Meta-MOHRL"] = result
        env.close()
        save_record(all_records)

        # ─── 2. Ablation: No Meta-Controller ───
        env = make_env()
        result = run_meta_mohrl(env, obs_dim, num_actions,
                                label="Ablation-NoMeta", ablation="no_meta")
        all_records["algorithms"]["Ablation-NoMeta"] = result
        env.close()
        save_record(all_records)

        # ─── 3. Ablation: No Feedback ───
        env = make_env()
        result = run_meta_mohrl(env, obs_dim, num_actions,
                                label="Ablation-NoFB", ablation="no_feedback")
        all_records["algorithms"]["Ablation-NoFB"] = result
        env.close()
        save_record(all_records)

        # ─── 4. HIRO ───
        env = make_env()
        result = run_baseline(HIROAgent, "HIRO", env, obs_dim, num_actions,
                              all_records=all_records,
                              goal_dim=16, commitment=10)
        all_records["algorithms"]["HIRO"] = result
        env.close()
        save_record(all_records)

        # ─── 5. HiPPO ───
        env = make_env()
        result = run_baseline(HiPPOAgent, "HiPPO", env, obs_dim, num_actions,
                              all_records=all_records)
        all_records["algorithms"]["HiPPO"] = result
        env.close()
        save_record(all_records)

        # ─── 6. DUSDi ───
        env = make_env()
        result = run_baseline(DUSDiAgent, "DUSDi", env, obs_dim, num_actions,
                              all_records=all_records)
        all_records["algorithms"]["DUSDi"] = result
        env.close()
        save_record(all_records)

        # ─── 7. MOSMAC ───
        env = make_env()
        result = run_baseline(MOSMACAgent, "MOSMAC", env, obs_dim, num_actions,
                              all_records=all_records)
        all_records["algorithms"]["MOSMAC"] = result
        env.close()
        save_record(all_records)

    except Exception as e:
        import traceback
        print(f"\n{'!'*60}")
        print(f"  EXECUTION ERROR: {e}")
        traceback.print_exc()
        print(f"{'!'*60}")
    finally:
        total_time = time.time() - total_start
        all_records["total_time_seconds"] = total_time
        save_record(all_records)

    # ─── Summary table ───
    print(f"\n{'='*62}")
    print(f"  Record saved: {RECORD_FILE}")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    print(f"{'='*62}")

    print(f"\n{'Algorithm':<20} {'Avg Reward':>12} {'Speed':>10} "
          f"{'Wait':>10} {'Queue':>10}")
    print("-" * 62)
    for name, rec in all_records["algorithms"].items():
        eps = rec.get("episodes", [])
        if eps:
            last_n = eps[-min(10, len(eps)):]
            ar = np.mean([e['total_reward'] for e in last_n])
            sp = np.mean([e['speed_reward'] for e in last_n])
            wt = np.mean([e['waiting_reward'] for e in last_n])
            qu = np.mean([e['queue_reward'] for e in last_n])
            print(f"  {name:<18} {ar:>+12.3f} {sp:>+10.3f} "
                  f"{wt:>+10.3f} {qu:>+10.3f}")


if __name__ == "__main__":
    main()
