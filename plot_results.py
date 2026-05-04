"""
Publication-quality visualization suite.

Renders 9 comprehensive plots from the training JSON records:
  1. Convergence comparison (all 7 algorithms, multi-seed shaded)
  2. Ablation bar charts (Meta-MOHRL vs 2 ablations per objective)
  3. Radar chart (all algorithms across speed/wait/queue)
  4. Meta-learning adaptation (T1/T2 + beta + meta-loss)
  5. Hypervolume optimization curves
  6. Config gamma adaptation over meta iterations
  7. Reward distributions (violin/box)
  8. 3D Pareto front
  9. Multi-Agent Decentralization (Independent rewards)

Usage:
    python plot_results.py
"""

import os, sys, json, argparse, glob

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ───────── Color Palette ─────────
PALETTE = {
    "Meta-MOHRL":      "#1565C0",
    "Ablation-NoMeta": "#FF8F00",
    "Ablation-NoFB":   "#7B1FA2",
    "HIRO":            "#E65100",
    "HiPPO":           "#2E7D32",
    "DUSDi":           "#C62828",
    "MOSMAC":          "#00BCD4",
}

ALL_ALGOS = ["Meta-MOHRL", "Ablation-NoMeta", "Ablation-NoFB",
             "HIRO", "HiPPO", "DUSDi", "MOSMAC"]

def _color(name):
    return PALETTE.get(name, "#555555")

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def smooth(y, w=10):
    if len(y) < w:
        return y
    out = np.convolve(y, np.ones(w) / w, mode="valid")
    pad = [out[0]] * (len(y) - len(out))
    return np.concatenate([pad, out])

def load_all_records(results_dir):
    files = glob.glob(os.path.join(results_dir, "experiment_record_seed*.json"))
    records = []
    for f in files:
        with open(f, 'r') as file:
            records.append(json.load(file))
    return records

def get_episodes_list(records, name):
    all_eps = []
    for r in records:
        try:
            all_eps.append(r["algorithms"][name]["episodes"])
        except (KeyError, TypeError):
            pass
    return all_eps

# ────────── Plot 1: Overall Reward Convergence (Shaded) ──────────
def plot_convergence(records, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Version 3: Training Convergence (5 Seeds)",
                 fontsize=16, fontweight="bold")

    metrics = [
        ("Total Reward", "total_reward"),
        ("Speed Reward", "speed_reward"),
        ("Waiting Time Reward", "waiting_reward"),
    ]

    for i, (ax, (title, key)) in enumerate(zip(axes, metrics)):
        for name in ALL_ALGOS:
            all_eps = get_episodes_list(records, name)
            if not all_eps:
                continue
            
            # Aggregate across seeds
            curves = []
            for eps in all_eps:
                vals = [e[key] for e in eps]
                sm = smooth(vals, w=20)
                curves.append(sm)
                
            curves = np.array(curves)
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            x = range(1, len(mean_curve) + 1)
            
            ax.plot(x, mean_curve, label=name, color=_color(name), linewidth=2)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                            color=_color(name), alpha=0.15, edgecolor='none')

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        
        if i == 1:
            ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)

    path = os.path.join(out_dir, "version_3_reward_overall_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 2: Ablation Study ──────────
def plot_ablation(records, out_dir):
    ablation_algos = ["Meta-MOHRL", "Ablation-NoMeta", "Ablation-NoFB"]
    objectives = ["speed_reward", "waiting_reward", "queue_reward"]
    obj_labels = ["Speed ↑", "Waiting Time ↓", "Queue Length ↓"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Version 3: Ablation Study — Component Contribution",
                 fontsize=16, fontweight="bold")

    x = np.arange(len(objectives))
    width = 0.25

    for i, name in enumerate(ablation_algos):
        all_eps = get_episodes_list(records, name)
        if not all_eps:
            continue
            
        seed_means = []
        for eps in all_eps:
            last_n = eps[-50:]
            seed_means.append([np.mean([e[obj] for e in last_n]) for obj in objectives])
            
        seed_means = np.array(seed_means)
        means = np.mean(seed_means, axis=0)
        stds = np.std(seed_means, axis=0)
        
        ax.bar(x + i * width, means, width, label=name,
               color=_color(name), alpha=0.85, yerr=stds,
               capsize=4, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(obj_labels)
    ax.set_ylabel("Mean Reward (last 50 episodes)")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    path = os.path.join(out_dir, "version_3_ablation_study.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 3: Radar Chart ──────────
def plot_radar(records, out_dir):
    categories = ["Speed", "Wait Reduction", "Queue Reduction",
                   "Convergence Rate", "Stability"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Version 3: Multi-Objective Algorithm Comparison",
                 fontsize=16, fontweight="bold", y=1.05)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Pre-calculate mins and maxes for normalization
    raw_data = {}
    for name in ALL_ALGOS:
        all_eps = get_episodes_list(records, name)
        if not all_eps: continue
        eps = all_eps[0] # Just use first seed for radar to avoid over-averaging
        speed = np.array([e["speed_reward"] for e in eps])
        wait = np.array([e["waiting_reward"] for e in eps])
        queue = np.array([e["queue_reward"] for e in eps])
        
        last_n = max(1, len(speed) // 5)
        s = np.mean(speed[-last_n:])
        w = np.mean(wait[-last_n:])
        q = np.mean(queue[-last_n:])
        conv = np.mean(speed[-last_n:]) - np.mean(speed[:last_n])
        stab = -np.std(speed[-last_n:])
        
        raw_data[name] = [s, w, q, conv, stab]
        
    mins = np.min(list(raw_data.values()), axis=0)
    maxs = np.max(list(raw_data.values()), axis=0)
    ranges = np.where(maxs - mins == 0, 1e-6, maxs - mins)

    for name, vals in raw_data.items():
        norm_vals = (np.array(vals) - mins) / ranges
        norm_vals = norm_vals.tolist()
        norm_vals += norm_vals[:1]
        
        ax.plot(angles, norm_vals, 'o-', linewidth=2, label=name,
                color=_color(name), markersize=4)
        ax.fill(angles, norm_vals, alpha=0.1, color=_color(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    path = os.path.join(out_dir, "version_3_radar_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 4: Meta-Learning Adaptation ──────────
def plot_meta_adaptation(records, out_dir):
    meta_data = records[0].get("algorithms", {}).get("Meta-MOHRL", {})
    cfg_hist = meta_data.get("config_history", [])
    meta_losses = meta_data.get("meta_loss", [])

    if not cfg_hist:
        return

    iters = list(range(1, len(cfg_hist) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Version 3: Meta-Controller Configuration Adaptation",
                 fontsize=16, fontweight="bold")

    ax = axes[0]
    t1_vals = [c.get("T1", 10) for c in cfg_hist]
    t2_vals = [c.get("T2", 5) for c in cfg_hist]
    ax.plot(iters, t1_vals, "o-", label="High-level $T^1$", linewidth=2.5, color="#1565C0")
    ax.plot(iters, t2_vals, "s-", label="Mid-level $T^2$", linewidth=2.5, color="#00897B")
    ax.set_title("Temporal Commitment Horizons")
    ax.set_xlabel("Meta Iteration")
    ax.legend()

    ax = axes[1]
    b12 = [c.get("beta12", 0.5) for c in cfg_hist]
    b23 = [c.get("beta23", 0.5) for c in cfg_hist]
    ax.plot(iters, b12, "^-", label=r"$\beta^{12}$", linewidth=2.5, color="#C62828")
    ax.plot(iters, b23, "v-", label=r"$\beta^{23}$", linewidth=2.5, color="#7B1FA2")
    ax.set_title("Bidirectional Feedback Gains")
    ax.set_xlabel("Meta Iteration")
    ax.legend()

    ax = axes[2]
    if meta_losses:
        ax.plot(iters[:len(meta_losses)], meta_losses, "D-", color="#424242", linewidth=2.5)
        ax.fill_between(iters[:len(meta_losses)], 0, meta_losses, color="#424242", alpha=0.1)
    ax.set_title("Meta-Objective Loss")
    ax.set_xlabel("Meta Iteration")

    path = os.path.join(out_dir, "version_3_meta_learning_adaptation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 5: Hypervolume Optimization ──────────
def plot_hypervolume(records, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Version 3: Hypervolume Indicator Over Training",
                 fontsize=16, fontweight="bold")

    for name in ALL_ALGOS:
        all_eps = get_episodes_list(records, name)
        if not all_eps: continue
        eps = all_eps[0]
        
        speed = np.array([e["speed_reward"] for e in eps])
        wait = np.array([e["waiting_reward"] for e in eps])
        queue = np.array([e["queue_reward"] for e in eps])
        
        ref = np.array([-100.0, -100.0, -100.0])
        window = max(5, len(speed) // 50)
        hvs = []
        for i in range(0, len(speed), window):
            chunk_end = min(i + window, len(speed))
            s_avg = np.mean(speed[i:chunk_end])
            w_avg = np.mean(wait[i:chunk_end])
            q_avg = np.mean(queue[i:chunk_end])
            hv = max(0, s_avg - ref[0]) * max(0, w_avg - ref[1]) * max(0, q_avg - ref[2])
            hvs.append(hv)

        step_hvs = np.maximum.accumulate(hvs)
        x = np.linspace(1, len(speed), len(step_hvs))
        ax.plot(x, step_hvs, label=name, color=_color(name), linewidth=2, drawstyle='steps-post')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Hypervolume Indicator")
    ax.legend(loc='lower right')

    path = os.path.join(out_dir, "version_3_hypervolume_optimization.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 6: Gamma Adaptation ──────────
def plot_gamma_adaptation(records, out_dir):
    meta_data = records[0].get("algorithms", {}).get("Meta-MOHRL", {})
    cfg_hist = meta_data.get("config_history", [])

    if not cfg_hist: return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Version 3: Discount Factor Adaptation Over Meta-Iterations",
                 fontsize=16, fontweight="bold")

    iters = list(range(1, len(cfg_hist) + 1))
    ax.plot(iters, [c.get("gamma1", 0.99) for c in cfg_hist], "o-", label=r"$\gamma^1$", linewidth=2.5, color="#1565C0")
    ax.plot(iters, [c.get("gamma2", 0.99) for c in cfg_hist], "s-", label=r"$\gamma^2$", linewidth=2.5, color="#FF8F00")
    ax.plot(iters, [c.get("gamma3", 0.99) for c in cfg_hist], "^-", label=r"$\gamma^3$", linewidth=2.5, color="#2E7D32")

    ax.set_xlabel("Meta Iteration")
    ax.set_ylabel("Discount Factor $\\gamma$")
    ax.legend()

    path = os.path.join(out_dir, "version_3_config_gamma_adaptation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 7: Reward Distributions ──────────
def plot_reward_distributions(records, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Version 3: Reward Distribution Analysis",
                 fontsize=16, fontweight="bold")

    objectives = [
        ("Speed Reward", "speed_reward"),
        ("Waiting Time Reward", "waiting_reward"),
        ("Queue Length Reward", "queue_reward"),
    ]

    for ax, (title, key) in zip(axes, objectives):
        data, labels, colors = [], [], []
        for name in ALL_ALGOS:
            all_eps = get_episodes_list(records, name)
            if not all_eps: continue
            

            pooled = []
            for eps in all_eps:
                n_last = max(1, len(eps) // 5)
                pooled.extend([e[key] for e in eps[-n_last:]])
            
            data.append(pooled)
            labels.append(name)
            colors.append(_color(name))

        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)

    path = os.path.join(out_dir, "version_3_reward_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 8: 3D Pareto Front ──────────
def plot_pareto_3d(records, out_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Version 3: 3D Pareto Front in Objective Space",
                 fontsize=16, fontweight="bold")

    for name in ALL_ALGOS:
        all_eps = get_episodes_list(records, name)
        if not all_eps: continue
        eps = all_eps[0] # Plot for one representative seed
        
        indices = np.linspace(0, len(eps) - 1, min(500, len(eps)), dtype=int)
        speeds = [eps[i]["speed_reward"] for i in indices]
        waits = [eps[i]["waiting_reward"] for i in indices]
        queues = [eps[i]["queue_reward"] for i in indices]

        ax.scatter(speeds, waits, queues, c=_color(name), label=name,
                  alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel("Speed Reward ↑")
    ax.set_ylabel("Waiting Time ↓")
    ax.set_zlabel("Queue Length ↓")
    
    # Zoom in to the dense optimal frontier cluster
    ax.set_xlim(left=-20)
    ax.set_ylim(top=0)
    ax.set_zlim(top=0)
    
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=25, azim=135)

    path = os.path.join(out_dir, "version_3_pareto_front_3d.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

# ────────── Plot 9: Multi-Agent Decentralization ──────────
def plot_multi_agent_decentralization(records, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Version 3: Cooperative Multi-Agent Decentralization (Meta-MOHRL)",
                 fontsize=16, fontweight="bold")

    all_eps = get_episodes_list(records, "Meta-MOHRL")
    if not all_eps: return
    eps = all_eps[0]
    
    agent_curves = {"A0": [], "A1": [], "B0": [], "B1": []}
    
    for e in eps:
        if "agent_rewards" in e:
            for k in agent_curves.keys():
                agent_curves[k].append(e["agent_rewards"].get(k, 0))
    
    if not agent_curves["A0"]:
        return # Missing data
        
    x = range(1, len(agent_curves["A0"]) + 1)
    
    labels = {
        "A0": "Core Intersection A0 (Sacrificing Queue)",
        "A1": "Core Intersection A1 (Sacrificing Queue)",
        "B0": "Edge Intersection B0 (Maximizing Throughput)",
        "B1": "Edge Intersection B1 (Maximizing Throughput)"
    }
    colors = {"A0": "#1f77b4", "A1": "#ff7f0e", "B0": "#2ca02c", "B1": "#d62728"}
    
    for agent, curve in agent_curves.items():
        sm = smooth(curve, w=30)
        ax.plot(x, sm, label=labels[agent], color=colors[agent], linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Agent Independent Total Reward")
    ax.legend(loc="lower right")
    
    path = os.path.join(out_dir, "version_3_multi_agent_decentralization.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    records = load_all_records(args.results_dir)
    if not records:
        print("ERROR: No records found!")
        sys.exit(1)

    print(f"\nGenerating Version 3 plots (Multi-Seed N={len(records)}) → {args.output_dir}")

    plot_convergence(records, args.output_dir)
    plot_ablation(records, args.output_dir)
    plot_radar(records, args.output_dir)
    plot_meta_adaptation(records, args.output_dir)
    plot_hypervolume(records, args.output_dir)
    plot_gamma_adaptation(records, args.output_dir)
    plot_reward_distributions(records, args.output_dir)
    plot_pareto_3d(records, args.output_dir)
    plot_multi_agent_decentralization(records, args.output_dir)

    print(f"\n  ✓ All 9 plots rendered successfully.")

if __name__ == "__main__":
    main()
