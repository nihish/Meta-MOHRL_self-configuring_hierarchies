"""
Publication-quality visualization for Meta-MOHRL-v2
---------------------------------------------------
Produces:
  1.  meta_mohrl_5seeds_convergence.png  — 5-seed training convergence
  2.  meta_mohrl_pareto_improved.png     — improved 3D Pareto front

Both figures are saved to  Meta-MOHRL-v2/figures/
"""

import os, json, sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(ROOT, "Meta-MOHRL-v2", "results")
OUT_DIR  = os.path.join(ROOT, "Meta-MOHRL-v2", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

# ── Algo palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "Meta-MOHRL":      "#1565C0",
    "Ablation-NoMeta": "#EF6C00",
    "Ablation-NoFB":   "#7B1FA2",
    "HIRO":            "#BF360C",
    "HiPPO":           "#2E7D32",
    "DUSDi":           "#C62828",
    "MOSMAC":          "#0097A7",
}
ALL_ALGOS = list(PALETTE.keys())

# ── Distinct, high-contrast colours for 5 seeds ─────────────────────────────
SEED_COLORS = ["#E53935", "#43A047", "#8E24AA", "#FB8C00", "#00ACC1"]
SEED_MARKERS = ["o", "s", "^", "D", "v"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ema(y, span=60):
    """Exponential moving average."""
    a = 2.0 / (span + 1)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    return out


def setup_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Serif",
        "font.size":         14,
        "axes.labelsize":    16,
        "axes.titlesize":    18,
        "axes.titleweight":  "bold",
        "axes.labelweight":  "bold",
        "xtick.labelsize":   13,
        "ytick.labelsize":   13,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.grid":         True,
        "grid.alpha":        0.20,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#cccccc",
        "legend.fontsize":   12,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_seed_data():
    """Return  metric -> list[np.ndarray]  (one per seed, Meta-MOHRL only)."""
    metrics = ["total_reward", "speed_reward", "waiting_reward", "queue_reward"]
    data = {m: [] for m in metrics}

    for seed in SEEDS:
        fp = os.path.join(SEED_DIR, f"experiment_record_seed{seed}.json")
        if not os.path.exists(fp):
            print(f"  [WARN] Missing: {fp}")
            continue
        with open(fp) as f:
            rec = json.load(f)
        eps = rec["algorithms"]["Meta-MOHRL"]["episodes"]
        for m in metrics:
            data[m].append(np.array([e[m] for e in eps], dtype=float))
        print(f"  Loaded seed {seed}  ({len(eps)} episodes)")

    return data


def load_all_algos(seed=42):
    """Return  algo -> {speed, wait, queue} arrays  (full training)."""
    fp = os.path.join(SEED_DIR, f"experiment_record_seed{seed}.json")
    with open(fp) as f:
        rec = json.load(f)
    result = {}
    for name in ALL_ALGOS:
        eps = rec["algorithms"].get(name, {}).get("episodes", [])
        if not eps:
            continue
        result[name] = {
            "speed": np.array([e["speed_reward"]  for e in eps]),
            "wait":  np.array([e["waiting_reward"] for e in eps]),
            "queue": np.array([e["queue_reward"]   for e in eps]),
        }
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — 5-seed convergence  (fixed colours, legend placement, scaling)
# ═════════════════════════════════════════════════════════════════════════════
def plot_5seeds_convergence(seed_data):
    print("\n[1] Generating 5-seed convergence plot ...")

    panels = [
        ("Total Reward",        "total_reward"),
        ("Speed Reward",        "speed_reward"),
        ("Waiting-Time Reward", "waiting_reward"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("white")

    for ax, (title, key) in zip(axes, panels):
        arrays = seed_data.get(key, [])
        if not arrays:
            ax.set_title(title)
            continue

        min_len = min(len(a) for a in arrays)
        x = np.arange(1, min_len + 1)

        smoothed = [ema(a[:min_len], span=80) for a in arrays]

        # ── Per-seed lines — distinct colours + markers ──────────────────
        for i, sm in enumerate(smoothed):
            # Thin markers every N episodes for identification
            marker_every = max(1, min_len // 12)
            ax.plot(x, sm,
                    color=SEED_COLORS[i],
                    linewidth=1.5,
                    alpha=0.75,
                    marker=SEED_MARKERS[i],
                    markersize=5.5,
                    markevery=marker_every,
                    markeredgewidth=0.6,
                    markeredgecolor="white")

        # ── Mean + std band ──────────────────────────────────────────────
        mat  = np.vstack(smoothed)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        ax.fill_between(x, mean - std, mean + std,
                        color="#263238", alpha=0.10)
        ax.plot(x, mean,
                color="#0D47A1", linewidth=3.0,
                path_effects=[pe.withStroke(linewidth=5, foreground="white"),
                              pe.Normal()])

        # ── Y-axis scaling: zoom into the converged region ───────────────
        # Use the 5th and 95th percentile of raw values to set the window
        all_vals = np.concatenate([a[:min_len] for a in arrays])
        y_lo = np.percentile(all_vals, 2)
        y_hi = np.percentile(all_vals, 98)
        margin = (y_hi - y_lo) * 0.12
        ax.set_ylim(y_lo - margin, y_hi + margin)

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=16)
        ax.set_ylabel("Reward", fontsize=16)
        ax.set_xlim(1, min_len)
        ax.set_facecolor("white")

    # ── Single shared legend below figure ──────────────────────────────────
    seed_handles = [
        Line2D([0], [0],
               color=SEED_COLORS[i], linewidth=2.5, alpha=0.85,
               marker=SEED_MARKERS[i], markersize=8,
               markeredgecolor="white", markeredgewidth=0.6,
               label=f"Seed {SEEDS[i]}")
        for i in range(len(SEEDS))
    ]
    mean_handle = Line2D([0], [0],
                         color="#0D47A1", linewidth=3.5,
                         label="Mean (5 seeds)")
    band_handle = plt.Rectangle((0, 0), 1, 1,
                                fc="#263238", alpha=0.15,
                                label="\u00b11 Std Dev")

    fig.suptitle(
        "Meta-MOHRL \u2014 Training Convergence Across 5 Independent Seeds",
        fontsize=20, fontweight="bold", y=1.02)

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    fig.legend(handles=seed_handles + [mean_handle, band_handle],
               loc="lower center",
               ncol=7,
               fontsize=13,
               framealpha=0.95,
               bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2,
               handletextpad=0.5,
               handlelength=2.5)

    out = os.path.join(OUT_DIR, "meta_mohrl_5seeds_convergence.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — 3D Pareto front (like version 3 but decluttered)
# ═════════════════════════════════════════════════════════════════════════════
def plot_pareto_3d(algo_data):
    """
    3D Pareto front in (Speed, Wait, Queue) objective space.

    Key design choices:
      - Last 25% episodes only (convergence region)
      - Sparse sub-sampled scatter (low alpha) + bold centroids
      - Floor-plane drop projections to show X-Y separation
      - Camera at elev=30, azim=225 to maximise visual cluster separation
      - Algorithm names annotated directly on the centroid
    """
    print("\n[2] Generating improved 3D Pareto front ...")

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    algo_markers = {
        "Meta-MOHRL": "o", "Ablation-NoMeta": "^", "Ablation-NoFB": "s",
        "HIRO": "D", "HiPPO": "p", "DUSDi": "h", "MOSMAC": "v",
    }

    # ── Pre-compute global bounds for floor projections ──────────────────
    all_sp = np.concatenate([algo_data[n]["speed"][int(len(algo_data[n]["speed"])*0.75):]
                             for n in ALL_ALGOS if n in algo_data])
    all_wt = np.concatenate([algo_data[n]["wait"][int(len(algo_data[n]["wait"])*0.75):]
                             for n in ALL_ALGOS if n in algo_data])
    all_qu = np.concatenate([algo_data[n]["queue"][int(len(algo_data[n]["queue"])*0.75):]
                             for n in ALL_ALGOS if n in algo_data])

    pad_sp = (all_sp.max() - all_sp.min()) * 0.12
    pad_wt = (all_wt.max() - all_wt.min()) * 0.12
    pad_qu = (all_qu.max() - all_qu.min()) * 0.15

    z_floor = all_qu.min() - pad_qu  # floor for drop projections

    centroids = {}

    for name in ALL_ALGOS:
        d = algo_data.get(name)
        if d is None:
            continue
        color = PALETTE[name]

        n = len(d["speed"])
        start = int(n * 0.75)
        sp = d["speed"][start:]
        wt = d["wait"][start:]
        qu = d["queue"][start:]

        # Sub-sample to ~100 points
        stride = max(1, len(sp) // 100)
        sp_s, wt_s, qu_s = sp[::stride], wt[::stride], qu[::stride]

        # ── Scatter cloud (low opacity) ──────────────────────────────────
        ax.scatter(sp_s, wt_s, qu_s,
                   c=color, alpha=0.22, s=12,
                   marker=algo_markers.get(name, "o"),
                   edgecolors="none", depthshade=True)

        # ── Bold centroid ────────────────────────────────────────────────
        cx, cy, cz = sp.mean(), wt.mean(), qu.mean()
        centroids[name] = (cx, cy, cz)

        ax.scatter([cx], [cy], [cz],
                   c=color, s=280,
                   marker=algo_markers.get(name, "o"),
                   edgecolors="white", linewidths=2.5,
                   zorder=10, depthshade=False,
                   label=name)

        # ── Floor projection (drop shadow on Z-floor) ───────────────────
        ax.scatter([cx], [cy], [z_floor],
                   c=color, alpha=0.30, s=100,
                   marker=algo_markers.get(name, "o"),
                   edgecolors="none", depthshade=False)
        # Vertical dashed drop line from centroid to floor
        ax.plot([cx, cx], [cy, cy], [cz, z_floor],
                color=color, alpha=0.25, linewidth=1.0, linestyle=":")

        # ── Convex hull envelope ─────────────────────────────────────────
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([sp_s, wt_s, qu_s])
            if len(pts) >= 6:
                hull = ConvexHull(pts)
                faces = [pts[simplex] for simplex in hull.simplices]
                mesh = Poly3DCollection(faces, alpha=0.04,
                                        facecolor=color,
                                        edgecolor=color,
                                        linewidth=0.2)
                ax.add_collection3d(mesh)
        except Exception:
            pass

    # ── Centroid annotations with leader lines ───────────────────────────
    # Stagger text offsets so labels don't overlap
    # Sort centroids by speed (X) to assign offsets systematically
    sorted_names = sorted(centroids.keys(), key=lambda n: centroids[n][0])
    offsets_z = [18, -22, 15, -18, 12, -15, 10]  # alternating up/down
    for idx, nm in enumerate(sorted_names):
        cx, cy, cz = centroids[nm]
        z_off = offsets_z[idx % len(offsets_z)]
        label_z = cz + z_off

        # Leader line
        ax.plot([cx, cx], [cy, cy], [cz, label_z],
                color=PALETTE[nm], alpha=0.6, linewidth=1.2)

        ax.text(cx, cy, label_z,
                nm, fontsize=12, fontweight="bold",
                color=PALETTE[nm], ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=4,
                                           foreground="white")])

    # ── Connect centroids with a Pareto front polyline ───────────────────
    # Sort by speed to trace the Pareto front
    front_names = sorted(centroids.keys(), key=lambda n: centroids[n][0])
    front_pts = np.array([centroids[n] for n in front_names])
    ax.plot(front_pts[:, 0], front_pts[:, 1], front_pts[:, 2],
            color="#37474F", linewidth=1.8, linestyle="--",
            alpha=0.5, zorder=8)

    # ── Axis labels & limits ─────────────────────────────────────────────
    ax.set_xlabel("\nSpeed Reward  ($\\uparrow$ better)", fontsize=17,
                  labelpad=16)
    ax.set_ylabel("\nWaiting-Time Reward  ($\\uparrow$ better)", fontsize=17,
                  labelpad=16)
    ax.set_zlabel("\nQueue-Length Reward  ($\\uparrow$ better)", fontsize=17,
                  labelpad=16)

    ax.set_xlim(all_sp.min() - pad_sp, all_sp.max() + pad_sp)
    ax.set_ylim(all_wt.min() - pad_wt, all_wt.max() + pad_wt)
    ax.set_zlim(z_floor, all_qu.max() + pad_qu)

    # Camera angle: azim=225 gives the best cluster separation
    ax.view_init(elev=28, azim=225)
    ax.tick_params(axis="both", labelsize=14)

    # ── Legend below plot ────────────────────────────────────────────────
    handles = [
        Line2D([0], [0],
               marker=algo_markers.get(n, "o"),
               color="w",
               markerfacecolor=PALETTE[n],
               markersize=12,
               markeredgecolor="white",
               markeredgewidth=1.5,
               label=n, linewidth=0)
        for n in ALL_ALGOS if n in algo_data
    ]

    fig.suptitle(
        "3D Pareto Front in Objective Space\n(Convergence Region, last 25% Episodes)",
        fontsize=20, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    fig.legend(handles=handles,
               loc="lower center",
               ncol=len(handles),
               fontsize=14,
               framealpha=0.95,
               bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.2,
               handletextpad=0.4)

    out = os.path.join(OUT_DIR, "meta_mohrl_pareto_improved.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    setup_style()
    print("=" * 65)
    print("  Meta-MOHRL-v2 -- Seed & Pareto Visualisation Suite (v2)")
    print("=" * 65)

    print("\nLoading per-seed data ...")
    seed_data = load_seed_data()

    print("\nLoading all-algorithm data (seed 42) ...")
    algo_data = load_all_algos(seed=42)
    print(f"  Algorithms: {list(algo_data.keys())}")

    plot_5seeds_convergence(seed_data)
    plot_pareto_3d(algo_data)

    print("\n" + "=" * 65)
    print(f"  All figures saved to:  {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
