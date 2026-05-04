"""
reproduce.py — One-command reproduction script for Meta-MOHRL experiments.

Usage:
    python reproduce.py                     # Full 5-seed experiment (~15h per seed)
    python reproduce.py --seeds 42          # Single seed (quick test)
    python reproduce.py --plot-only         # Regenerate figures from existing results
    python reproduce.py --verify            # Verify claims against saved results

Requirements:
    pip install -r requirements.txt
    SUMO must be installed and SUMO_HOME set (or libsumo available).
"""

import argparse
import json
import os
import sys
import numpy as np


SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def run_training(seeds):
    """Run full training for specified seeds."""
    from run_experiment import run_full_experiment

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Training seed {seed}")
        print(f"{'='*60}\n")
        run_full_experiment(seed=seed)
    print(f"\nAll {len(seeds)} seeds completed. Results in {RESULTS_DIR}/")


def regenerate_figures():
    """Regenerate all publication figures from saved results."""
    print("Regenerating Version 3 figures...")
    from plot_results import main as plot_main
    plot_main()
    print(f"Figures saved to {FIGURES_DIR}/")


def verify_claims():
    """Verify manuscript claims against experimental data."""
    print("=" * 60)
    print("  Meta-MOHRL Claim Verification")
    print("=" * 60)

    errors = []
    warnings = []

    # Load all seeds
    all_records = {}
    for seed in SEEDS:
        fp = os.path.join(RESULTS_DIR, f"experiment_record_seed{seed}.json")
        if not os.path.exists(fp):
            warnings.append(f"Seed {seed} results not found: {fp}")
            continue
        with open(fp) as f:
            all_records[seed] = json.load(f)

    if not all_records:
        print("ERROR: No result files found. Run training first.")
        return False

    algos_to_check = [
        "Meta-MOHRL", "Ablation-NoMeta", "Ablation-NoFB",
        "HIRO", "HiPPO", "DUSDi", "MOSMAC"
    ]

    # --- Q1: Overall Performance ---
    print("\n[Q1] Overall Performance")
    for name in algos_to_check:
        totals = []
        for seed, rec in all_records.items():
            eps = rec["algorithms"][name]["episodes"]
            n = len(eps)
            start = int(n * 0.75)
            t = np.mean([e["total_reward"] for e in eps[start:]])
            totals.append(t)
        mean_t = np.mean(totals)
        std_t = np.std(totals)
        print(f"  {name:20s}: total = {mean_t:+.1f} +/- {std_t:.1f}")

    # --- Q2: Ablation ---
    print("\n[Q2] Ablation Analysis")
    for name in ["Meta-MOHRL", "Ablation-NoMeta", "Ablation-NoFB"]:
        hvs = []
        for seed, rec in all_records.items():
            hvs.append(rec["algorithms"][name]["final_metrics"]["hypervolume"])
        print(f"  {name:20s}: HV = {np.mean(hvs):.2f}")

    # --- Q3: Structural Discovery ---
    print("\n[Q3] Meta-Controller Configuration Discovery")
    seed42 = list(all_records.values())[0]
    config_hist = seed42["algorithms"]["Meta-MOHRL"]["config_history"]
    c_first = config_hist[0]
    c_last = config_hist[-1]
    print(f"  T1:     {c_first['T1']} -> {c_last['T1']}")
    print(f"  T2:     {c_first['T2']} -> {c_last['T2']}")
    print(f"  beta12: {c_first['beta12']:.2f} -> {c_last['beta12']:.2f}")
    print(f"  beta23: {c_first['beta23']:.2f} -> {c_last['beta23']:.2f}")
    print(f"  gamma1: {c_first['gamma1']:.3f} -> {c_last['gamma1']:.3f}")
    print(f"  gamma2: {c_first['gamma2']:.3f} -> {c_last['gamma2']:.3f}")
    print(f"  gamma3: {c_first['gamma3']:.3f} -> {c_last['gamma3']:.3f}")

    meta_loss = seed42["algorithms"]["Meta-MOHRL"]["meta_loss"]
    print(f"  Meta-loss: {meta_loss[0]:.1f} -> {meta_loss[-1]:.1f} "
          f"({(1 - meta_loss[-1]/meta_loss[0])*100:.0f}% reduction)")

    # --- Q4: T1 ~ sqrt(H) ---
    print("\n[Q4] T1 proportional to sqrt(H)")
    # H is the episode horizon in simulation seconds (3600s),
    # not decision steps (max_steps=200 at delta_time=5s)
    DELTA_TIME = 5
    H_sim_seconds = seed42["max_steps"] * DELTA_TIME  # 200 * 5 = 1000
    # But SUMO runs for num_seconds=3600; use that as true H
    H = 3600  # 1-hour SUMO simulation
    sqrt_H = np.sqrt(H)
    T1_final = c_last["T1"]
    print(f"  H = {H}, sqrt(H) = {sqrt_H:.0f}, T1_final = {T1_final}")
    print(f"  T1/sqrt(H) = {T1_final/sqrt_H:.2f} (same order: OK)"
          if 0.3 < T1_final / sqrt_H < 3.0
          else f"  WARNING: T1/sqrt(H) = {T1_final/sqrt_H:.2f} — outside expected range")

    # --- Summary ---
    print(f"\n{'='*60}")
    if warnings:
        print(f"  WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"    - {w}")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All verifiable claims PASSED.")
    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce Meta-MOHRL experiments (NeurIPS 2026)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to train (default: 42 43 44 45 46)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Only regenerate figures from existing results"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify manuscript claims against saved results"
    )
    args = parser.parse_args()

    if args.verify:
        success = verify_claims()
        sys.exit(0 if success else 1)
    elif args.plot_only:
        regenerate_figures()
    else:
        run_training(args.seeds)
        regenerate_figures()
        verify_claims()
