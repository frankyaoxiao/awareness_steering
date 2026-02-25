#!/usr/bin/env python
"""Analyze steering results with double-layer bootstrap CIs (matching fortress)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_ci(score_path, n_boot=10000, ci=95, seed=42):
    """Two-level cluster bootstrap CI for awareness rate.

    Level 1: resample prompts with replacement.
    Level 2: resample completions within each prompt (binomial draw).
    """
    rng = np.random.default_rng(seed)

    # Group by prompt
    prompt_scores = defaultdict(lambda: {"aware": 0, "valid": 0})
    with open(score_path) as f:
        for line in f:
            row = json.loads(line)
            if row["aware"] is not None:
                pid = row["prompt_id"]
                prompt_scores[pid]["valid"] += 1
                if row["aware"]:
                    prompt_scores[pid]["aware"] += 1

    aware = np.array([s["aware"] for s in prompt_scores.values()])
    valid = np.array([s["valid"] for s in prompt_scores.values()])
    k = len(aware)

    if k == 0:
        return 0.0, 0.0

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        a, v = aware[idx], valid[idx]
        resampled = rng.binomial(v, np.where(v > 0, a / v, 0))
        boot_means[i] = resampled.sum() / v.sum() if v.sum() > 0 else 0

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return lo, hi


def analyze(run_dir):
    run_dir = Path(run_dir)
    scores_dir = run_dir / "scores"

    if not scores_dir.exists():
        print(f"No scores directory found at {scores_dir}")
        return

    # Find all score files
    score_files = sorted(scores_dir.glob("*.jsonl"))
    if not score_files:
        print("No score files found.")
        return

    # Group into baseline / steered pairs
    models = set()
    for f in score_files:
        name = f.stem
        if name.endswith("_steered"):
            models.add(name[: -len("_steered")])
        else:
            models.add(name)

    print("=" * 70)
    print(f"RESULTS: {run_dir.name}")
    print("=" * 70)
    print()

    for model in sorted(models):
        baseline_path = scores_dir / f"{model}.jsonl"
        steered_path = scores_dir / f"{model}_steered.jsonl"

        for label, path in [("baseline", baseline_path), ("steered", steered_path)]:
            if not path.exists():
                print(f"  {model} ({label}): NO DATA")
                continue

            scores = [json.loads(line) for line in open(path)]
            total = len(scores)
            aware = sum(1 for s in scores if s["aware"] is True)
            failed = sum(1 for s in scores if s["aware"] is None)
            valid = total - failed
            rate = aware / valid if valid > 0 else 0

            lo, hi = bootstrap_ci(path)

            print(
                f"  {model:>20} ({label:>8}): "
                f"{rate:6.2%}  [{lo:6.2%}, {hi:6.2%}]  "
                f"({aware}/{valid}, {failed} failed)"
            )

        # Check for non-overlapping CIs
        if baseline_path.exists() and steered_path.exists():
            b_lo, b_hi = bootstrap_ci(baseline_path)
            s_lo, s_hi = bootstrap_ci(steered_path)
            overlap = not (b_hi < s_lo or s_hi < b_lo)
            print(f"  {'':>20} CIs {'overlap' if overlap else 'DO NOT overlap (significant!)'}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory")
    args = parser.parse_args()
    analyze(args.run_dir)
