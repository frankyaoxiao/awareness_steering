#!/usr/bin/env python
"""Plot steering results as a bar chart with bootstrap CIs, matching fortress style."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse fortress style
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fortress" / "style"))
from plot_config import setup_style, apply_suptitle, COLORS

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def bootstrap_ci(score_path, n_boot=10000, ci=95, seed=42):
    """Two-level cluster bootstrap CI for awareness rate."""
    rng = np.random.default_rng(seed)

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


def plot_steering(run_dir, output=None, title=None):
    run_dir = Path(run_dir)
    scores_dir = run_dir / "scores"

    if not scores_dir.exists():
        print(f"ERROR: {scores_dir} not found")
        return

    # Collect baseline and steered pairs
    labels = []
    rates = []
    ci_lo = []
    ci_hi = []
    bar_colors = []

    # Find models
    score_files = sorted(scores_dir.glob("*.jsonl"))
    models = set()
    for f in score_files:
        name = f.stem
        if name.endswith("_steered"):
            models.add(name[:-len("_steered")])
        else:
            models.add(name)

    for model in sorted(models):
        for suffix, label, color_idx in [("", "Baseline", 0), ("_steered", "Steered", 2)]:
            path = scores_dir / f"{model}{suffix}.jsonl"
            if not path.exists():
                continue

            scores = [json.loads(line) for line in open(path)]
            total = len(scores)
            aware_count = sum(1 for s in scores if s["aware"] is True)
            failed = sum(1 for s in scores if s["aware"] is None)
            valid = total - failed
            obs = aware_count / valid if valid > 0 else 0

            lo, hi = bootstrap_ci(path)

            labels.append(f"{model}\n{label}")
            rates.append(obs * 100)
            ci_lo.append((obs - lo) * 100)
            ci_hi.append((hi - obs) * 100)
            bar_colors.append(COLORS[color_idx])

            print(f"  {model} ({label}): {obs:.2%}  [{lo:.2%}, {hi:.2%}]  ({aware_count}/{valid})")

    yerr = [ci_lo, ci_hi]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.2))
    bars = ax.bar(labels, rates, color=bar_colors, width=0.6, edgecolor="none",
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.2, "color": "#333333"})

    for bar, rate, hi in zip(bars, rates, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Eval Awareness (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    max_upper = max(r + h for r, h in zip(rates, ci_hi)) if rates else 10
    ax.set_ylim(0, max_upper * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)

    if title:
        apply_suptitle(fig, title, fontsize=14, ax=ax)
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.88)

    if output is None:
        output = Path("figs") / run_dir.name / "steering.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved -> {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent.parent / "fortress" / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    plot_steering(args.run_dir, args.output, args.title)
