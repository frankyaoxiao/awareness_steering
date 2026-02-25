#!/usr/bin/env python
"""Run per-layer steering sweep to identify most effective layers.

Submits one Slurm job per layer, each using 1 GPU. Runs up to N layers in parallel.
Uses smaller sample size for quick iteration.

Usage:
    python scripts/layer_sweep.py --config configs/default.yaml --layers 0-63 --n 10 --num-prompts 20 --max-concurrent 4
"""

import argparse
import asyncio
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_DIR = Path(__file__).parent.parent.resolve()


def parse_layer_spec(spec):
    """Parse layer spec like '0-63' or '10,20,30' into a list of ints."""
    layers = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


def write_sbatch_script(layer, model, config, config_path, prompts_path, log_dir, slurm_cfg, sweep_cfg):
    """Write sbatch script for one layer experiment."""
    short_name = model["short_name"]
    partition = slurm_cfg.get("partition", "compute")
    total_gpus_per_node = slurm_cfg.get("gpus_per_node", 8)
    cpus_per_node = slurm_cfg.get("cpus_per_node", 64)
    cpus = cpus_per_node // total_gpus_per_node  # 1 GPU

    slurm_log_dir = log_dir / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    model_json = json.dumps(model)

    # Override config for this layer
    override = json.dumps({
        "steering": {"layers": [layer]},
        "sampling": {"n": sweep_cfg["n"], "batch_size": sweep_cfg.get("batch_size", 8)},
        "paths": {"num_prompts": sweep_cfg["num_prompts"]},
    })

    script = f"""#!/bin/bash
#SBATCH --job-name=layer{layer}_{short_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time=06:00:00
#SBATCH --output={slurm_log_dir}/layer_{layer}.out
#SBATCH --error={slurm_log_dir}/layer_{layer}.err

set -euo pipefail
cd {REPO_DIR}

export SCORER_MAX_CONCURRENT=50
export CUDA_VISIBLE_DEVICES=0

uv run python worker.py \\
    '{model_json}' \\
    '{config_path}' \\
    '{prompts_path}' \\
    '{log_dir / f"layer_{layer}"}' \\
    --override '{override}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix=f"layer{layer}_", dir=slurm_log_dir)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


async def slurm_worker(slot_id, queue, model, config, config_path, prompts_path, log_dir, slurm_cfg, sweep_cfg):
    """Pull layers from queue, submit one job per layer."""
    while True:
        try:
            layer = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        # Check if already done
        layer_dir = log_dir / f"layer_{layer}" / "scores"
        if layer_dir.exists():
            score_files = list(layer_dir.glob("*.jsonl"))
            if len(score_files) >= 2:
                print(f"[slot {slot_id}] Skipping layer {layer}: already scored")
                queue.task_done()
                continue

        script_path = write_sbatch_script(
            layer, model, config, config_path, prompts_path, log_dir, slurm_cfg, sweep_cfg
        )

        proc = await asyncio.create_subprocess_exec(
            "sbatch", "--parsable", "--wait", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job_id_line = await proc.stdout.readline()
        job_id = job_id_line.decode().strip()
        print(f"[slot {slot_id}] Submitted layer {layer} (job {job_id})")

        ret = await proc.wait()
        queue.task_done()

        if ret == 0:
            print(f"[slot {slot_id}] Done: layer {layer} (job {job_id})")
        else:
            print(f"[slot {slot_id}] FAILED: layer {layer} (job {job_id}, exit {ret})")


def analyze_sweep(log_dir, layers):
    """Analyze all layer results and rank by steering effectiveness."""
    print("\n" + "=" * 80)
    print("LAYER SWEEP RESULTS")
    print("=" * 80)

    results = []
    for layer in layers:
        layer_dir = log_dir / f"layer_{layer}"
        score_base = layer_dir / "scores"
        if not score_base.exists():
            continue

        for suffix, label in [("", "baseline"), ("_steered", "steered")]:
            # Find score files
            score_files = list(score_base.glob(f"*{suffix}.jsonl"))
            if not score_files:
                continue
            path = score_files[0]
            model_name = path.stem.replace("_steered", "")

            scores = [json.loads(line) for line in open(path)]
            total = len(scores)
            aware = sum(1 for s in scores if s["aware"] is True)
            failed = sum(1 for s in scores if s["aware"] is None)
            valid = total - failed
            rate = aware / valid if valid > 0 else 0

            if label == "baseline":
                base_rate = rate
            else:
                steer_rate = rate
                delta = base_rate - steer_rate
                results.append({
                    "layer": layer,
                    "baseline": base_rate,
                    "steered": steer_rate,
                    "delta": delta,
                    "n_valid": valid,
                })

    if not results:
        print("No results found.")
        return

    # Sort by delta (biggest reduction first)
    results.sort(key=lambda x: -x["delta"])

    print(f"\n{'Layer':>6} {'Baseline':>10} {'Steered':>10} {'Delta':>10} {'N':>6}")
    print("-" * 50)
    for r in results:
        marker = " ***" if r["delta"] > 0.10 else ""
        print(f"  {r['layer']:>4}   {r['baseline']:>8.2%}   {r['steered']:>8.2%}   {r['delta']:>+8.2%}   {r['n_valid']:>4}{marker}")

    # Save results
    csv_path = log_dir / "layer_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("layer,baseline,steered,delta,n_valid\n")
        for r in results:
            f.write(f"{r['layer']},{r['baseline']:.4f},{r['steered']:.4f},{r['delta']:.4f},{r['n_valid']}\n")
    print(f"\nSaved -> {csv_path}")

    # Recommendations
    top = [r for r in results if r["delta"] > 0.05]
    if top:
        print(f"\nTop layers (>5% reduction): {[r['layer'] for r in top]}")
    else:
        print("\nNo layers showed >5% reduction individually.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", default="0-63", help="Layer spec (e.g., '0-63' or '10,20,30')")
    parser.add_argument("--n", type=int, default=10, help="Samples per prompt")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent Slurm jobs")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    layers = parse_layer_spec(args.layers)
    model = config["models"][0]  # Use first model
    slurm_cfg = config.get("slurm", {})

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (Path("runs") / f"layer_sweep_{timestamp}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    shutil.copy2(config_path, log_dir / "config.yaml")

    prompts_path = Path(config["paths"]["prompts"]).resolve()

    sweep_cfg = {
        "n": args.n,
        "num_prompts": args.num_prompts,
        "batch_size": args.batch_size,
    }

    print(f"Log dir: {log_dir}")
    print(f"Model: {model['short_name']}")
    print(f"Layers: {len(layers)} ({layers[0]}-{layers[-1]})")
    print(f"Prompts: {args.num_prompts}, Samples: {args.n}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    queue = asyncio.Queue()
    for layer in layers:
        queue.put_nowait(layer)

    await asyncio.gather(
        *[
            slurm_worker(i, queue, model, config, config_path, prompts_path, log_dir, slurm_cfg, sweep_cfg)
            for i in range(args.max_concurrent)
        ]
    )

    analyze_sweep(log_dir, layers)


if __name__ == "__main__":
    asyncio.run(main())
