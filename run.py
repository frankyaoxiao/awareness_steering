#!/usr/bin/env python
"""Orchestrator: submit per-model Slurm jobs for steered generation + scoring."""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_DIR = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation steering experiment")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Output directory for this run. Default: runs/<timestamp>",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    return parser.parse_args()


def write_sbatch_script(model, config, config_path, prompts_path, log_dir, slurm_cfg):
    """Write a temporary sbatch script for one model. Returns the script path."""
    short_name = model["short_name"]
    tp = slurm_cfg.get("tensor_parallel_size", 4)
    job_name = slurm_cfg.get("job_name", "steering")
    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("timeout", "12:00:00")
    gpus_per_node = slurm_cfg.get("gpus_per_node", 8)
    cpus_per_node = slurm_cfg.get("cpus_per_node", 64)
    cpus = cpus_per_node // gpus_per_node * tp
    scorer_limit = config["scoring"]["max_concurrent"] // slurm_cfg.get("max_concurrent", 2)

    slurm_log_dir = log_dir / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    model_json = json.dumps(model)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{short_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={tp}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time={timeout}
#SBATCH --output={slurm_log_dir}/{short_name}.out
#SBATCH --error={slurm_log_dir}/{short_name}.err

set -euo pipefail
cd {REPO_DIR}

export SCORER_MAX_CONCURRENT={scorer_limit}

uv run python worker.py \\
    '{model_json}' \\
    '{config_path}' \\
    '{prompts_path}' \\
    '{log_dir}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix=f"steering_{short_name}_", dir=slurm_log_dir)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


async def slurm_worker(slot_id, queue, config, config_path, prompts_path, log_dir, slurm_cfg, n_prompts):
    """Pull models from shared queue, submit sbatch --wait for each."""
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        short_name = model["short_name"]

        # Skip if both conditions are already scored
        score_base = log_dir / "scores" / f"{short_name}.jsonl"
        score_steer = log_dir / "scores" / f"{short_name}_steered.jsonl"
        expected = config["sampling"]["n"] * n_prompts
        if score_base.exists() and score_steer.exists():
            n_base = sum(1 for _ in open(score_base))
            n_steer = sum(1 for _ in open(score_steer))
            if n_base >= expected and n_steer >= expected:
                print(f"[slot {slot_id}] Skipping {short_name}: already scored")
                queue.task_done()
                continue
            else:
                print(f"[slot {slot_id}] Incomplete {short_name}: re-running")
                for p in [score_base, score_steer,
                          log_dir / "completions" / f"{short_name}.jsonl",
                          log_dir / "completions" / f"{short_name}_steered.jsonl"]:
                    if p.exists():
                        p.unlink()

        script_path = write_sbatch_script(
            model, config, config_path, prompts_path, log_dir, slurm_cfg
        )

        proc = await asyncio.create_subprocess_exec(
            "sbatch", "--parsable", "--wait", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job_id_line = await proc.stdout.readline()
        job_id = job_id_line.decode().strip()
        print(f"[slot {slot_id}] Submitted {short_name} (job {job_id})")

        ret = await proc.wait()
        queue.task_done()

        if ret == 0:
            print(f"[slot {slot_id}] Done: {short_name} (job {job_id})")
        else:
            print(f"[slot {slot_id}] FAILED: {short_name} (job {job_id}, exit {ret})")


def analyze(config, log_dir):
    print("\n" + "=" * 60)
    print("RESULTS: Eval Awareness Rate")
    print("=" * 60)

    results = []
    score_dir = log_dir / "scores"

    for model in config["models"]:
        short_name = model["short_name"]
        for suffix, label in [("", "baseline"), ("_steered", "steered")]:
            name = f"{short_name}{suffix}"
            path = score_dir / f"{name}.jsonl"
            if not path.exists():
                print(f"  {name:>25}: NO DATA")
                continue

            scores = [json.loads(line) for line in open(path)]
            total = len(scores)
            aware = sum(1 for s in scores if s["aware"] is True)
            failed = sum(1 for s in scores if s["aware"] is None)
            valid = total - failed
            rate = aware / valid if valid > 0 else 0

            results.append({
                "model": name,
                "total": total,
                "aware": aware,
                "failed": failed,
                "awareness_rate": rate,
            })
            print(f"  {name:>25}: {rate:6.2%}  ({aware}/{valid}, {failed} failed)")

    csv_path = log_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,total,aware,failed,awareness_rate\n")
        for r in results:
            f.write(
                f"{r['model']},{r['total']},{r['aware']},{r['failed']},{r['awareness_rate']:.4f}\n"
            )
    print(f"\nSaved -> {csv_path}")


async def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    slurm_cfg = config.get("slurm", {})

    # Resolve all paths to absolute
    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (Path("runs") / timestamp).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    shutil.copy2(config_path, log_dir / "config.yaml")

    prompts_path = Path(config["paths"]["prompts"]).resolve()
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found.")
        sys.exit(1)

    shutil.copy2(prompts_path, log_dir / "prompts.jsonl")

    n_prompts = sum(1 for _ in open(prompts_path))
    num_prompts = config["paths"].get("num_prompts")
    if num_prompts:
        n_prompts = min(n_prompts, num_prompts)

    # Pre-flight: check steering vectors exist
    vectors_dir = Path(config["paths"]["vectors"]).resolve()
    missing = []
    for model in config["models"]:
        vp = vectors_dir / f"{model['short_name']}.pt"
        if not vp.exists():
            missing.append(model["short_name"])
    if missing:
        print(f"ERROR: Missing steering vectors for: {', '.join(missing)}")
        print(f"Run: python extract_vectors.py --config {args.config}")
        sys.exit(1)

    max_concurrent = slurm_cfg.get("max_concurrent", 2)

    print(f"Log dir: {log_dir}")
    print(f"Prompts: {n_prompts} (from {prompts_path})")
    print(f"Models: {len(config['models'])}")
    print(f"Max concurrent Slurm jobs: {max_concurrent}")
    print()

    # Shared work queue
    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    await asyncio.gather(
        *[
            slurm_worker(i, queue, config, config_path, prompts_path, log_dir, slurm_cfg, n_prompts)
            for i in range(max_concurrent)
        ]
    )

    analyze(config, log_dir)


if __name__ == "__main__":
    asyncio.run(main())
