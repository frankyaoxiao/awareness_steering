#!/usr/bin/env python
"""Extract contrastive steering vectors from models.

Orchestrator mode (dispatches Slurm jobs):
    python extract_vectors.py --config configs/default.yaml

Worker mode (called by Slurm):
    python extract_vectors.py --worker --model '{"id": "...", "short_name": "..."}' --config configs/default.yaml
"""

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Worker: extract steering vector for one model
# ---------------------------------------------------------------------------


def extract_vector(model_id, contrastive_prompts_path, vectors_dir):
    """Load model, run contrastive prompts, compute mean diff per layer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(contrastive_prompts_path) as f:
        prompts = json.load(f)

    deploy_prompts = prompts["deployment"]
    eval_prompts = prompts["evaluation"]
    n_deploy = len(deploy_prompts)

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()

    # Format all prompts with chat template
    all_texts = []
    for prompt in deploy_prompts + eval_prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        all_texts.append(text)

    # Tokenize with left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(all_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Running forward pass ({len(all_texts)} prompts, seq_len={inputs['input_ids'].shape[1]})...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (num_layers+1) tensors, each (batch, seq_len, hidden_size)
    # Index 0 = embedding output, 1..num_layers = layer outputs
    hidden_states = outputs.hidden_states

    # Find last non-pad token position for each sequence
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

    # Compute per-layer mean difference
    steering_vectors = []
    for layer_idx in range(1, len(hidden_states)):  # skip embedding
        hs = hidden_states[layer_idx]  # (batch, seq_len, hidden_size)

        # Extract activation at last non-pad token for each prompt
        acts = hs[torch.arange(hs.size(0), device=hs.device), last_positions]  # (batch, hidden)

        deploy_mean = acts[:n_deploy].float().mean(dim=0)
        eval_mean = acts[n_deploy:].float().mean(dim=0)

        # Deployment direction: deploy - eval
        diff = (deploy_mean - eval_mean).to(torch.bfloat16)
        steering_vectors.append(diff)

    result = torch.stack(steering_vectors).cpu()  # (num_layers, hidden_size)
    print(f"Steering vector shape: {result.shape}")
    return result


def worker_main(args):
    """Worker mode: extract vector for one model and save."""
    import torch

    model = json.loads(args.model)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_id = model["id"]
    short_name = model["short_name"]
    vectors_dir = Path(config["paths"]["vectors"])
    vectors_dir.mkdir(parents=True, exist_ok=True)
    output_path = vectors_dir / f"{short_name}.pt"

    contrastive_prompts_path = Path(config["steering"]["contrastive_prompts"])

    vector = extract_vector(model_id, contrastive_prompts_path, vectors_dir)
    torch.save(vector, output_path)
    print(f"Saved: {output_path} (shape {vector.shape})")


# ---------------------------------------------------------------------------
# Orchestrator: dispatch Slurm jobs
# ---------------------------------------------------------------------------


def write_sbatch_script(model, config, config_path, slurm_cfg):
    """Write a temporary sbatch script for vector extraction."""
    short_name = model["short_name"]
    partition = slurm_cfg.get("partition", "compute")
    gpus_per_node = slurm_cfg.get("gpus_per_node", 8)
    cpus_per_node = slurm_cfg.get("cpus_per_node", 64)
    cpus = cpus_per_node // gpus_per_node  # 1 GPU worth of CPUs

    log_dir = REPO_DIR / "vectors" / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_json = json.dumps(model)

    script = f"""#!/bin/bash
#SBATCH --job-name=extract_{short_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time=0:30:00
#SBATCH --output={log_dir}/{short_name}.out
#SBATCH --error={log_dir}/{short_name}.err

set -euo pipefail
cd {REPO_DIR}

uv run python extract_vectors.py \\
    --worker \\
    --model '{model_json}' \\
    --config '{config_path}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix=f"extract_{short_name}_", dir=log_dir)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


async def slurm_worker(slot_id, queue, config, config_path, vectors_dir):
    """Pull models from shared queue, submit extraction jobs."""
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        short_name = model["short_name"]
        output_path = vectors_dir / f"{short_name}.pt"

        if output_path.exists():
            print(f"[slot {slot_id}] Skipping {short_name}: vector already exists")
            queue.task_done()
            continue

        slurm_cfg = config.get("slurm", {})
        script_path = write_sbatch_script(model, config, config_path, slurm_cfg)

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


async def orchestrator_main(args):
    """Orchestrator mode: dispatch extraction jobs via Slurm."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_path = Path(args.config).resolve()
    vectors_dir = Path(config["paths"]["vectors"]).resolve()
    vectors_dir.mkdir(parents=True, exist_ok=True)

    slurm_cfg = config.get("slurm", {})
    max_concurrent = slurm_cfg.get("max_concurrent", 4)

    print(f"Vectors dir: {vectors_dir}")
    print(f"Models: {len(config['models'])}")
    print(f"Max concurrent Slurm jobs: {max_concurrent}")
    print()

    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    await asyncio.gather(
        *[
            slurm_worker(i, queue, config, config_path, vectors_dir)
            for i in range(max_concurrent)
        ]
    )

    # Summary
    print("\n" + "=" * 40)
    for model in config["models"]:
        path = vectors_dir / f"{model['short_name']}.pt"
        status = "OK" if path.exists() else "MISSING"
        print(f"  {model['short_name']:>15}: {status}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Extract contrastive steering vectors")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--worker", action="store_true", help="Run in worker mode (called by Slurm)")
    parser.add_argument("--model", type=str, help="Model JSON (worker mode only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        asyncio.run(orchestrator_main(args))
