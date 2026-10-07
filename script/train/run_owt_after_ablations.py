"""Queue 32K-tokenizer OpenWebText pretraining after the TinyStories jobs finish."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_previous_jobs(baseline_pid: int, manifest_path: Path) -> None:
    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unfinished = [name for name, run in manifest.items() if run["exit_code"] is None]
        if not is_running(baseline_pid) and not unfinished:
            break
        print(
            f"waiting: baseline_running={is_running(baseline_pid)} "
            f"ablations_running={len(unfinished)}",
            flush=True,
        )
        time.sleep(30)

    failures = {name: run["exit_code"] for name, run in manifest.items() if run["exit_code"] != 0}
    if failures:
        print(f"Ablation jobs finished with errors: {failures}", flush=True)


def verify_baseline(checkpoint_path: Path, expected_steps: int) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    iteration = int(checkpoint["iteration"])
    if iteration != expected_steps:
        raise RuntimeError(f"Baseline checkpoint is at step {iteration}, expected {expected_steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pid", type=int, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("checkpoints/tinystories_10k.pt"))
    parser.add_argument("--ablations-manifest", type=Path, default=Path("ablations/tinystories/manifest.json"))
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if not 1 <= args.num_gpus <= torch.cuda.device_count():
        raise ValueError("num-gpus must fit the available CUDA devices")
    if 128 % args.num_gpus != 0:
        raise ValueError("num-gpus must divide the global batch size 128")

    wait_for_previous_jobs(args.baseline_pid, args.ablations_manifest)
    verify_baseline(args.baseline_checkpoint, 10000)

    print("Preparing OpenWebText with the 32K BPE tokenizer", flush=True)
    subprocess.run([sys.executable, "-m", "script.train.prepare_owt"], check=True)

    checkpoint = args.output_dir / "owt_32k.pt"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(args.num_gpus), "-m", "script.train.train_lm",
        "--train-data", "data/tokenized/owt_32k/train.npy",
        "--valid-data", "data/tokenized/owt_32k/valid.npy",
        "--checkpoint", str(checkpoint),
        "--seed", "336",
        "--architecture", "baseline",
        "--vocab-size", "32000",
        "--context-length", "256",
        "--d-model", "512",
        "--num-layers", "4",
        "--num-heads", "16",
        "--d-ff", "1344",
        "--rope-theta", "10000",
        "--batch-size", str(128 // args.num_gpus),
        "--grad-accum-steps", "1",
        "--total-tokens", "327680000",
        "--max-lr", "1e-3",
        "--min-lr", "1e-4",
        "--warmup-steps", "1000",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--eps", "1e-8",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--log-every", "100",
        "--eval-every", "1000",
        "--eval-batches", str(10 * args.num_gpus),
        "--save-every", "1000",
        "--swanlab-mode", "local",
        "--swanlab-project", "cs336-assignment1",
        "--swanlab-name", "owt-32k-baseline",
        "--swanlab-logdir", "swanlog",
    ]
    print("Starting OWT pretraining: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
