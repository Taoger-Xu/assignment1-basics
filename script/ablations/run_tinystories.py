"""Run matched-budget TinyStories architecture ablations on separate GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = (
    ("no_rmsnorm_base_lr", "no_rmsnorm", "2e-3"),
    ("no_rmsnorm_mid_lr", "no_rmsnorm", "5e-4"),
    ("no_rmsnorm_low_lr", "no_rmsnorm", "1e-4"),
    ("post_norm", "post_norm", "2e-3"),
    ("nope", "nope", "2e-3"),
    ("silu", "silu", "2e-3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="1,2,3,4,5,6")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("ablations/tinystories"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = [value.strip() for value in args.devices.split(",")]
    if len(devices) != len(EXPERIMENTS) or len(set(devices)) != len(devices):
        raise ValueError(f"Provide {len(EXPERIMENTS)} distinct GPU indices")
    if not 1000 < args.steps <= 10000:
        raise ValueError("steps must be between 1001 and 10000")

    train_data = Path("data/tokenized/tinystories_10k/train.npy")
    valid_data = Path("data/tokenized/tinystories_10k/valid.npy")
    if not train_data.is_file() or not valid_data.is_file():
        raise FileNotFoundError("Run script/train/prepare_tinystories.py first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    active: list[tuple[str, subprocess.Popen[bytes], object]] = []
    manifest: dict[str, dict[str, str | int | None]] = {}
    try:
        for device, (name, architecture, max_lr) in zip(devices, EXPERIMENTS, strict=True):
            checkpoint = args.output_dir / f"{name}.pt"
            log_path = args.output_dir / f"{name}.log"
            command = [
                sys.executable, "-m", "script.train.train_lm",
                "--train-data", str(train_data),
                "--valid-data", str(valid_data),
                "--checkpoint", str(checkpoint),
                "--device", f"cuda:{device}",
                "--seed", "336",
                "--architecture", architecture,
                "--vocab-size", "10000",
                "--context-length", "256",
                "--d-model", "512",
                "--num-layers", "4",
                "--num-heads", "16",
                "--d-ff", "1344",
                "--rope-theta", "10000",
                "--batch-size", "128",
                "--max-steps", str(args.steps),
                "--cosine-cycle-steps", "10000",
                "--max-lr", max_lr,
                "--min-lr", "1e-4" if max_lr == "2e-3" else "1e-5",
                "--warmup-steps", "1000",
                "--beta1", "0.9",
                "--beta2", "0.95",
                "--eps", "1e-8",
                "--weight-decay", "0.1",
                "--grad-clip", "1.0",
                "--log-every", "100",
                "--eval-every", "1000",
                "--eval-batches", "10",
                "--save-every", "1000",
                "--swanlab-mode", "local",
                "--swanlab-project", "cs336-assignment1-ablations",
                "--swanlab-name", name,
                "--swanlab-logdir", str(args.output_dir / "swanlog"),
            ]
            logfile = log_path.open("wb")
            process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
            active.append((name, process, logfile))
            manifest[name] = {
                "architecture": architecture,
                "max_lr": max_lr,
                "gpu": int(device),
                "steps": args.steps,
                "pid": process.pid,
                "checkpoint": str(checkpoint),
                "log": str(log_path),
                "exit_code": None,
            }
            print(f"started {name} on cuda:{device} pid={process.pid}", flush=True)

        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        while active:
            time.sleep(20)
            for name, process, logfile in active[:]:
                exit_code = process.poll()
                if exit_code is None:
                    continue
                logfile.close()
                manifest[name]["exit_code"] = exit_code
                active.remove((name, process, logfile))
                print(f"finished {name} exit_code={exit_code}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        for _, process, _ in active:
            process.terminate()
        for _, process, logfile in active:
            process.wait()
            logfile.close()
        raise


if __name__ == "__main__":
    main()
