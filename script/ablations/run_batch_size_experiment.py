"""Tune and compare TinyStories training across single-GPU batch sizes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import subprocess
import sys


CONTEXT_LENGTH = 256
LEARNING_RATES: dict[int, tuple[float, float]] = {
    1: (1e-4, 3e-4),
    8: (3e-4, 8e-4),
    32: (6e-4, 1.5e-3),
    64: (8e-4, 2e-3),
    128: (1e-3, 2e-3),
    176: (1e-3, 2e-3),
}
VALID_LOSS = re.compile(r"step=(\d+) valid_loss=([-+\d.eE]+)")
FINAL_STATS = re.compile(
    r"completed_steps=(\d+) elapsed_seconds=([\d.]+) "
    r"processed_tokens=([\d,]+) tokens_per_second=([\d,]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128,176")
    parser.add_argument("--target-tokens", type=int, default=8_388_608)
    parser.add_argument("--pilot-tokens", type=int, default=1_048_576)
    parser.add_argument("--eval-points", type=int, default=20)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("batch_size_experiment/tinystories")
    )
    return parser.parse_args()


def train_command(
    *, batch_size: int, device: int, steps: int, learning_rate: float,
    eval_every: int, log_every: int, checkpoint: Path, swanlog: Path,
    name: str,
) -> list[str]:
    return [
        sys.executable, "-m", "script.train.train_lm",
        "--train-data", "data/tokenized/tinystories_10k/train.npy",
        "--valid-data", "data/tokenized/tinystories_10k/valid.npy",
        "--checkpoint", str(checkpoint),
        "--device", f"cuda:{device}",
        "--seed", "336",
        "--vocab-size", "10000",
        "--context-length", str(CONTEXT_LENGTH),
        "--d-model", "512",
        "--num-layers", "4",
        "--num-heads", "16",
        "--d-ff", "1344",
        "--rope-theta", "10000",
        "--batch-size", str(batch_size),
        "--max-steps", str(steps),
        "--max-lr", str(learning_rate),
        "--min-lr", str(learning_rate / 10),
        "--warmup-steps", str(max(1, steps // 10)),
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--eps", "1e-8",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--log-every", str(log_every),
        "--eval-every", str(eval_every),
        "--eval-batch-size", "128",
        "--eval-batches", "10",
        "--eval-seed", "90336",
        "--save-every", str(steps),
        "--swanlab-mode", "local",
        "--swanlab-project", "cs336-batch-size-experiment",
        "--swanlab-name", name,
        "--swanlab-logdir", str(swanlog),
    ]


def run_one(command: list[str], log_path: Path) -> tuple[float, dict[str, float | int]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    log = log_path.read_text(encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"Training failed ({completed.returncode}) in {log_path}:\n"
            + "\n".join(log.splitlines()[-12:])
        )
    validations = VALID_LOSS.findall(log)
    if not validations:
        raise RuntimeError(f"No validation loss in {log_path}")
    final_loss = float(validations[-1][1])
    if not math.isfinite(final_loss):
        raise RuntimeError(f"Non-finite validation loss in {log_path}")
    match = FINAL_STATS.search(log)
    if match is None:
        raise RuntimeError(f"No completion statistics in {log_path}")
    steps, seconds, tokens, throughput = match.groups()
    return final_loss, {
        "steps": int(steps),
        "seconds": float(seconds),
        "tokens": int(tokens.replace(",", "")),
        "tokens_per_second": int(throughput.replace(",", "")),
    }


def run_batch(
    batch_size: int, device: int, args: argparse.Namespace
) -> dict[str, object]:
    if batch_size not in LEARNING_RATES:
        raise ValueError(f"No learning-rate candidates for batch {batch_size}")
    pilot_budget = max(args.pilot_tokens, batch_size * CONTEXT_LENGTH * 64)
    pilot_steps = math.ceil(pilot_budget / (batch_size * CONTEXT_LENGTH))
    batch_dir = args.output_dir / f"batch_{batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for learning_rate in LEARNING_RATES[batch_size]:
        name = f"pilot_lr_{learning_rate:g}"
        log_path = batch_dir / f"{name}.log"
        print(f"batch={batch_size} pilot lr={learning_rate:g} starting", flush=True)
        command = train_command(
            batch_size=batch_size, device=device, steps=pilot_steps,
            learning_rate=learning_rate, eval_every=pilot_steps,
            log_every=max(1, pilot_steps // 5),
            checkpoint=batch_dir / f"{name}.pt",
            swanlog=batch_dir / name / "swanlog", name=f"b{batch_size}-{name}",
        )
        try:
            final_loss, stats = run_one(command, log_path)
        except RuntimeError as exc:
            trials.append({"lr": learning_rate, "error": str(exc)})
            print(f"batch={batch_size} pilot lr={learning_rate:g} failed", flush=True)
        else:
            trials.append({"lr": learning_rate, "valid_loss": final_loss, **stats})
            print(
                f"batch={batch_size} pilot lr={learning_rate:g} "
                f"valid_loss={final_loss:.4f}", flush=True,
            )

    successful = [trial for trial in trials if "valid_loss" in trial]
    if not successful:
        raise RuntimeError(f"All learning-rate trials failed for batch {batch_size}")
    best = min(successful, key=lambda trial: trial["valid_loss"])
    best_lr = float(best["lr"])
    full_steps = math.ceil(args.target_tokens / (batch_size * CONTEXT_LENGTH))
    full_log = batch_dir / "full.log"
    print(f"batch={batch_size} full run lr={best_lr:g} starting", flush=True)
    command = train_command(
        batch_size=batch_size, device=device, steps=full_steps,
        learning_rate=best_lr,
        eval_every=max(1, full_steps // args.eval_points),
        log_every=max(1, full_steps // (args.eval_points * 2)),
        checkpoint=batch_dir / "full.pt",
        swanlog=batch_dir / "full" / "swanlog",
        name=f"b{batch_size}-full",
    )
    final_loss, stats = run_one(command, full_log)
    print(f"batch={batch_size} full valid_loss={final_loss:.4f}", flush=True)
    result: dict[str, object] = {
        "batch_size": batch_size,
        "gpu": device,
        "pilot_steps": pilot_steps,
        "pilots": trials,
        "selected_lr": best_lr,
        "final_valid_loss": final_loss,
        "log": str(full_log),
        "checkpoint": str(batch_dir / "full.pt"),
        **stats,
    }
    (batch_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    batches = [int(value) for value in args.batch_sizes.split(",")]
    devices = [int(value) for value in args.devices.split(",")]
    if len(batches) != len(devices) or len(set(devices)) != len(devices):
        raise ValueError("Use one distinct GPU per batch size")
    if args.target_tokens <= 0 or args.pilot_tokens <= 0 or args.eval_points <= 0:
        raise ValueError("Token budgets and eval-points must be positive")
    for path in (
        Path("data/tokenized/tinystories_10k/train.npy"),
        Path("data/tokenized/tinystories_10k/valid.npy"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = {}
    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(run_batch, batch_size, device, args): batch_size
            for batch_size, device in zip(batches, devices, strict=True)
        }
        for future in as_completed(futures):
            batch_size = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures[batch_size] = str(exc)
                print(f"batch={batch_size} failed: {exc}", flush=True)

    results.sort(key=lambda item: int(item["batch_size"]))
    if args.merge_existing:
        previous_path = args.output_dir / "summary.json"
        if not previous_path.is_file():
            raise FileNotFoundError(previous_path)
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["target_tokens"] != args.target_tokens:
            raise ValueError("Existing summary uses a different token budget")
        merged = {
            int(result["batch_size"]): result
            for result in previous["results"]
        }
        merged.update({int(result["batch_size"]): result for result in results})
        results = [merged[batch] for batch in sorted(merged)]
        failures = {**previous["failures"], **failures}
    summary = {"target_tokens": args.target_tokens, "results": results, "failures": failures}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)
    if failures:
        raise RuntimeError(f"Batch runs failed: {list(failures)}")


if __name__ == "__main__":
    main()
