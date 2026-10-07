from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import time
from pathlib import Path

import numpy as np
import swanlab
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cs336_basics.model.model import TransformerLM
from cs336_basics.optim.adamw import AdamW
from cs336_basics.train.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.train.data import TokenBatchLoader
from cs336_basics.train.grad_clip import clip_gradients
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.schedule import cosine_learning_rate_schedule
from script.ablations.models import ARCHITECTURES, build_model

"""
每隔若干步切换到 model.eval()，
在 torch.no_grad() 下抽取若干个验证 batch，报告平均验证损失
"""
@torch.no_grad()
def evaluate(
    model: TransformerLM,
    loader: TokenBatchLoader,
    batch_size: int,
    num_batches: int,
) -> float:
    model.eval()
    losses = []

    for _ in range(num_batches):
        inputs, targets = loader.sample(batch_size)
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets).item())

    model.train()
    return sum(losses) / len(losses)

"""
解析命令行参数。**至少提供训练集和验证集路径、checkpoint 路径、设备、随机种子，
以及 vocab_size、context_length、d_model、num_layers、num_heads、d_ff、
rope_theta、batch_size、训练步数、AdamW 超参数、学习率调度参数、裁剪阈值和日志间隔。
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=336)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="baseline")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--total-tokens", type=int)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-cycle-steps", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--swanlab-project", default="cs336-assignment1")
    parser.add_argument("--swanlab-name", default=None)
    parser.add_argument("--swanlab-mode", choices=("local", "offline", "online"), default="local")
    parser.add_argument("--swanlab-logdir", type=Path, default=Path("swanlog"))

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    is_primary = rank == 0
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl", device_id=torch.device(args.device))

    if args.total_tokens is not None:
        if args.total_tokens <= 0:
            raise ValueError("total-tokens must be positive")
        if args.max_steps is not None:
            raise ValueError("Set either --total-tokens or --max-steps, not both")
        tokens_per_step = (
            args.batch_size * args.grad_accum_steps * args.context_length * world_size
        )
        args.max_steps = (args.total_tokens + tokens_per_step - 1) // tokens_per_step
    elif args.max_steps is None:
        args.max_steps = 1000

    if args.cosine_cycle_steps is None:
        args.cosine_cycle_steps = args.max_steps

    if args.max_steps <= args.warmup_steps:
        raise ValueError("max_steps must be greater than warmup_steps")
    if args.cosine_cycle_steps <= args.warmup_steps:
        raise ValueError("cosine-cycle-steps must be greater than warmup-steps")
    if min(
        args.batch_size,
        args.grad_accum_steps,
        args.log_every,
        args.eval_every,
        args.eval_batches,
        args.save_every,
    ) <= 0:
        raise ValueError("batch size and intervals must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)

    if str(args.device).startswith("cuda"):
        # Use the GPU's TF32 matrix-multiply path while keeping model tensors
        # and optimizer state in float32.
        torch.backends.cuda.matmul.allow_tf32 = True

    # 只建立文件映射；抽样时才读取需要的 token。
    train_tokens = np.load(args.train_data, mmap_mode="r")
    valid_tokens = np.load(args.valid_data, mmap_mode="r")

    train_loader = TokenBatchLoader(
        train_tokens, args.context_length, args.device
    )
    valid_loader = TokenBatchLoader(
        valid_tokens, args.context_length, args.device
    )

    model = build_model(
        architecture=args.architecture,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if is_primary:
        print(
            f"model_parameters={parameter_count:,} batch_size_per_gpu={args.batch_size} "
            f"world_size={world_size} grad_accum_steps={args.grad_accum_steps} "
            f"context_length={args.context_length} steps={args.max_steps} "
            f"tokens_per_step={args.batch_size * args.grad_accum_steps * args.context_length * world_size:,}",
            flush=True,
        )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        if is_primary:
            print(f"Resumed after step {start_step}")

    train_model = (
        DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else model
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = None
    if is_primary:
        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_name,
            mode=args.swanlab_mode,
            logdir=str(args.swanlab_logdir),
            config={**config, "world_size": world_size},
        )
        swanlab.define_metric("train/loss_vs_time", x_axis="time/wall_seconds")
        swanlab.define_metric("valid/loss_vs_time", x_axis="time/wall_seconds")
    started_at = time.perf_counter()

    try:
        for step in range(start_step, args.max_steps):
            lr = cosine_learning_rate_schedule(
                it=step,
                max_learning_rate=args.max_lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_steps,
                cosine_cycle_iters=args.cosine_cycle_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_model.train()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            for micro_step in range(args.grad_accum_steps):
                sync_context = (
                    train_model.no_sync()
                    if distributed and micro_step < args.grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    inputs, targets = train_loader.sample(args.batch_size)
                    logits = train_model(inputs)
                    loss = cross_entropy(logits, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss at step {step + 1}: {loss.item()}"
                        )
                    accumulated_loss += loss.item()
                    (loss / args.grad_accum_steps).backward()
            clip_gradients(train_model.parameters(), args.grad_clip)
            optimizer.step()

            completed_steps = step + 1
            metrics: dict[str, float | int] = {}

            if is_primary and (
                completed_steps % args.log_every == 0 or completed_steps == args.max_steps
            ):
                train_loss = accumulated_loss / args.grad_accum_steps
                metrics["train/loss"] = train_loss
                metrics["train/loss_vs_time"] = train_loss
                print(
                    f"step={completed_steps} lr={lr:.6g} train_loss={train_loss:.4f}",
                    flush=True,
                )

            if is_primary and (
                completed_steps % args.eval_every == 0 or completed_steps == args.max_steps
            ):
                if args.eval_seed is not None:
                    random_state = np.random.get_state()
                    eval_index = (completed_steps + args.eval_every - 1) // args.eval_every
                    np.random.seed(args.eval_seed + eval_index)
                try:
                    valid_loss = evaluate(
                        model,
                        valid_loader,
                        args.eval_batch_size or args.batch_size,
                        args.eval_batches,
                    )
                finally:
                    if args.eval_seed is not None:
                        np.random.set_state(random_state)
                metrics["valid/loss"] = valid_loss
                metrics["valid/loss_vs_time"] = valid_loss
                print(f"step={completed_steps} valid_loss={valid_loss:.4f}", flush=True)

            if metrics and run is not None:
                run.log(
                    {
                        "time/wall_seconds": time.perf_counter() - started_at,
                        "train/lr": lr,
                        **metrics,
                    },
                    step=completed_steps,
                )

            if is_primary and completed_steps % args.save_every == 0 and completed_steps < args.max_steps:
                save_checkpoint(model, optimizer, completed_steps, args.checkpoint)

        # 最后一步即使不在保存间隔上，也保存一次。
        if is_primary:
            save_checkpoint(model, optimizer, args.max_steps, args.checkpoint)
            elapsed = time.perf_counter() - started_at
            processed_tokens = (
                (args.max_steps - start_step)
                * args.batch_size
                * args.grad_accum_steps
                * args.context_length
                * world_size
            )
            print(
                f"completed_steps={args.max_steps} elapsed_seconds={elapsed:.1f} "
                f"processed_tokens={processed_tokens:,} "
                f"tokens_per_second={processed_tokens / elapsed:,.0f}",
                flush=True,
            )
        if distributed:
            dist.barrier(device_ids=[local_rank])
    finally:
        if run is not None:
            run.finish()
        if distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
