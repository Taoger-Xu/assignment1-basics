"""Try one TinyStories-sized optimizer step to probe a GPU's batch limit."""

from __future__ import annotations

import argparse

import torch

from cs336_basics.model.model import TransformerLM
from cs336_basics.optim.adamw import AdamW
from cs336_basics.train.loss import cross_entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    torch.manual_seed(336)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    model = TransformerLM(10000, 256, 512, 4, 16, 1344, 10000.0, device=device)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    token_ids = torch.randint(0, 10000, (args.batch_size, 256), device=device)
    targets = torch.randint(0, 10000, (args.batch_size, 256), device=device)
    logits = model(token_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    print(
        f"batch={args.batch_size} loss={loss.item():.4f} "
        f"peak_allocated_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f} "
        f"peak_reserved_GiB={torch.cuda.max_memory_reserved(device) / 2**30:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
