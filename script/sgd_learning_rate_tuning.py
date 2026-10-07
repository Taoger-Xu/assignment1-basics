"""Run the assignment's toy SGD experiment for several learning rates."""

from __future__ import annotations

import torch

from cs336_basics.optim.sgd import SGD


def main() -> None:
    torch.manual_seed(336)
    initial_weights = 5 * torch.randn(10, 10)

    for lr in (1.0, 10.0, 100.0, 1000.0):
        weights = torch.nn.Parameter(initial_weights.clone())
        optimizer = SGD([weights], lr=lr)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = weights.square().mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"lr={lr:g}: " + ", ".join(f"{value:.4g}" for value in losses))


if __name__ == "__main__":
    main()
