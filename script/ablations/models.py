"""Architecture variants for the TinyStories ablation experiments."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from cs336_basics.model.model import Linear, TransformerBlock, TransformerLM


Architecture = Literal["baseline", "no_rmsnorm", "post_norm", "nope", "silu"]
ARCHITECTURES: tuple[Architecture, ...] = (
    "baseline", "no_rmsnorm", "post_norm", "nope", "silu"
)


class PostNormBlock(nn.Module):
    """Keep the baseline parameters but move both norms after their residuals."""

    def __init__(self, block: TransformerBlock) -> None:
        super().__init__()
        self.ln1 = block.ln1
        self.attn = block.attn
        self.ln2 = block.ln2
        self.ffn = block.ffn

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self.ln1(x + self.attn(x, token_positions=token_positions))
        return self.ln2(z + self.ffn(z))


class SiLUFeedForward(nn.Module):
    """Two-matrix FFN; width 4*d_model approximately matches SwiGLU params."""

    def __init__(
        self, d_model: int, device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        super().__init__()
        d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        return self.w2(hidden * torch.sigmoid(hidden))


def build_model(
    architecture: Architecture,
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TransformerLM:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )

    if architecture == "no_rmsnorm":
        for layer in model.layers:
            layer.ln1 = nn.Identity()
            layer.ln2 = nn.Identity()
        model.ln_final = nn.Identity()
    elif architecture == "post_norm":
        model.layers = nn.ModuleList(PostNormBlock(layer) for layer in model.layers)
    elif architecture == "nope":
        for layer in model.layers:
            layer.attn.rope = None
    elif architecture == "silu":
        for layer in model.layers:
            layer.ffn = SiLUFeedForward(d_model, device, dtype)

    return model
