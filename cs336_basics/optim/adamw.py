from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                state = self.state[parameter]

                # 首次遇到该参数时，创建与参数同形状的矩估计。
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)

                state["step"] += 1
                t = state["step"]
                # 历史信息
                m = state["m"] # 方向
                v = state["v"] # 大小

                # m_t = beta1*m_(t-1) + (1-beta1)*g_t
                m.copy_(beta1 * m + (1 - beta1) * grad)
                # v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
                v.copy_(beta2 * v + (1 - beta2) * grad.square())

                # 修正系数
                adjusted_lr = (
                    lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                )

                # 解耦的权重衰减。
                parameter.mul_(1 - lr * weight_decay)

                # 自适应梯度更新。
                parameter -= adjusted_lr * m / (v.sqrt() + eps)

        return loss
