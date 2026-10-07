"""SGD example from Assignment 1, with a 1/sqrt(t+1) learning-rate decay."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch

"""
SGD 的核心任务是：拿到 loss.backward() 算出的梯度，原地更新模型参数。
这份实现还让学习率随每个参数的更新次数逐渐减小：
"""
class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    """
    backward() 把梯度写入每个参数的 parameter.grad。接下来的 step() 读取这些梯度并更新参数。
    它本身不会重新计算模型的前向传播或梯度
    optimizer.zero_grad()
    loss = ...
    loss.backward() backward() 把梯度写入每个参数的 parameter.grad
    optimizer.step()

    参数更新是训练算法的操作，不应被 PyTorch 记录为下一段可求导计算。否则 parameter.add_(...)
    这样的原地更新会与自动求导机制冲突。no_grad 让更新可以安全地原地进行。
    """
    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                """
                self.state 是 PyTorch 优化器为每个参数单独保存的状态。首次更新时没有 "t"，所以取 0；之后每更新一次，就把 t 加一。
                单独记录的好处是：某个参数如果连续几步没有梯度、被跳过，它自己的计数也不会增加。
                """
                t = state.get("t", 0)
                # parameter -= lr / math.sqrt(t + 1) * parameter.grad
                # add_ 末尾的下划线表示原地修改
                parameter.add_(parameter.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1
        return loss
