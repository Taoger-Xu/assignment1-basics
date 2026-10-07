from collections.abc import Iterable

import torch


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 所有参数的梯度共同构成一个向量，计算其整体 L2 范数。
    per_parameter_norms = [
        torch.linalg.vector_norm(grad.float()) for grad in grads
    ]
    total_norm = torch.linalg.vector_norm(
        torch.stack(per_parameter_norms)
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)