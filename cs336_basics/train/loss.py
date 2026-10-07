import torch


def cross_entropy(
    logits: torch.Tensor,   # (..., vocab_size)
    targets: torch.Tensor,  # (...)
) -> torch.Tensor:
    """计算所有样本及位置的平均交叉熵。"""
    # 减去每组最大值，防止 exp 溢出。
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    # log(Σ exp(s_k))：沿最后一维，即词表维求和。
    log_denominator = torch.log(torch.exp(shifted).sum(dim=-1))

    # s_t：取出每组中正确 token 对应的 logit。
    correct_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

      # 每组的损失为 log(Σ exp(s_k)) - s_t，再对所有组取平均
    return (log_denominator - correct_logits).mean()