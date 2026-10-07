import torch

from cs336_basics.model.model import TransformerLM
from cs336_basics.tokenizer.api import Tokenizer


@torch.inference_mode()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    eos_token_id: int,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            # 模型一次最多接收 context_length 个 token。
            context = token_ids[-model.context_length:]
            inputs = torch.tensor([context], dtype=torch.long, device=device)

            # 只取最后一个位置对「下一个 token」的预测。
            logits = model(inputs)[0, -1, :]
            probabilities = torch.softmax(logits / temperature, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_ids = torch.sort(
                    probabilities, descending=True
                )

                # 保留概率最大的 token，直到累计概率达到 top_p；
                # 跨过阈值的那个 token 也要保留。
                previous_cumulative = (
                    sorted_probs.cumsum(dim=-1) - sorted_probs
                )
                keep = previous_cumulative < top_p

                sorted_probs = sorted_probs * keep
                chosen_in_sorted = torch.multinomial(sorted_probs, 1)
                next_id = sorted_ids[chosen_in_sorted].item()
            else:
                next_id = torch.multinomial(probabilities, 1).item()

            if next_id == eos_token_id:
                break

            token_ids.append(next_id)

    finally:
        model.train(was_training)

    return tokenizer.decode(token_ids)