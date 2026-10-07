import numpy as np
import torch


class TokenBatchLoader:
    def __init__(
        self,
        dataset: np.ndarray,
        context_length: int,
        device: str,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(dataset) <= context_length:
            raise ValueError("dataset is too short")

        self.dataset = dataset
        self.context_length = context_length
        self.device = device

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # 随机抽样，# shape: (3,)
        starts = np.random.randint(
            0,
            len(self.dataset) - self.context_length,
            size=batch_size,
        )

        # [0, 1, 2, 3]，shape: (4,)
        offsets = np.arange(self.context_length)
        indices = starts[:, None] + offsets[None, :]

        inputs = self.dataset[indices]
        targets = self.dataset[indices + 1]

        return (
            torch.as_tensor(inputs, dtype=torch.long, device=self.device),
            torch.as_tensor(targets, dtype=torch.long, device=self.device),
        )
