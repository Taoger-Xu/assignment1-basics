# cs336_basics/tokenizer/train.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .corpus import CorpusReader
from .specification import TokenizerSpec, create_byte_vocabulary
from .statistics import (
    IncrementalPairStatistics,
)


@dataclass
class BPETrainer:
    """
    BPE tokenizer 训练器。

    vocab_size 包含：
    - 256 个初始 byte token；
    - special tokens；
    - 所有 merge 产生的新 token。
    """

    vocab_size: int
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")

        initial_size = 256 + len(self.special_tokens)

        if self.vocab_size < initial_size:
            raise ValueError(
                "vocab_size must be at least "
                f"{initial_size}, got {self.vocab_size}"
            )

    def initialize_spec(self) -> TokenizerSpec:
        """创建包含初始 byte token 和 special token 的 tokenizer 状态。"""
        return create_byte_vocabulary(self.special_tokens)

    def train_from_sequences(
        self,
        sequences: dict[tuple[bytes, ...], int],
    ) -> TokenizerSpec:
        """
        根据已经预处理好的字节序列训练 BPE。

        sequences 的格式：

            {
                (b"l", b"o", b"w"): 5,
                (b"l", b"o", b"w", b"e", b"r"): 2,
            }
        """
        spec = self.initialize_spec()
        statistics = IncrementalPairStatistics(sequences)

        while spec.vocab_size < self.vocab_size:
            best_pair = statistics.best_pair()
            if best_pair is None:
                break

            left, right = best_pair
            if left + right in spec.vocab._token_to_id:
                raise ValueError(
                    f"Merged token already exists: {left + right!r}"
                )

            spec.add_merge(left, right)
            statistics.merge_best()

        return spec

    def train_from_file(
        self,
        input_path: str | Path,
        num_processes: int = 1,
    ) -> TokenizerSpec:
        """
        从语料文件读取数据并训练 BPE。
        """
        reader = CorpusReader(
            input_path=input_path,
            special_tokens=self.special_tokens,
            num_processes=num_processes,
        )

        sequences = dict(reader.collect_frequencies())

        return self.train_from_sequences(sequences)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Iterable[str] = (),
    num_processes: int = 1,
) -> TokenizerSpec:
    """
    训练 byte-level BPE tokenizer 的便捷入口。
    """
    trainer = BPETrainer(
        vocab_size=vocab_size,
        special_tokens=tuple(special_tokens),
    )

    return trainer.train_from_file(input_path, num_processes=num_processes)
