# cs336_basics/tokenizer/encode.py

from __future__ import annotations

from typing import Iterable, Iterator

from .corpus import CorpusReader
from .specification import TokenizerSpec


class Encoder:
    """使用已经训练好的 BPE 规则将文本编码为 token ID。"""

    def __init__(self, spec: TokenizerSpec) -> None:
        self.spec = spec

        self._special_tokens = set(spec.special_tokens)
        self._special_token_pattern = self._build_special_pattern()

    def _build_special_pattern(self):
        """
        构造 special token 的匹配模式。

        使用 CorpusReader 中的切分逻辑所需的正则表达式。
        """
        if not self._special_tokens:
            return None

        import re

        escaped = [
            re.escape(token)
            for token in sorted(
                self._special_tokens,
                key=len,
                reverse=True,
            )
        ]

        return re.compile("|".join(escaped))

    def _split_text(
        self,
        text: str,
    ) -> list[tuple[bool, str]]:
        """
        将文本切成普通文本和 special token。

        返回值中的 bool：
        - True：special token；
        - False：普通文本。
        """
        if self._special_token_pattern is None:
            return [(False, text)]

        pieces: list[tuple[bool, str]] = []
        start = 0

        for match in self._special_token_pattern.finditer(text):
            if match.start() > start:
                pieces.append(
                    (False, text[start : match.start()])
                )

            pieces.append((True, match.group(0)))
            start = match.end()

        if start < len(text):
            pieces.append((False, text[start:]))

        return pieces

    @staticmethod
    def _merge_pair(
        sequence: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        将当前序列中所有非重叠的目标 pair 合并一次。
        """
        left, right = pair
        merged = left + right

        result: list[bytes] = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1

        return tuple(result)

    def _apply_merges(
        self,
        sequence: tuple[bytes, ...],
    ) -> tuple[bytes, ...]:
        """
        按训练时的顺序应用所有 merge。
        """
        current = sequence

        for rule in self.spec.merges:
            current = self._merge_pair(
                current,
                rule.pair,
            )

        return current

    def _encode_regular_text(self, text: str) -> list[int]:
        """
        编码不包含 special token 的普通文本。
        """
        if not text:
            return []

        reader = CorpusReader(
            input_path="",
            special_tokens=(),
        )

        pretokenized = reader.pretokenize(text)
        token_ids: list[int] = []

        for pretoken in pretokenized:
            initial_sequence = reader.encode_pretoken(pretoken)
            merged_sequence = self._apply_merges(initial_sequence)

            for token in merged_sequence:
                token_ids.append(self.spec.token_id(token))

        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 序列。
        """
        token_ids: list[int] = []

        for is_special, piece in self._split_text(text):
            if not piece:
                continue

            if is_special:
                token_ids.append(
                    self.spec.special_token_id(piece)
                )
            else:
                token_ids.extend(
                    self._encode_regular_text(piece)
                )

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        逐段编码可迭代文本。
        """
        for text in iterable:
            yield from self.encode(text)


def encode(
    text: str,
    spec: TokenizerSpec,
) -> list[int]:
    """便捷函数。"""
    return Encoder(spec).encode(text)