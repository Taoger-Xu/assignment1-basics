# cs336_basics/tokenizer/decode.py

from __future__ import annotations

from collections.abc import Iterable

from .specification import TokenizerSpec


class Decoder:
    """使用 TokenizerSpec 将 token ID 序列还原为文本。"""

    def __init__(self, spec: TokenizerSpec) -> None:
        self.spec = spec

    def decode_bytes(
        self,
        token_ids: Iterable[int],
    ) -> bytes:
        """
        将 token ID 序列拼接为原始 bytes。

        如果 token ID 不存在，TokenizerSpec 会抛出 KeyError。
        """
        chunks: list[bytes] = []

        for token_id in token_ids:
            token_bytes = self.spec.token_bytes(token_id)
            chunks.append(token_bytes)

        return b"".join(chunks)

    def decode(
        self,
        token_ids: Iterable[int],
    ) -> str:
        """
        将 token ID 序列解码为文本。

        errors="replace" 会把非法 UTF-8 字节替换为 Unicode
        replacement character：�。
        """
        raw_bytes = self.decode_bytes(token_ids)
        return raw_bytes.decode("utf-8", errors="replace")


def decode(
    token_ids: Iterable[int],
    spec: TokenizerSpec,
) -> str:
    """便捷函数。"""
    return Decoder(spec).decode(token_ids)