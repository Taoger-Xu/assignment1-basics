# cs336_basics/tokenizer/serialization.py

from __future__ import annotations

import base64
import json
from pathlib import Path

from .specification import MergeRule, TokenizerSpec, Vocabulary


def _encode_bytes(value: bytes) -> str:
    """将 bytes 编码为可写入 JSON 的 Base64 字符串。"""
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: str) -> bytes:
    """将 Base64 字符串恢复为 bytes。"""
    return base64.b64decode(value.encode("ascii"))


def spec_to_dict(spec: TokenizerSpec) -> dict:
    """
    将 TokenizerSpec 转换为 JSON 兼容的字典。
    """
    vocab = [
        {
            "id": token_id,
            "token": _encode_bytes(token),
        }
        for token_id, token in spec.vocab.items()
    ]

    merges = [
        {
            "left": _encode_bytes(rule.left),
            "right": _encode_bytes(rule.right),
        }
        for rule in spec.merges
    ]

    return {
        "format_version": 1,
        "vocab": vocab,
        "merges": merges,
        "special_tokens": list(spec.special_tokens),
    }


def spec_from_dict(data: dict) -> TokenizerSpec:
    """
    从 JSON 兼容字典恢复 TokenizerSpec。
    """
    vocab = Vocabulary()

    for item in sorted(data["vocab"], key=lambda item: item["id"]):
        token_id = int(item["id"])
        token = _decode_bytes(item["token"])
        vocab.add_token(token, token_id)

    for special_token in data.get("special_tokens", []):
        token_id = vocab.token_to_id(
            special_token.encode("utf-8")
        )
        vocab._special_token_to_id[special_token] = token_id

    merges = [
        MergeRule(
            left=_decode_bytes(item["left"]),
            right=_decode_bytes(item["right"]),
        )
        for item in data["merges"]
    ]

    return TokenizerSpec(
        vocab=vocab,
        merges=merges,
        special_tokens=tuple(
            data.get("special_tokens", [])
        ),
    )


def save_spec(
    spec: TokenizerSpec,
    path: str | Path,
) -> None:
    """
    将 tokenizer 状态保存为 JSON 文件。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = spec_to_dict(spec)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_spec(
    path: str | Path,
) -> TokenizerSpec:
    """
    从 JSON 文件加载 tokenizer 状态。
    """
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return spec_from_dict(data)


class TokenizerSerializer:
    """
    面向对象的序列化接口。
    """

    @staticmethod
    def save(
        spec: TokenizerSpec,
        path: str | Path,
    ) -> None:
        save_spec(spec, path)

    @staticmethod
    def load(
        path: str | Path,
    ) -> TokenizerSpec:
        return load_spec(path)