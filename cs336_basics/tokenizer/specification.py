# cs336_basics/tokenizer/specification.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


ByteToken = bytes
TokenId = int
Pair = tuple[ByteToken, ByteToken]


@dataclass(frozen=True)
class MergeRule:
    """
    表示一条 BPE merge 规则。

    例如：
        (b"s", b"t") -> b"st"
    """

    left: ByteToken
    right: ByteToken

    @property
    def merged(self) -> ByteToken:
        """返回合并后的 token 字节串。"""
        return self.left + self.right

    @property
    def pair(self) -> Pair:
        """返回原始 pair。"""
        return self.left, self.right

class Vocabulary:
    """维护 token bytes 与 token ID 的双向映射，hash表。"""

    def __init__(self) -> None:
        self._id_to_token: dict[TokenId, ByteToken] = {}
        self._token_to_id: dict[ByteToken, TokenId] = {}
        self._special_token_to_id: dict[str, TokenId] = {}

    @property
    def size(self) -> int:
        """当前词表大小。"""
        return len(self._id_to_token)

    def add_token(
        self,
        token: ByteToken,
        token_id: int | None = None,
    ) -> TokenId:
        """
        向词表加入普通 byte token 或 merge token。

        如果 token 已经存在，返回已有 ID。
        如果未指定 token_id，则使用当前词表末尾位置。
        """
        if token in self._token_to_id:
            return self._token_to_id[token]

        if token_id is None:
            token_id = self.size

        if token_id in self._id_to_token:
            raise ValueError(f"Token ID already exists: {token_id}")

        if token in self._token_to_id:
            raise ValueError(f"Token already exists: {token!r}")

        self._id_to_token[token_id] = token
        self._token_to_id[token] = token_id

        return token_id

    def add_special_token(
        self,
        text: str,
        token_id: int | None = None,
    ) -> TokenId:
        """
        添加 special token。

        special token 的字符串形式用于文本匹配，
        其 UTF-8 字节形式用于 vocabulary 存储。
        """
        if text in self._special_token_to_id:
            return self._special_token_to_id[text]

        token_bytes = text.encode("utf-8")
        assigned_id = self.add_token(token_bytes, token_id)

        self._special_token_to_id[text] = assigned_id
        return assigned_id

    def token_to_id(self, token: ByteToken) -> TokenId:
        """根据 token bytes 查找 token ID。"""
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"Unknown token: {token!r}") from exc

    def id_to_token(self, token_id: TokenId) -> ByteToken:
        """根据 token ID 查找 token bytes。"""
        try:
            return self._id_to_token[token_id]
        except KeyError as exc:
            raise KeyError(f"Unknown token ID: {token_id}") from exc

    def special_token_to_id(self, text: str) -> TokenId:
        """根据 special token 字符串查找 ID。"""
        try:
            return self._special_token_to_id[text]
        except KeyError as exc:
            raise KeyError(f"Unknown special token: {text}") from exc

    def contains(self, token: ByteToken) -> bool:
        """判断普通 token 是否在词表中。"""
        return token in self._token_to_id

    def items(self):
        """按 token ID 顺序返回 vocabulary 内容。"""
        return sorted(self._id_to_token.items())

    def to_dict(self) -> dict[int, bytes]:
        """导出 ID 到 bytes 的映射。"""
        return dict(self._id_to_token)

    @classmethod
    def from_dict(cls, vocab: dict[int, bytes]) -> "Vocabulary":
        """从已保存的 vocabulary 构造对象。"""
        result = cls()

        for token_id, token in sorted(vocab.items()):
            result.add_token(token, token_id)

        return result

@dataclass
class TokenizerSpec:
    """
    tokenizer 的完整静态配置。

    它包含：
    - vocabulary；
    - 按创建顺序排列的 merge rules；
    - special token 配置。
    """

    vocab: Vocabulary
    merges: list[MergeRule] = field(default_factory=list)
    special_tokens: tuple[str, ...] = ()

    def add_merge(self, left: bytes, right: bytes) -> int:
        """
        记录一条 merge，并把生成的 token 加入 vocabulary。

        返回新 token 的 ID。
        """
        rule = MergeRule(left=left, right=right)

        if rule in self.merges:
            merged_id = self.vocab.token_to_id(rule.merged)
            return merged_id

        self.merges.append(rule)
        return self.vocab.add_token(rule.merged)

    def merge_pairs(self) -> list[Pair]:
        """返回按训练顺序排列的原始 pair。"""
        return [rule.pair for rule in self.merges]

    def token_id(self, token: bytes) -> int:
        """查找普通 token ID。"""
        return self.vocab.token_to_id(token)

    def token_bytes(self, token_id: int) -> bytes:
        """查找 token bytes。"""
        return self.vocab.id_to_token(token_id)

    def special_token_id(self, token: str) -> int:
        """查找 special token ID。"""
        return self.vocab.special_token_to_id(token)

    @property
    def vocab_size(self) -> int:
        """返回当前词表大小。"""
        return self.vocab.size

    def to_dict(self) -> dict:
        """导出为适合序列化的普通 Python 对象。"""
        return {
            "vocab": self.vocab.to_dict(),
            "merges": [
                {
                    "left": rule.left,
                    "right": rule.right,
                }
                for rule in self.merges
            ],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerSpec":
        """从序列化数据恢复 tokenizer 状态。"""
        vocab = Vocabulary.from_dict(data["vocab"])

        merges = [
            MergeRule(
                left=merge["left"],
                right=merge["right"],
            )
            for merge in data["merges"]
        ]

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=tuple(data["special_tokens"]),
        )

def create_byte_vocabulary(
    special_tokens: Iterable[str] = (),
) -> TokenizerSpec:
    """
    创建初始 byte-level vocabulary。

    词表首先加入 256 个单字节 token，
    再加入 special tokens。
    """
    vocab = Vocabulary()

    for value in range(256):
        vocab.add_token(bytes([value]))

    special_tokens = tuple(special_tokens)

    for token in special_tokens:
        vocab.add_special_token(token)

    return TokenizerSpec(
        vocab=vocab,
        merges=[],
        special_tokens=special_tokens,
    )