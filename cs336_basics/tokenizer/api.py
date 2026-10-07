from __future__ import annotations

from collections.abc import Iterable

from .decode import Decoder
from .encode import Encoder
from .specification import MergeRule, TokenizerSpec, Vocabulary


class Tokenizer:
    """作业接口所需的 vocabulary/merges tokenizer 门面。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        vocabulary = Vocabulary.from_dict(vocab)
        specials = tuple(special_tokens or ())

        for token in specials:
            token_bytes = token.encode("utf-8")
            token_id = vocabulary.token_to_id(token_bytes)
            vocabulary._special_token_to_id[token] = token_id

        spec = TokenizerSpec(
            vocab=vocabulary,
            merges=[MergeRule(left, right) for left, right in merges],
            special_tokens=specials,
        )

        self._encoder = Encoder(spec)
        self._decoder = Decoder(spec)

    def encode(self, text: str) -> list[int]:
        return self._encoder.encode(text)

    def encode_iterable(self, iterable: Iterable[str]):
        return self._encoder.encode_iterable(iterable)

    def decode(self, ids: Iterable[int]) -> str:
        return self._decoder.decode(ids)
