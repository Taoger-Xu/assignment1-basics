# cs336_basics/tokenizer/corpus.py

from __future__ import annotations

from collections import Counter
from multiprocessing import Pool
from pathlib import Path
try:
    import regex as re
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    import re
from typing import Iterable


if re.__name__ == "regex":
    _GPT2_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
else:
    _GPT2_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+"""
    )

GPT2_PATTERN = re.compile(_GPT2_PATTERN, re.UNICODE)


def _count_chunk(args: tuple[str, tuple[str, ...]]) -> Counter[tuple[bytes, ...]]:
    text, special_tokens = args
    reader = CorpusReader("", special_tokens)
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in reader.split_special_tokens(text):
        for pretoken in reader.pretokenize(chunk):
            if pretoken:
                counts[reader.encode_pretoken(pretoken)] += 1
    return counts


class CorpusReader:
    """读取语料并生成带频次的 UTF-8 字节序列。
    read_text
        ↓
    split_special_tokens
        ↓
    pretokenize
        ↓
    encode_pretoken
        ↓
    iter_sequences
        ↓
    collect_frequencies
    """

    def __init__(
        self,
        input_path: str | Path,
        special_tokens: Iterable[str] | None = None,
        num_processes: int = 1,
    ) -> None:
        self.input_path = Path(input_path)
        self.special_tokens = tuple(special_tokens or ())
        if num_processes < 1:
            raise ValueError("num_processes must be positive")
        self.num_processes = num_processes

        escaped = [re.escape(token) for token in self.special_tokens]
        if escaped:
            self._special_pattern = re.compile("|".join(escaped))
        else:
            self._special_pattern = None

    def read_text(self) -> str:
        """读取完整语料。"""
        return self.input_path.read_text(encoding="utf-8")

    def split_special_tokens(self, text: str) -> list[str]:
        """
        按 special token 分隔文档，禁止 merge 跨文档边界
        hello<|endoftext|>world --> ["hello", "world"]
        special token 本身不会作为普通文本参与 BPE 统计，
        因此普通文本不会跨越 special token 边界。
        """
        if self._special_pattern is None:
            return [text]

        chunks: list[str] = []
        start = 0

        # 使用正则搜索每个 special token；
        for match in self._special_pattern.finditer(text):
            if match.start() > start:
                chunks.append(text[start : match.start()])

            start = match.end()

        if start < len(text):
            chunks.append(text[start:])

        return chunks

    def pretokenize(self, text: str) -> list[str]:
        """将普通文本切分为 pre-token。
        "the cat ate" --> ["the", " cat", " ate"]
        """
        return [match.group(0) for match in GPT2_PATTERN.finditer(text)]

    @staticmethod
    def encode_pretoken(pretoken: str) -> tuple[bytes, ...]:
        """
        将一个 pre-token 转换成单字节 token 序列。

        例如：
            "low" -> (b"l", b"o", b"w")
        """
        encoded = pretoken.encode("utf-8")
        return tuple(bytes([value]) for value in encoded)

    def iter_sequences(self) -> Iterable[tuple[bytes, ...]]:
        """为每一个chunk逐个产生 pre-token 对应的字节序列。"""
        text = self.read_text()

        for chunk in self.split_special_tokens(text):
            if not chunk:
                continue

            for pretoken in self.pretokenize(chunk):
                if pretoken:
                    yield self.encode_pretoken(pretoken)

    def collect_frequencies(self) -> Counter[tuple[bytes, ...]]:
        """统计每个字节序列的出现次数。
        pre-token 的 UTF-8 字节序列 + 该序列出现的次数
        比如low low lower要返回
        (b"l", b"o", b"w"): 2
        (b"l", b"o", b"w", b"e", b"r"): 1
        """
        if self.num_processes > 1:
            return self.collect_frequencies_parallel()

        frequencies: Counter[tuple[bytes, ...]] = Counter()

        for sequence in self.iter_sequences():
            frequencies[sequence] += 1

        return frequencies

    def _make_chunks(self, text: str, num_chunks: int) -> list[str]:
        """在 special token 边界切分近似等大的文本块。"""
        if num_chunks <= 1 or not text:
            return [text]

        boundaries = [0]
        if self._special_pattern is not None:
            boundaries.extend(
                match.start() for match in self._special_pattern.finditer(text)
            )
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        target = max(1, len(text) // num_chunks)
        chunks: list[str] = []
        start = 0
        for boundary in boundaries[1:]:
            if boundary - start >= target:
                chunks.append(text[start:boundary])
                start = boundary
        if start < len(text):
            chunks.append(text[start:])
        return chunks or [text]

    def collect_frequencies_parallel(
        self,
        num_processes: int | None = None,
    ) -> Counter[tuple[bytes, ...]]:
        """并行统计 pre-token 频次，并在主进程合并 Counter。"""
        workers = num_processes or self.num_processes
        if workers <= 1:
            return self.collect_frequencies()

        chunks = self._make_chunks(self.read_text(), workers)
        args = [(chunk, self.special_tokens) for chunk in chunks]
        frequencies: Counter[tuple[bytes, ...]] = Counter()
        with Pool(processes=min(workers, len(args))) as pool:
            for partial in pool.map(_count_chunk, args):
                frequencies.update(partial)
        return frequencies


def load_corpus_frequencies(
    input_path: str | Path,
    special_tokens: Iterable[str] | None = None,
) -> Counter[tuple[bytes, ...]]:
    """便捷函数：读取语料并返回 pre-token 频次。"""
    reader = CorpusReader(
        input_path=input_path,
        special_tokens=special_tokens,
    )
    return reader.collect_frequencies()
