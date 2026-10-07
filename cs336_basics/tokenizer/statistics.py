# cs336_basics/tokenizer/statistics.py

from __future__ import annotations

from collections import Counter
import heapq
from typing import Iterable, Mapping


ByteToken = bytes
TokenSequence = tuple[ByteToken, ...]
Pair = tuple[ByteToken, ByteToken]


class _DescendingPair:
    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: "_DescendingPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DescendingPair) and self.pair == other.pair


def _reverse_pair(pair: Pair) -> _DescendingPair:
    """使 heap 在频次相同时优先选择字典序更大的 pair。"""
    return _DescendingPair(pair)


class IncrementalPairStatistics:
    """维护 BPE 训练所需的 pair 计数、出现索引和最大堆。"""

    def __init__(self, sequences: dict[TokenSequence, int]) -> None:
        self.sequences = dict(sequences)
        self.pair_counts: Counter[Pair] = count_pair_frequencies(self.sequences)
        self.pair_sequences: dict[Pair, set[TokenSequence]] = {}
        for sequence in self.sequences:
            for index in range(len(sequence) - 1):
                pair = (sequence[index], sequence[index + 1])
                self.pair_sequences.setdefault(pair, set()).add(sequence)
        self.heap: list[tuple[int, tuple[bytes, bytes], Pair]] = [
            (-count, _reverse_pair(pair), pair)
            for pair, count in self.pair_counts.items()
        ]
        heapq.heapify(self.heap)

    def best_pair(self) -> Pair | None:
        """从堆中取出当前有效的最高频 pair。"""
        while self.heap:
            neg_count, _, pair = self.heap[0]
            if -neg_count > 0 and self.pair_counts.get(pair, 0) == -neg_count:
                return pair
            heapq.heappop(self.heap)
        return None

    def merge_best(self) -> Pair | None:
        """合并当前最高频 pair，并局部更新统计。"""
        pair = self.best_pair()
        if pair is None:
            return None

        affected = self.pair_sequences.get(pair, set()).copy()
        for sequence in affected:
            frequency = self.sequences.pop(sequence)
            self._remove_sequence(sequence, frequency)
            merged = merge_sequence(sequence, pair)
            self.sequences[merged] = self.sequences.get(merged, 0) + frequency
            self._add_sequence(merged, frequency)

        return pair

    def _remove_sequence(self, sequence: TokenSequence, frequency: int) -> None:
        for index in range(len(sequence) - 1):
            current = (sequence[index], sequence[index + 1])
            self.pair_counts[current] -= frequency
            self.pair_sequences.get(current, set()).discard(sequence)
            heapq.heappush(
                self.heap,
                (-self.pair_counts[current], _reverse_pair(current), current),
            )

    def _add_sequence(self, sequence: TokenSequence, frequency: int) -> None:
        for index in range(len(sequence) - 1):
            current = (sequence[index], sequence[index + 1])
            self.pair_counts[current] += frequency
            self.pair_sequences.setdefault(current, set()).add(sequence)
            heapq.heappush(
                self.heap,
                (-self.pair_counts[current], _reverse_pair(current), current),
            )


def count_pair_frequencies(
    sequences: Mapping[TokenSequence, int],
) -> Counter[Pair]:
    """
    统计所有 token 序列中的相邻 pair。

    sequences 的格式：

        {
            (b"l", b"o", b"w"): 5,
            (b"l", b"o", b"w", b"e", b"r"): 2,
        }

    value 表示该序列在语料中的出现次数。
    """
    pair_counts: Counter[Pair] = Counter()

    for sequence, sequence_frequency in sequences.items():
        if sequence_frequency <= 0:
            continue

        for index in range(len(sequence) - 1):
            pair = (sequence[index], sequence[index + 1])
            pair_counts[pair] += sequence_frequency

    return pair_counts


def choose_best_pair(
    pair_counts: Mapping[Pair, int],
) -> Pair | None:
    """
    选择下一条 BPE merge。

    优先级：

    1. 出现频率更高；
    2. 频率相同时，选择字典序更大的 pair。

    如果没有任何 pair，返回 None。
    """
    if not pair_counts:
        return None

    return max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], pair),
    )


def merge_sequence(
    sequence: TokenSequence,
    pair: Pair,
) -> TokenSequence:
    """
    将一个序列中所有非重叠的目标 pair 合并。

    例如：

        sequence = (b"a", b"b", b"a", b"b")
        pair = (b"a", b"b")

        result = (b"ab", b"ab")
    """
    left, right = pair
    merged_token = left + right

    result: list[ByteToken] = []
    index = 0

    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left
            and sequence[index + 1] == right
        ):
            result.append(merged_token)
            index += 2
        else:
            result.append(sequence[index])
            index += 1

    return tuple(result)


def apply_merge_to_sequences(
    sequences: Mapping[TokenSequence, int],
    pair: Pair,
) -> dict[TokenSequence, int]:
    """
    将指定 pair 应用到所有序列，并重新聚合相同结果的频次。
    """
    merged_sequences: Counter[TokenSequence] = Counter()

    for sequence, frequency in sequences.items():
        new_sequence = merge_sequence(sequence, pair)
        merged_sequences[new_sequence] += frequency

    return dict(merged_sequences)


def pair_exists(
    sequences: Iterable[TokenSequence],
    pair: Pair,
) -> bool:
    """判断指定 pair 是否至少出现在一个序列中。"""
    left, right = pair

    for sequence in sequences:
        for index in range(len(sequence) - 1):
            if sequence[index] == left and sequence[index + 1] == right:
                return True

    return False
