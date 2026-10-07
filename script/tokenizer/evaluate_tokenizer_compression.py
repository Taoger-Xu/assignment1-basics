#!/usr/bin/env python3
"""Evaluate bytes/token compression on sampled documents."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator
from pathlib import Path

from cs336_basics.tokenizer.api import Tokenizer
from cs336_basics.tokenizer.serialization import load_spec


ROOT = Path(__file__).resolve().parents[2]


def iter_documents(path: Path, delimiter: str) -> Iterator[str]:
    """Read bounded chunks, preserving UTF-8 and delimiters across boundaries."""
    if not delimiter:
        raise ValueError("delimiter must not be empty")
    pending = ""
    with path.open(encoding="utf-8", newline="") as stream:
        while chunk := stream.read(1024 * 1024):
            parts = (pending + chunk).split(delimiter)
            pending = parts.pop()
            for document in parts:
                if document.strip():
                    yield document
    if pending.strip():
        yield pending


def sample_documents(
    path: Path,
    delimiter: str,
    count: int,
    seed: int,
) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    documents: list[str] = []
    seen = 0
    for seen, document in enumerate(iter_documents(path, delimiter), start=1):
        if seen <= count:
            documents.append(document)
        else:
            index = rng.randrange(seen)
            if index < count:
                documents[index] = document
    if seen < count:
        raise ValueError(
            f"{path} contains only {seen} documents; "
            f"cannot sample {count}"
        )
    return documents


def evaluate(
    corpus_path: Path,
    tokenizer_path: Path,
    delimiter: str,
    count: int,
    seed: int,
) -> dict[str, float | int]:
    documents = sample_documents(corpus_path, delimiter, count, seed)
    return evaluate_documents(documents, tokenizer_path)


def evaluate_documents(
    documents: list[str], tokenizer_path: Path,
) -> dict[str, float | int]:
    """Encode an existing sample so comparisons use exactly the same text."""
    spec = load_spec(tokenizer_path)
    tokenizer = Tokenizer(
        spec.vocab.to_dict(),
        spec.merge_pairs(),
        list(spec.special_tokens),
    )
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)
    total_tokens = 0
    for document in documents:
        ids = tokenizer.encode(document)
        if tokenizer.decode(ids) != document:
            raise ValueError(f"Tokenizer round-trip failed: {tokenizer_path}")
        total_tokens += len(ids)
    if total_tokens == 0:
        raise ValueError("Sampled documents produced no tokens")
    return {
        "documents": len(documents),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


def print_result(name: str, result: dict[str, float | int]) -> None:
    print(f"{name}:")
    print(f"  documents       = {result['documents']}")
    print(f"  bytes           = {result['bytes']}")
    print(f"  tokens          = {result['tokens']}")
    print(f"  bytes/token     = {result['bytes_per_token']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinystories-corpus", type=Path, default=ROOT / "data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--tinystories-tokenizer", type=Path, default=ROOT / "artifacts/tinystories_tokenizer_10k.json")
    parser.add_argument("--openwebtext-corpus", type=Path, default=ROOT / "data/owt_valid.txt")
    parser.add_argument("--openwebtext-tokenizer", type=Path, default=ROOT / "artifacts/owt_tokenizer_32k.json")
    parser.add_argument("--delimiter", default="<|endoftext|>")
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    tiny = evaluate(
        args.tinystories_corpus,
        args.tinystories_tokenizer,
        args.delimiter,
        args.num_documents,
        args.seed,
    )
    owt_documents = sample_documents(
        args.openwebtext_corpus,
        args.delimiter,
        args.num_documents,
        args.seed,
    )
    owt = evaluate_documents(owt_documents, args.openwebtext_tokenizer)
    cross = evaluate_documents(owt_documents, args.tinystories_tokenizer)
    token_change = (cross["tokens"] / owt["tokens"] - 1) * 100

    print_result("TinyStories tokenizer on TinyStories", tiny)
    print_result("OpenWebText tokenizer on OpenWebText", owt)
    print_result("TinyStories tokenizer on the same OpenWebText sample", cross)
    print(f"  token count change vs OpenWebText tokenizer = {token_change:+.2f}%")
    print("\nAssignment response (UTF-8 bytes / total tokens; document delimiters excluded):")
    print(
        f"在随机抽取的 {args.num_documents} 篇 TinyStories 文档上，10K tokenizer 的压缩率为 "
        f"{tiny['bytes_per_token']:.4f} 字节/token。"
        f"在随机抽取的 {args.num_documents} 篇 OpenWebText 文档上，32K tokenizer 的压缩率为 "
        f"{owt['bytes_per_token']:.4f} 字节/token。"
    )
    print("\n跨语料评测结论（同一批 OpenWebText 文档）：")
    direction = "增加" if token_change >= 0 else "减少"
    comparison = "更细，压缩效率更低" if token_change > 0 else "更粗，压缩效率更高"
    if token_change == 0:
        comparison = "粒度在总 token 数上相同，压缩率相同"
    print(
        f"对同一批 {args.num_documents} 篇 OpenWebText 文档，TinyStories 10K tokenizer 的压缩率为 "
        f"{cross['bytes_per_token']:.4f} 字节/token，而 OpenWebText 32K tokenizer 为 "
        f"{owt['bytes_per_token']:.4f} 字节/token。"
        f"TinyStories tokenizer 产生的 token 数{direction}了 {abs(token_change):.2f}%，"
        f"表明这批文本的切分{comparison}；两者编码后均能完整还原原文。"
    )


if __name__ == "__main__":
    main()
