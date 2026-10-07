#!/usr/bin/env python3
"""Measure encode-only throughput and extrapolate to 825 GB of text."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cs336_basics.tokenizer.api import Tokenizer
from cs336_basics.tokenizer.serialization import load_spec
from script.tokenizer.evaluate_tokenizer_compression import ROOT, sample_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/owt_valid.txt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/owt_tokenizer_32k.json")
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--delimiter", default="<|endoftext|>")
    parser.add_argument("--pile-gb", type=float, default=825.0, help="Decimal GB (10^9 bytes)")
    args = parser.parse_args()

    if args.pile_gb <= 0:
        parser.error("--pile-gb must be positive")

    documents = sample_documents(args.corpus, args.delimiter, args.num_documents, args.seed)
    spec = load_spec(args.tokenizer)
    tokenizer = Tokenizer(spec.vocab.to_dict(), spec.merge_pairs(), list(spec.special_tokens))
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)

    start = time.perf_counter()
    total_tokens = sum(len(tokenizer.encode(document)) for document in documents)
    elapsed = time.perf_counter() - start
    throughput = total_bytes / elapsed
    pile_seconds = args.pile_gb * 1_000_000_000 / throughput

    print(f"documents={len(documents)}")
    print(f"utf8_bytes={total_bytes}")
    print(f"tokens={total_tokens}")
    print(f"encode_seconds={elapsed:.4f}")
    print(f"throughput_bytes_per_second={throughput:.2f}")
    print(f"estimated_pile_days={pile_seconds / 86400:.2f}")
    print(
        f"在 {len(documents)} 篇 OpenWebText 样本文档上，32K tokenizer 的单进程编码吞吐量约为 "
        f"{throughput:.0f} 字节/秒。按此速度线性估算，编码 825 GB（十进制）的 "
        f"The Pile 约需 {pile_seconds / 86400:.1f} 天；估算只包含编码时间。"
    )


if __name__ == "__main__":
    main()
