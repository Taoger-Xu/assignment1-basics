#!/usr/bin/env python3
"""Train and report the 10K TinyStories byte-level BPE tokenizer."""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from cs336_basics.tokenizer.serialization import save_spec
from cs336_basics.tokenizer.train import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    spec = train_bpe(
        args.input,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=args.num_processes,
    )
    elapsed = time.perf_counter() - start
    save_spec(spec, args.output)

    longest_id, longest_token = max(
        spec.vocab.items(), key=lambda item: len(item[1])
    )
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"vocab_size={spec.vocab_size}")
    print(f"merges={len(spec.merges)}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"peak_rss_mb={peak_mb:.1f}")
    print(f"longest_token_id={longest_id}")
    print(f"longest_token={longest_token!r}")
    print(f"longest_token_bytes={len(longest_token)}")
    print(f"saved_to={args.output}")


if __name__ == "__main__":
    main()
