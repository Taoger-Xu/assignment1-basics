"""Tokenize TinyStories into compact, memory-mapped-ready NumPy arrays.

The fast Rust BPE implementation in ``tiktoken`` is used only as an execution
backend. Its vocabulary and merge ranks are built from our trained tokenizer
JSON, and a rank-to-ID map preserves this project's token IDs exactly.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import tempfile
import time
from pathlib import Path

import numpy as np
import tiktoken

from cs336_basics.tokenizer.corpus import GPT2_PATTERN
from cs336_basics.tokenizer.serialization import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tinystories_tokenizer_10k.json"),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        default=[Path("data/TinyStoriesV2-GPT4-train.txt.1")],
    )
    parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/tokenized/tinystories_10k")
    )
    parser.add_argument("--chunk-chars", type=int, default=1 << 20)
    return parser.parse_args()


def build_fast_encoder(tokenizer_path: Path):
    """Return a fast encoder and mapping from its BPE ranks to project IDs."""
    spec = load_spec(tokenizer_path)
    special_bytes = {token.encode("utf-8") for token in spec.special_tokens}
    regular_tokens = sorted(
        (
            (token_id, token_bytes)
            for token_id, token_bytes in spec.vocab.items()
            if token_bytes not in special_bytes
        ),
        key=lambda item: item[0],
    )
    mergeable_ranks = {
        token_bytes: rank
        for rank, (_, token_bytes) in enumerate(regular_tokens)
    }
    # Give special tokens temporary IDs outside the ordinary BPE ranks; the
    # output mapping translates those IDs back to this tokenizer's IDs.
    special_tokens = {
        token: len(regular_tokens) + index
        for index, token in enumerate(spec.special_tokens)
    }
    encoder = tiktoken.Encoding(
        name=f"cs336-{tokenizer_path.stem}",
        pat_str=GPT2_PATTERN.pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    rank_to_token_id = np.asarray(
        [token_id for token_id, _ in regular_tokens]
        + [spec.special_token_id(token) for token in spec.special_tokens],
        dtype=np.uint16,
    )
    return spec, encoder, rank_to_token_id


def _encode_text(text: str, encoder, rank_to_token_id: np.ndarray) -> np.ndarray:
    ranks = encoder.encode(text, allowed_special="all")
    return rank_to_token_id[np.asarray(ranks, dtype=np.int32)]


def tokenize_to_raw(
    paths: list[Path],
    raw_path: Path,
    encoder,
    rank_to_token_id: np.ndarray,
    special_token: str,
    special_id: int,
    chunk_chars: int,
) -> tuple[int, int]:
    """Stream text files to a uint16 raw file; return token and document counts."""
    token_count = 0
    document_count = 0
    pending = ""
    with raw_path.open("wb") as output:
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                while chunk := source.read(chunk_chars):
                    pending += chunk
                    boundary = pending.rfind(special_token)
                    if boundary >= 0:
                        end = boundary + len(special_token)
                        complete_text = pending[:end]
                        ids = _encode_text(complete_text, encoder, rank_to_token_id)
                        ids.tofile(output)
                        token_count += ids.size
                        document_count += complete_text.count(special_token)
                        pending = pending[end:]

            # Input files can end midway through a document. Treat each shard
            # boundary as an end-of-document boundary before reading the next.
            if pending:
                ids = _encode_text(pending, encoder, rank_to_token_id)
                ids.tofile(output)
                np.asarray([special_id], dtype=np.uint16).tofile(output)
                token_count += ids.size + 1
                document_count += 1
            pending = ""

    return token_count, document_count


def is_prefix_file(smaller: Path, larger: Path) -> bool:
    """Detect an incomplete download alongside its complete copy."""
    if smaller.stat().st_size >= larger.stat().st_size:
        return False
    with smaller.open("rb") as left, larger.open("rb") as right:
        while chunk := left.read(8 << 20):
            if chunk != right.read(len(chunk)):
                return False
    return True


def write_npy(raw_path: Path, output_path: Path, token_count: int) -> None:
    source = np.memmap(raw_path, mode="r", dtype=np.uint16, shape=(token_count,))
    target = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint16, shape=(token_count,)
    )
    copy_chunk = 8_000_000
    for start in range(0, token_count, copy_chunk):
        target[start : start + copy_chunk] = source[start : start + copy_chunk]
    target.flush()
    del target
    del source


def prepare_split(
    name: str,
    paths: list[Path],
    output_path: Path,
    encoder,
    rank_to_token_id: np.ndarray,
    special_token: str,
    special_id: int,
    chunk_chars: int,
) -> None:
    if not paths:
        raise ValueError(f"No input files supplied for {name}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for first, second in combinations(paths, 2):
        if first.resolve() == second.resolve():
            raise ValueError(f"Duplicate input file: {first}")
        if is_prefix_file(first, second) or is_prefix_file(second, first):
            raise ValueError(
                f"One input is an incomplete prefix of another: {first}, {second}"
            )

    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        prefix=f".{name}-", suffix=".u16", dir=output_path.parent, delete=False
    ) as raw_file:
        raw_path = Path(raw_file.name)

    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    try:
        token_count, documents = tokenize_to_raw(
            paths,
            raw_path,
            encoder,
            rank_to_token_id,
            special_token,
            special_id,
            chunk_chars,
        )
        write_npy(raw_path, temporary_output, token_count)
        temporary_output.replace(output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)

    source_bytes = sum(path.stat().st_size for path in paths)
    elapsed = time.perf_counter() - started
    print(
        f"{name}: {source_bytes:,} bytes -> {token_count:,} tokens "
        f"({source_bytes / token_count:.3f} bytes/token), {documents:,} "
        f"document boundaries, {elapsed:.1f}s; saved {output_path}"
    )


def main() -> None:
    args = parse_args()
    if args.chunk_chars <= 0:
        raise ValueError("chunk-chars must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec, encoder, rank_to_token_id = build_fast_encoder(args.tokenizer)
    if len(spec.special_tokens) != 1:
        raise ValueError("TinyStories tokenizer is expected to have one special token")
    special_token = spec.special_tokens[0]
    special_id = spec.special_token_id(special_token)

    train_paths = args.train_data
    prepare_split(
        "train",
        train_paths,
        args.output_dir / "train.npy",
        encoder,
        rank_to_token_id,
        special_token,
        special_id,
        args.chunk_chars,
    )
    prepare_split(
        "valid",
        [args.valid_data],
        args.output_dir / "valid.npy",
        encoder,
        rank_to_token_id,
        special_token,
        special_id,
        args.chunk_chars,
    )


if __name__ == "__main__":
    main()
