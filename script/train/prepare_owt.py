"""Encode OpenWebText train/validation text with the project's 32K BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from script.train.prepare_tinystories import build_fast_encoder, prepare_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/owt_tokenizer_32k.json"))
    parser.add_argument("--train-data", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--valid-data", type=Path, default=Path("data/owt_valid.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokenized/owt_32k"))
    parser.add_argument("--chunk-chars", type=int, default=1 << 20)
    args = parser.parse_args()
    if args.chunk_chars <= 0:
        raise ValueError("chunk-chars must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec, encoder, rank_to_token_id = build_fast_encoder(args.tokenizer)
    if tuple(spec.special_tokens) != ("<|endoftext|>",):
        raise ValueError("Expected one <|endoftext|> special token")

    special_token = "<|endoftext|>"
    special_id = spec.special_token_id(special_token)
    prepare_split(
        "train", [args.train_data], args.output_dir / "train.npy", encoder,
        rank_to_token_id, special_token, special_id, args.chunk_chars,
    )
    prepare_split(
        "valid", [args.valid_data], args.output_dir / "valid.npy", encoder,
        rank_to_token_id, special_token, special_id, args.chunk_chars,
    )


if __name__ == "__main__":
    main()
