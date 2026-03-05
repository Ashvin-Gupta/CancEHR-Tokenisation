from __future__ import annotations

from pathlib import Path
from typing import List

from src.tokenization.algorithms.hf_parquet_bpe_trainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLUMN,
    DEFAULT_DATA_ROOT,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOCAB_SIZES,
    run_vocab_experiments,
)


def main() -> None:
    """
    CLI entrypoint for training/evaluating BPE tokenizers on parquet text columns.

    Example:
        python -m src.tokenization.train_parquet_bpe \\
            --data_root translated \\
            --column text_sequence_with_time \\
            --vocab_sizes 4000 8000 \\
            --output_dir artifacts/tokenizers/experiments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate BPE tokenizers on translated parquet text."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory containing train/ and test/ parquet files.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=DEFAULT_COLUMN,
        help="Name of the text column in the parquet files.",
    )
    parser.add_argument(
        "--vocab_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_VOCAB_SIZES,
        help="One or more vocab sizes to train/evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write tokenizer artifacts and vocab.csv files.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help="Minimum token frequency for inclusion in the vocab.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used during evaluation.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    run_vocab_experiments(
        data_root=data_root,
        column=args.column,
        vocab_sizes=list(args.vocab_sizes),
        output_dir=output_dir,
        min_frequency=args.min_frequency,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()