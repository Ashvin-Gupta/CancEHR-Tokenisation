from __future__ import annotations

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from src.postprocessing.bpe_encoding import (
    encode_files_to_pickles,
    find_parquet_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode parquet EHR text with a trained BPE tokenizer into .pkl files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing train/ and test/ parquet files.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Directory of the trained tokenizer (e.g. artifacts/tokenizers/mimic_ed_bytebpe_10k).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="text_sequence_with_time",
        help="Name of the text column in the parquet files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to write pickled timelines into.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on total number of rows to encode.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    parquet_files = find_parquet_files(data_root)
    if not parquet_files:
        raise ValueError(f"No parquet files found under {data_root}")

    encode_files_to_pickles(
        tokenizer=tokenizer,
        data_root=data_root,
        parquet_files=parquet_files,
        column=args.column,
        output_root=output_root,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()

