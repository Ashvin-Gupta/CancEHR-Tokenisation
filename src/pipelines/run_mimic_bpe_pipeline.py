from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.preprocessing.mimic_direct_translator import MIMICDirectTranslator
from src.postprocessing.bpe_encoding import (
    encode_files_to_pickles,
    find_parquet_files as find_parquet_files_for_encoding,
)
from src.tokenization.algorithms.hf_parquet_bpe_trainer import run_vocab_experiments


def _maybe_run_translation(cfg: Dict[str, Any]) -> None:
    translation_cfg = cfg.get("translation")
    if not translation_cfg:
        return

    if not translation_cfg.get("enabled", True):
        return

    lookup_dir = Path(translation_cfg["lookup_dir"]).expanduser().resolve()
    meds_root = Path(translation_cfg["meds_root"]).expanduser().resolve()
    output_root = Path(translation_cfg["output_root"]).expanduser().resolve()
    splits: List[str] = translation_cfg.get("splits", ["train", "test"])
    overwrite: bool = translation_cfg.get("overwrite", False)
    max_files = translation_cfg.get("max_files")

    translator = MIMICDirectTranslator(lookup_dir=lookup_dir)

    for split in splits:
        in_dir = meds_root / split
        out_dir = output_root / split
        if not in_dir.is_dir():
            print(f"[translation] Skipping split '{split}' (no directory at {in_dir})")
            continue

        exists = out_dir.is_dir() and any(out_dir.glob("*.parquet"))
        if exists and not overwrite:
            print(f"[translation] Found existing translated files in {out_dir}, skipping.")
            continue

        print(f"[translation] Processing split '{split}' from {in_dir} -> {out_dir}")
        translator.process_parquet_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            max_files=max_files,
        )


def _maybe_run_bpe_training(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("bpe_training")
    if not train_cfg:
        return

    if not train_cfg.get("enabled", True):
        return

    data_root = Path(train_cfg["data_root"]).expanduser().resolve()
    column = train_cfg.get("column", "text_sequence_with_time")
    vocab_sizes: List[int] = train_cfg.get("vocab_sizes", [])
    if not vocab_sizes:
        raise ValueError("bpe_training.vocab_sizes must be a non-empty list of integers.")

    output_dir = Path(train_cfg["output_dir"]).expanduser().resolve()
    min_frequency = int(train_cfg.get("min_frequency", 2))
    batch_size = int(train_cfg.get("batch_size", 1000))
    max_train_files = train_cfg.get("max_train_files")
    max_test_files = train_cfg.get("max_test_files")
    max_train_rows = train_cfg.get("max_train_rows")
    max_test_rows = train_cfg.get("max_test_rows")

    run_vocab_experiments(
        data_root=data_root,
        column=column,
        vocab_sizes=vocab_sizes,
        output_dir=output_dir,
        min_frequency=min_frequency,
        batch_size=batch_size,
        max_train_files=max_train_files,
        max_test_files=max_test_files,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )


def _maybe_run_encoding(cfg: Dict[str, Any]) -> None:
    enc_cfg = cfg.get("encoding")
    if not enc_cfg:
        return

    if not enc_cfg.get("enabled", True):
        return

    data_root = Path(enc_cfg["data_root"]).expanduser().resolve()
    column = enc_cfg.get("column", "text_sequence_with_time")
    tokenizer_dir = Path(enc_cfg["tokenizer_dir"]).expanduser().resolve()
    output_root = Path(enc_cfg["output_root"]).expanduser().resolve()
    max_rows = enc_cfg.get("max_rows")
    batch_size = int(enc_cfg.get("batch_size", 256))
    splits: List[str] = enc_cfg.get("splits", ["train", "test"])
    max_files = enc_cfg.get("max_files")

    parquet_files = find_parquet_files_for_encoding(data_root)
    if splits:
        splits_set = set(splits)
        parquet_files = [p for p in parquet_files if p.parent.name in splits_set]
    if max_files is not None:
        parquet_files = parquet_files[: int(max_files)]
    if not parquet_files:
        raise ValueError(f"[encoding] No parquet files found under {data_root}")

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    encode_files_to_pickles(
        tokenizer=tokenizer,
        data_root=data_root,
        parquet_files=parquet_files,
        column=column,
        output_root=output_root,
        max_rows=max_rows,
        batch_size=batch_size,
    )


def run_pipeline(config_path: Path) -> None:
    with config_path.open("r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    _maybe_run_translation(cfg)
    _maybe_run_bpe_training(cfg)
    _maybe_run_encoding(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a config-driven MIMIC pipeline: MEDS -> natural language translation -> "
            "BPE tokenizer training -> BPE encoding."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file describing translation/bpe_training/encoding steps.",
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()

    run_pipeline(config_path)


if __name__ == "__main__":
    main()

