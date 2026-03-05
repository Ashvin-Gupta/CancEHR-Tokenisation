from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from collections import Counter

import polars as pl
from tqdm import tqdm
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


# Defaults are only used by CLIs or pipelines that don't override them.
DEFAULT_DATA_ROOT = Path("translated")
DEFAULT_COLUMN = "text_sequence_with_time"
DEFAULT_VOCAB_SIZES: List[int] = [4_000]
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/experiments")
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_BATCH_SIZE = 1000


def iter_parquet_column(
    files: Iterable[Path],
    column: str,
    max_rows: Optional[int] = None,
) -> Iterator[str]:
    """Stream values from a specific column across parquet files (optionally capped)."""
    yielded = 0
    for file in files:
        if max_rows is not None and yielded >= max_rows:
            break

        remaining = None if max_rows is None else max_rows - yielded
        if remaining is not None and remaining <= 0:
            break

        # Use lazy scan + head() so debug runs don't read huge files.
        lf = pl.scan_parquet(str(file)).select(pl.col(column))
        if remaining is not None:
            lf = lf.head(int(remaining))
        df = lf.collect()

        for value in df[column]:
            if max_rows is not None and yielded >= max_rows:
                break
            if value:
                yielded += 1
                yield str(value)


def find_parquet_files(data_root: Path, split: str) -> List[Path]:
    """Collect parquet files for a specific split (train or test)."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.parquet"))


def train_tokenizer(
    train_files: List[Path],
    column: str,
    vocab_size: int,
    min_frequency: int = 2,
    max_train_rows: Optional[int] = None,
) -> PreTrainedTokenizerFast:
    """Train a HF BPE tokenizer on text drawn from parquet columns."""
    core = HFTokenizer(models.BPE(unk_token="<unk>"))

    # EHR-optimized pre-tokenization: split digits individually, then apply ByteLevel.
    core.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    core.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<bos>", "<eos>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    iterator = iter_parquet_column(files=train_files, column=column, max_rows=max_train_rows)
    core.train_from_iterator(iterator=iterator, trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token=None,
    )


def evaluate_and_save_vocab(
    tokenizer: PreTrainedTokenizerFast,
    test_files: List[Path],
    column: str,
    output_dir: Path,
    vocab_size: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_test_rows: Optional[int] = None,
) -> dict:
    """Evaluate the tokenizer on the test set and generate vocab.csv + metrics."""
    id_counts: Counter[int] = Counter()
    total_tokens = 0
    total_chars = 0
    num_sequences = 0
    rows_seen = 0

    for file_path in test_files:
        if max_test_rows is not None and rows_seen >= max_test_rows:
            break

        remaining = None if max_test_rows is None else max_test_rows - rows_seen
        lf = pl.scan_parquet(str(file_path)).select(pl.col(column)).drop_nulls()
        if remaining is not None:
            lf = lf.head(int(remaining))
        df = lf.collect()
        series = df[column].to_list()
        rows_seen += len(series)

        for start in tqdm(
            range(0, len(series), batch_size),
            desc=f"Eval Size {vocab_size}",
        ):
            batch_texts = series[start : start + batch_size]
            if not batch_texts:
                continue

            total_chars += sum(len(text) for text in batch_texts)
            num_sequences += len(batch_texts)

            encodings = tokenizer(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for ids in encodings:
                total_tokens += len(ids)
                id_counts.update(ids)

    avg_seq_length = total_tokens / num_sequences if num_sequences > 0 else 0
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    tokens_id = [idx for _token, idx in vocab_items]
    tokens_str = [token for token, _idx in vocab_items]
    tokens_count = [id_counts.get(idx, 0) for _token, idx in vocab_items]

    special_tokens = ["<unk>", "<bos>", "<eos>"]
    vocab_df = pl.DataFrame({"token": tokens_id, "str": tokens_str, "count": tokens_count})
    vocab_df = (
        vocab_df.with_columns(
            pl.col("str").is_in(special_tokens).alias("is_special")
        )
        .sort(by=["is_special", "count"], descending=[True, True])
        .drop("is_special")
    )

    exp_dir = output_dir / f"vocab_{vocab_size}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(exp_dir)
    vocab_df.write_csv(exp_dir / "vocab.csv")

    return {
        "vocab_size": vocab_size,
        "avg_seq_length": round(avg_seq_length, 2),
        "compression_ratio": round(compression_ratio, 2),
        "total_tokens_in_test": total_tokens,
    }


def run_vocab_experiments(
    data_root: Path,
    column: str,
    vocab_sizes: List[int],
    output_dir: Path,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_train_files: Optional[int] = None,
    max_test_files: Optional[int] = None,
    max_train_rows: Optional[int] = None,
    max_test_rows: Optional[int] = None,
) -> List[dict]:
    """
    Train and evaluate one or more vocab sizes on parquet-backed text.

    This is the main API used by both CLIs and higher-level pipelines.
    """
    train_files = find_parquet_files(data_root, "train")
    test_files = find_parquet_files(data_root, "test")
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    if not train_files or not test_files:
        raise ValueError("Ensure both 'train' and 'test' folders contain parquet files.")

    print(f"Starting experiments across vocab sizes: {vocab_sizes}")
    results: List[dict] = []

    for size in vocab_sizes:
        print(f"\n--- Training Tokenizer (Size: {size}) ---")
        tokenizer = train_tokenizer(
            train_files=train_files,
            column=column,
            vocab_size=size,
            min_frequency=min_frequency,
            max_train_rows=max_train_rows,
        )

        print(f"--- Evaluating Tokenizer (Size: {size}) ---")
        metrics = evaluate_and_save_vocab(
            tokenizer=tokenizer,
            test_files=test_files,
            column=column,
            output_dir=output_dir,
            vocab_size=size,
            batch_size=batch_size,
            max_test_rows=max_test_rows,
        )
        results.append(metrics)
        print(f"Results for {size}: {metrics}")

    pl.DataFrame(results).write_csv(output_dir / "experiment_metrics.csv")
    print(f"\nAll experiments complete! Metrics saved to {output_dir / 'experiment_metrics.csv'}")
    return results

