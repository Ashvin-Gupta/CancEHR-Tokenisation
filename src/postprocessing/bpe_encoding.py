from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import polars as pl
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def find_parquet_files(data_root: Path) -> List[Path]:
    """
    Collect all parquet files under data_root/train and data_root/test.
    """
    files: List[Path] = []
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        files.extend(sorted(split_dir.glob("*.parquet")))
    return files


def encode_files_to_pickles(
    tokenizer: PreTrainedTokenizerFast,
    data_root: Path,
    parquet_files: Iterable[Path],
    column: str,
    output_root: Path,
    max_rows: Optional[int] = None,
    batch_size: int = 256,
) -> None:
    """
    Encode all rows in the given parquet files and save as .pkl timelines.

    Each .pkl file contains a list of dicts:
        { "subject_id": <int>, "tokens": <List[int]>, "timestamps": <List[int]> }
    """
    rows_encoded = 0

    for file_path in parquet_files:
        df = pl.read_parquet(file_path, columns=["subject_id", column])
        subject_series = df["subject_id"]
        text_series = df[column]
        n_rows = len(text_series)

        timelines: List[Dict] = []

        for start in tqdm(
            range(0, n_rows, batch_size),
            desc=f"Encoding {file_path.name}",
        ):
            end = min(start + batch_size, n_rows)
            batch_subjects = []
            batch_texts = []

            for i in range(start, end):
                value = text_series[i]
                if value is None:
                    continue
                text = str(value)
                if not text:
                    continue

                batch_subjects.append(subject_series[i])
                batch_texts.append(text)

                rows_encoded += 1
                if max_rows is not None and rows_encoded >= max_rows:
                    break

            if not batch_texts:
                if max_rows is not None and rows_encoded >= max_rows:
                    break
                continue

            encodings = tokenizer(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            for subj, ids in zip(batch_subjects, encodings):
                # We no longer have true event timestamps in this flattened
                # text representation, so we use simple positional indices
                # as a surrogate timeline: [0, 1, 2, ..., len(tokens)-1].
                timestamps = list(range(len(ids)))
                timelines.append(
                    {
                        "subject_id": subj,
                        "tokens": ids,
                        "timestamps": timestamps,
                    }
                )

            if max_rows is not None and rows_encoded >= max_rows:
                break

        # Save one .pkl per source parquet file in a single directory.
        # Prefix with split name (train/test) to avoid collisions.
        split_name = file_path.parent.name  # e.g. "train" or "test"
        base_name = file_path.stem  # e.g. "0"
        save_path = output_root / f"{split_name}_{base_name}.pkl"
        output_root.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(timelines, f)

        if max_rows is not None and rows_encoded >= max_rows:
            break

