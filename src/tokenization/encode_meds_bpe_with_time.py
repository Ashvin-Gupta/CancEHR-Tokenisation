from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from src.preprocessing.mimic_direct_translator import MIMICDirectTranslator


def find_meds_parquet_files(meds_root: Path) -> List[Path]:
    """
    Collect all MEDS parquet files under meds_root/train and meds_root/test.
    """
    files: List[Path] = []
    for split in ("train", "test"):
        split_dir = meds_root / split
        if not split_dir.is_dir():
            continue
        files.extend(sorted(split_dir.glob("*.parquet")))
    return files


def _to_unix_seconds(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return 0.0
    # Naive timestamps are treated as UTC here.
    return float(ts.timestamp())


def build_word_timeline_for_patient(
    translator: MIMICDirectTranslator,
    patient_df: pd.DataFrame,
) -> Tuple[List[str], List[float]]:
    """
    Build a per-patient word-level timeline with real timestamps.

    Returns:
        words: list of tokens (strings), already lowercased
        times: list of unix-second timestamps aligned 1:1 with words
    """
    dob = pd.NaT
    first_event_time = pd.NaT

    meds_birth_rows = patient_df[patient_df["code"] == "MEDS_BIRTH"]
    if not meds_birth_rows.empty:
        dob = meds_birth_rows["time"].iloc[0]

    df_events = patient_df[patient_df["code"] != "MEDS_BIRTH"].copy()

    valid_times = df_events["time"].dropna()
    if not valid_times.empty:
        first_event_time = valid_times.iloc[0]

    words: List[str] = []
    times: List[float] = []

    # Age prefix
    if pd.notna(dob) and pd.notna(first_event_time):
        age_years = (first_event_time - dob).days / 365.25
        age_text = f"age {age_years:.1f};"
        age_tokens = age_text.lower().split()
        age_ts = _to_unix_seconds(first_event_time)
        for tok in age_tokens:
            words.append(tok)
            times.append(age_ts)

    last_time = pd.NaT

    for _idx, row in df_events.iterrows():
        current_time = row["time"]
        text = row["translated_text"]

        if not text or str(text).strip() == "":
            continue

        # Time bucket token(s) between events
        if pd.notna(current_time) and pd.notna(last_time):
            delta = current_time - last_time
            bucket = translator._get_time_bucket(delta)  # type: ignore[attr-defined]
            if bucket:
                bucket_text = f"{bucket};"
                bucket_tokens = bucket_text.lower().split()
                bucket_ts = _to_unix_seconds(current_time)
                for tok in bucket_tokens:
                    words.append(tok)
                    times.append(bucket_ts)

        if pd.notna(current_time):
            last_time = current_time

        # Event text tokens
        event_text = f"{str(text).lower().strip()};"
        event_tokens = event_text.split()
        event_ts = _to_unix_seconds(current_time)
        for tok in event_tokens:
            words.append(tok)
            times.append(event_ts)

    return words, times


def encode_meds_file(
    translator: MIMICDirectTranslator,
    tokenizer: PreTrainedTokenizerFast,
    parquet_path: Path,
    output_root: Path,
) -> None:
    """
    Encode one MEDS parquet file into BPE token timelines with real timestamps.
    """
    df = pd.read_parquet(parquet_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(by=["subject_id", "time"], na_position="first")

    # Translate each row to natural language once
    df["translated_text"] = df.apply(translator._translate_row, axis=1)  # type: ignore[attr-defined]

    timelines: List[Dict] = []

    for subject_id, patient_df in tqdm(
        df.groupby("subject_id"),
        desc=f"Encoding {parquet_path.name}",
    ):
        words, word_times = build_word_timeline_for_patient(translator, patient_df)
        if not words:
            continue

        enc = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        token_ids = enc["input_ids"]
        word_ids = enc.word_ids()

        expanded_times: List[float] = []
        for wid in word_ids:
            if wid is None:
                expanded_times.append(0.0)
            else:
                expanded_times.append(word_times[wid])

        timelines.append(
            {
                "subject_id": int(subject_id),
                "tokens": token_ids,
                "timestamps": expanded_times,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    save_path = output_root / f"{parquet_path.stem}.pkl"
    with save_path.open("wb") as f:
        pickle.dump(timelines, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start from MEDS-format parquet, translate to natural language with "
            "MIMICDirectTranslator, then BPE-encode with a trained tokenizer, "
            "producing subject-level timelines with real unix-second timestamps."
        )
    )
    parser.add_argument(
        "--lookup_dir",
        type=str,
        required=True,
        help="Directory containing MIMIC lookup CSVs (d_labitems, d_icd_*).",
    )
    parser.add_argument(
        "--meds_root",
        type=str,
        required=True,
        help="Root directory with MEDS parquet under train/ and test/.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Directory of the trained BPE tokenizer (e.g. artifacts/tokenizers/experiments/vocab_4000).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory to write .pkl timelines into.",
    )

    args = parser.parse_args()

    lookup_dir = Path(args.lookup_dir).expanduser().resolve()
    meds_root = Path(args.meds_root).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    translator = MIMICDirectTranslator(lookup_dir=str(lookup_dir))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    meds_files = find_meds_parquet_files(meds_root)
    if not meds_files:
        raise ValueError(f"No MEDS parquet files found under {meds_root}")

    for parquet_path in meds_files:
        encode_meds_file(
            translator=translator,
            tokenizer=tokenizer,
            parquet_path=parquet_path,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()

