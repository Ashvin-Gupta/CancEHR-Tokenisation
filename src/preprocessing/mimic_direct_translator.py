from pathlib import Path
from typing import Optional, Union

import pandas as pd


PathLike = Union[str, Path]


class MIMICDirectTranslator:
    """
    Translate MEDS-format MIMIC events into natural-language patient sequences.

    This is a dataset-specific preprocessing component that:
    - Uses MIMIC lookup tables for labs, diagnoses, and procedures.
    - Translates per-event codes into natural language.
    - Aggregates per-patient timelines with optional dynamic time buckets.
    """

    def __init__(self, lookup_dir: PathLike):
        """
        Args:
            lookup_dir: Directory containing MIMIC lookup CSVs
                (e.g. d_labitems.csv.gz, d_icd_diagnoses.csv.gz, d_icd_procedures.csv.gz).
        """
        self.lookup_dir = Path(lookup_dir)
        self._load_lookups()

    def _load_lookups(self) -> None:
        """Loads lab, diagnosis, and procedure descriptions."""
        lab_df = pd.read_csv(self.lookup_dir / "d_labitems.csv.gz")
        self.lab_lookup = lab_df.set_index("itemid")["label"].to_dict()

        diag_df = pd.read_csv(self.lookup_dir / "d_icd_diagnoses.csv.gz")
        diag_df["key"] = (
            diag_df["icd_version"].astype(str) + "_" + diag_df["icd_code"].astype(str)
        )
        self.diag_lookup = diag_df.set_index("key")["long_title"].to_dict()

        proc_df = pd.read_csv(self.lookup_dir / "d_icd_procedures.csv.gz")
        proc_df["key"] = (
            proc_df["icd_version"].astype(str) + "_" + proc_df["icd_code"].astype(str)
        )
        self.proc_lookup = proc_df.set_index("key")["long_title"].to_dict()

    def _format_value(self, val):
        """Formats values, forcing numbers to 1 decimal place."""
        if pd.isna(val):
            return ""

        val_str = str(val).strip()
        # Catch MIMIC's blank placeholder
        if val_str == "___":
            return ""

        try:
            # If it can be converted to a float, round to 1 decimal place
            f_val = float(val)
            return f"{f_val:.1f}"
        except ValueError:
            # If it's a string like "110/68" or "NEG", return as is
            return str(val)

    def _get_time_bucket(self, time_delta):
        """Calculates dynamic time buckets for MIMIC, returning the string token."""
        if pd.isna(time_delta):
            return ""

        total_seconds = time_delta.total_seconds()
        if total_seconds <= 0:
            return ""

        total_minutes = total_seconds / 60.0
        if total_minutes < 5.0:
            return ""  # ignore sub-5min gaps

        if total_minutes < 60.0:
            minutes = max(5, min(59, int(total_minutes)))
            return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"

        total_hours = total_seconds / 3600.0
        if total_hours < 24.0:
            hours = max(1, min(23, int(total_hours)))
            return f"{hours} hour" if hours == 1 else f"{hours} hours"

        total_days = total_seconds / 86400.0
        if total_days < 7.0:
            days = max(1, min(6, int(total_days)))
            return f"{days} day" if days == 1 else f"{days} days"

        weeks = max(1, int(total_days / 7.0))
        return f"{weeks} week" if weeks == 1 else f"{weeks} weeks"

    def _translate_row(self, row):
        """Translates a single dataframe row into a natural language phrase."""
        code = str(row.get("code", ""))

        # Format the value to 1 decimal place if it's numeric
        text_val = row.get("text_value")
        num_val = row.get("numeric_value")

        val_str = ""
        if pd.notna(text_val):
            val_str = f" {self._format_value(text_val)}"
        elif pd.notna(num_val):
            val_str = f" {self._format_value(num_val)}"

        try:
            if code.startswith("GENDER//"):
                return "Female" if code.split("//")[1] == "F" else "Male"
            elif code.startswith("RACE//"):
                race_val = (
                    code.split("//")[1]
                    .replace("RACE_", "")
                    .replace("_", " ")
                    .title()
                )
                return f"Race {race_val}"
            elif code.startswith("MARITAL_STATUS//"):
                return code.split("//")[1].upper()
            elif code.startswith("LAB//"):
                parts = code.split("//")
                item_id = int(parts[1])
                unit = parts[2] if len(parts) > 2 and parts[2] != "UNK" else ""
                lab_name = self.lab_lookup.get(item_id, parts[1])
                unit_str = f" {unit}" if unit and val_str.strip() else ""
                return f"{lab_name}{val_str}{unit_str}"
            elif code.startswith("DIAGNOSIS//ICD//"):
                parts = code.split("//")
                return self.diag_lookup.get(f"{parts[2]}_{parts[3]}", parts[3])
            elif code.startswith("PROCEDURE//ICD//"):
                parts = code.split("//")
                return self.proc_lookup.get(f"{parts[2]}_{parts[3]}", parts[3])
            elif any(x in code for x in ["ADMISSION", "DISCHARGE", "TRANSFER"]):
                return code.replace("//", " to ").replace("_", " ")
            elif code.startswith("MEDICATION//"):
                parts = code.split("//")
                if "START" in parts:
                    return f"start {parts[-1]}"
                return f"{parts[1]} {parts[-1]}"
            elif code in ["Blood Pressure", "Weight (Lbs)", "AGE", "BMI"]:
                return f"{code}{val_str}"
            else:
                clean_code = code.replace("//", " ").replace("_", " ")
                return f"{clean_code}{val_str}"
        except Exception:
            return f"{code}{val_str}"

    def _build_patient_sequences(self, patient_df):
        """
        Takes a patient's dataframe (sorted chronologically) and builds
        both the standard sequence and the sequence with time tokens.
        Calculates age and drops MEDS_BIRTH.
        """
        dob = pd.NaT
        first_event_time = pd.NaT

        # 1. Extract DOB
        meds_birth_rows = patient_df[patient_df["code"] == "MEDS_BIRTH"]
        if not meds_birth_rows.empty:
            dob = meds_birth_rows["time"].iloc[0]

        # 2. Filter out MEDS_BIRTH
        df_events = patient_df[patient_df["code"] != "MEDS_BIRTH"].copy()

        # 3. Find the timestamp of the first actual clinical event
        valid_times = df_events["time"].dropna()
        if not valid_times.empty:
            first_event_time = valid_times.iloc[0]

        # 4. Calculate Age string
        age_str = ""
        if pd.notna(dob) and pd.notna(first_event_time):
            age_years = (first_event_time - dob).days / 365.25
            age_str = f"age {age_years:.1f}; "

        seq_no_time = []
        seq_with_time = []
        last_time = pd.NaT

        # 5. Build the sequences
        for _, row in df_events.iterrows():
            current_time = row["time"]
            text = row["translated_text"]

            if not text or str(text).strip() == "":
                continue

            # Time token logic
            if pd.notna(current_time) and pd.notna(last_time):
                delta = current_time - last_time
                bucket = self._get_time_bucket(delta)
                if bucket:
                    seq_with_time.append(f"{bucket}; ")

            if pd.notna(current_time):
                last_time = current_time

            seq_no_time.append(f"{text}; ")
            seq_with_time.append(f"{text}; ")

        # Combine, prepend age, and force lowercase
        final_no_time = (age_str + "".join(seq_no_time)).lower().strip()
        final_with_time = (age_str + "".join(seq_with_time)).lower().strip()

        return pd.Series(
            {
                "text_sequence": final_no_time,
                "text_sequence_with_time": final_with_time,
            }
        )

    def process_parquet_directory(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        max_files: Optional[int] = None,
    ) -> None:
        """
        Process all parquet files in a directory into flattened patient sequences.

        Args:
            input_dir: Directory containing MEDS-format parquet files.
            output_dir: Directory to write translated, per-patient parquet files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".parquet")
        if max_files is not None:
            parquet_files = parquet_files[: int(max_files)]

        for input_path in parquet_files:
            output_path = output_dir / input_path.name
            print(f"\n=== Processing {input_path} ===")
            self.process_parquet(input_path, output_path)

    def process_parquet(self, input_path: PathLike, output_path: PathLike) -> None:
        """
        Translate a single MEDS-format parquet file and aggregate into patient-level sequences.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        print(f"Loading data from {input_path}...")
        df = pd.read_parquet(input_path)

        # Ensure time is datetime and sort
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values(by=["subject_id", "time"], na_position="first")

        print("Translating events to natural language...")
        df["translated_text"] = df.apply(self._translate_row, axis=1)

        print("Aggregating sequences per patient (calculating time tokens and age)...")
        # Apply the custom sequence builder per patient
        patient_sequences = (
            df.groupby("subject_id").apply(self._build_patient_sequences).reset_index()
        )

        print(f"Saving flattened sequences to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patient_sequences.to_parquet(output_path, index=False)
        print("Done!")


def main() -> None:
    """
    Simple CLI for translating MEDS-format parquet files to patient-level text sequences.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Translate MEDS-format parquet files into patient-level natural language "
            "sequences (with and without dynamic time tokens)."
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
        help="Root directory with MEDS parquet under split subdirs (e.g. train/, test/).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to write translated per-patient parquet files into.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process as subdirectories of meds_root.",
    )

    args = parser.parse_args()

    lookup_dir = Path(args.lookup_dir).expanduser().resolve()
    meds_root = Path(args.meds_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    translator = MIMICDirectTranslator(lookup_dir=lookup_dir)

    for split in args.splits:
        in_dir = meds_root / split
        out_dir = output_root / split
        if not in_dir.is_dir():
            print(f"Skipping split '{split}' (no directory at {in_dir})")
            continue
        translator.process_parquet_directory(input_dir=in_dir, output_dir=out_dir)


if __name__ == "__main__":
    main()

