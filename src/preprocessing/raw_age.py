import polars as pl
import math
from typing import List
from .base import BasePreprocessor

class RawAgePreprocessor(BasePreprocessor):
    """
    Calculates a patient's age and inserts it as a raw numeric value.
    Resulting sequence: "AGE" <number>
    e.g., AGE 45.0
    """
    def __init__(self, keep_meds_birth: bool = False, decimals: int = 0, **kwargs):
        super().__init__(matching_type="equals", matching_value="")
        self.keep_meds_birth = keep_meds_birth
        self.decimals = decimals
        print(f"RawAgePreprocessor initialized: Keep MEDS_BIRTH={keep_meds_birth}, Decimals={decimals}")

    def fit(self, event_files: List[str]) -> None:
        """Rule-based preprocessor, no fitting required."""
        pass

    def _calculate_age_from_timeline(self, subject_events: pl.DataFrame) -> float:
        """Calculates age at first event in years."""
        birth_events = subject_events.filter(pl.col("code") == "MEDS_BIRTH")
        if birth_events.is_empty():
            return -1.0

        birth_time = birth_events["time"].to_list()[0]
        if birth_time is None:
            return -1.0

        real_events = subject_events.filter(
            (pl.col("code") != "MEDS_BIRTH") & (pl.col("time").is_not_null()) &
            (~pl.col("code").str.starts_with("DEMOGRAPHICS//")) &
            (~pl.col("code").str.starts_with("GENDER//")) &
            (~pl.col("code").str.starts_with("ETHNICITY//")) &
            (~pl.col("code").str.starts_with("REGION//")) &
            (pl.col("code") != "STATIC_DATA_NO_CODE")
        ).sort("time")
        
        if real_events.is_empty():
            return -1.0

        first_medical_time = real_events["time"].to_list()[0]
        if first_medical_time is None:
            return -1.0
            
        time_delta_seconds = first_medical_time.timestamp() - birth_time.timestamp()
        
        return time_delta_seconds / (365.25 * 24 * 3600)

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        age_events = []
        subject_ids = events["subject_id"].unique()

        for subject_id in subject_ids:
            subject_events = events.filter(pl.col("subject_id") == subject_id)
            age_in_years = self._calculate_age_from_timeline(subject_events)

            if age_in_years >= 0:
                # Format the age to the specified decimals (e.g. 45 or 45.2)
                if self.decimals == 0:
                    formatted_age = int(round(age_in_years))
                else:
                    formatted_age = round(age_in_years, self.decimals)

                # Create the age event
                # code="AGE", numeric_value=45
                age_event = {
                    "subject_id": subject_id, 
                    "time": None,
                    "code": "AGE", 
                    "numeric_value": formatted_age,
                    "text_value": None
                }

                # Fill in schema
                for col_name in events.columns:
                    if col_name not in age_event:
                        age_event[col_name] = None
                
                age_events.append(age_event)

        if not self.keep_meds_birth:
            events = events.filter(pl.col("code") != "MEDS_BIRTH")

        if not age_events:
            return events

        age_df = pl.from_dicts(age_events, schema=events.schema)
        
        # Ensure schema compatibility (float vs int issues)
        for col in events.columns:
            original_dtype = events.select(col).dtypes[0]
            age_df = age_df.with_columns(pl.col(col).cast(original_dtype))

        combined_events = pl.concat([events, age_df]).sort(["subject_id", "time"])
        
        return combined_events