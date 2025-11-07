import polars as pl
import math
from typing import List
from .base import BasePreprocessor

class BinnedAgePreprocessor(BasePreprocessor):
    """
    Calculates a patient's age and bins it into 5-year ranges.
    Age bins: 20-24, 25-29, 30-34, ..., 95-99
    Ages outside 20-99 range are ignored.
    e.g., age 45 becomes AGE_45-49.
    """
    def __init__(self, keep_meds_birth: bool = False, **kwargs):
        super().__init__(matching_type="equals", matching_value="")
        self.keep_meds_birth = keep_meds_birth
        print(f"BinnedAgePreprocessor initialized: Keep MEDS_BIRTH = {self.keep_meds_birth}")

    def fit(self, event_files: List[str]) -> None:
        """This preprocessor is rule-based and does not need to be fitted."""
        print("BinnedAgePreprocessor fit complete (no training required).")
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

            age_event = None

            # Bin into 5-year ranges: 20-24, 25-29, ..., 95-99
            if 20 <= age_in_years < 100:
                age_int = int(math.floor(age_in_years))
                # Calculate the lower bound of the 5-year bin
                age_bin_lower = (age_int // 5) * 5
                # Ensure we don't go below 20
                if age_bin_lower < 20:
                    age_bin_lower = 20
                # Ensure we don't go above 95
                if age_bin_lower > 95:
                    age_bin_lower = 95
                age_bin_upper = age_bin_lower + 4
                
                # Create the age bin event
                age_event = {
                    "subject_id": subject_id, "time": None,
                    # "code": "AGE_bin", "numeric_value": None,
                    "code": f"AGE: {age_bin_lower}-{age_bin_upper}", "numeric_value": None,
                    "text_value": None
                }

                # Fill in schema for the event
                for col_name in events.columns:
                    if col_name not in age_event:
                        age_event[col_name] = None
                
                age_events.append(age_event)

        if not self.keep_meds_birth:
            events = events.filter(pl.col("code") != "MEDS_BIRTH")

        if not age_events:
            return events

        age_df = pl.from_dicts(age_events, schema=events.schema)
        combined_events = pl.concat([events, age_df]).sort(["subject_id", "time"])
        
        return combined_events

