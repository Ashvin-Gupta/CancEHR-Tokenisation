import polars as pl
import math
from typing import List
from .base import BasePreprocessor

class DecimalAgePreprocessor(BasePreprocessor):
    """
    Calculates a patient's age and decomposes it into two tokens:
    one for the decile (tens digit) and one for the unit (ones digit).
    e.g., age 45 becomes AGE_decile Q4 and AGE_unit Q5.
    """
    def __init__(self, keep_meds_birth: bool = False, **kwargs):
        super().__init__(matching_type="equals", matching_value="")
        self.keep_meds_birth = keep_meds_birth
        print(f"DecimalAgePreprocessor initialized: Keep MEDS_BIRTH = {self.keep_meds_birth}")

    def fit(self, event_files: List[str]) -> None:
        """This preprocessor is rule-based and does not need to be fitted."""
        print("DecimalAgePreprocessor fit complete (no training required).")
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

            decile_event = None
            unit_event = None

            if age_in_years >= 0:
                age_int = math.floor(age_in_years)
                decile = age_int // 10
                unit = age_int % 10

                # Create the decile event
                decile_event = {
                    "subject_id": subject_id, "time": None,
                    "code": "AGE_decile", "numeric_value": None,
                    "text_value": f"Q{decile}"
                }
                
                # Create the unit event
                unit_event = {
                    "subject_id": subject_id, "time": None,
                    "code": "AGE_unit", "numeric_value": None,
                    "text_value": f"Q{unit}"
                }

                # Fill in schema for both
                for col_name in events.columns:
                    if col_name not in decile_event:
                        decile_event[col_name] = None
                    if col_name not in unit_event:
                        unit_event[col_name] = None
                
                age_events.extend([decile_event, unit_event])

        if not self.keep_meds_birth:
            events = events.filter(pl.col("code") != "MEDS_BIRTH")

        if not age_events:
            return events

        age_df = pl.from_dicts(age_events, schema=events.schema)
        combined_events = pl.concat([events, age_df]).sort(["subject_id", "time"])
        
        return combined_events