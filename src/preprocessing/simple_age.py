import polars as pl
from typing import List, Dict, Any
from .base import BasePreprocessor

class SimpleAgePreprocessor(BasePreprocessor):
    """
    A preprocessor that calculates a patient's age at their first medical event
    and inserts it as a single static event with the code "AGE".
    """
    def __init__(self, keep_meds_birth: bool = False, **kwargs):
        """
        Args:
            keep_meds_birth (bool): Whether to keep the original MEDS_BIRTH token.
        """
        super().__init__(matching_type="equals", matching_value="")
        self.keep_meds_birth = keep_meds_birth
        print(f"SimpleAgePreprocessor initialized: Keep MEDS_BIRTH = {self.keep_meds_birth}")

    def fit(self, event_files: List[str]) -> None:
        """This preprocessor is rule-based and does not need to be fitted."""
        print("SimpleAgePreprocessor fit complete (no training required).")
        pass

    def _calculate_age_from_timeline(self, subject_events: pl.DataFrame) -> float:
        """
        Calculate age from the time delta between MEDS_BIRTH and the first real medical event.
        (This logic is reused from the EthosQuantileAgePreprocessor).
        """
        birth_events = subject_events.filter(pl.col("code") == "MEDS_BIRTH")
        if birth_events.is_empty():
            return 0.0
        
        birth_time = birth_events["time"].to_list()[0]
        if birth_time is None:
            return 0.0

        real_events = subject_events.filter(
            (pl.col("code") != "MEDS_BIRTH") &
            (pl.col("time").is_not_null()) &
            (~pl.col("code").str.starts_with("DEMOGRAPHICS//")) &
            (~pl.col("code").str.starts_with("GENDER//")) &
            (~pl.col("code").str.starts_with("RACE//")) &
            (~pl.col("code").str.starts_with("ETHNICITY//")) &
            (~pl.col("code").str.starts_with("REGION//")) &
            (pl.col("code") != "STATIC_DATA_NO_CODE")
        ).sort("time")
        
        if real_events.is_empty():
            return 0.0

        first_medical_time = real_events["time"].to_list()[0]
        if first_medical_time is None:
            return 0.0
            
        time_delta_seconds = first_medical_time.timestamp() - birth_time.timestamp()
        
        # Return age in years, rounded to one decimal place
        return round(max(0.0, time_delta_seconds / (365.25 * 24 * 3600)), 1)

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Replaces or supplements MEDS_BIRTH events with a single 'AGE' event.
        """
        age_events = []
        subject_ids = events["subject_id"].unique()

        for subject_id in subject_ids:
            subject_events = events.filter(pl.col("subject_id") == subject_id)
            age_in_years = self._calculate_age_from_timeline(subject_events)

            # Create a new event for the age
            age_event = {
                "subject_id": subject_id,
                "time": None,  # Null timestamp makes it a static event
                "code": "AGE",
                "numeric_value": age_in_years,
                "text_value": str(age_in_years) # Also add as text for consistency
            }
            age_events.append(age_event)

        # Remove original MEDS_BIRTH events if configured to do so
        if not self.keep_meds_birth:
            events = events.filter(pl.col("code") != "MEDS_BIRTH")

        if not age_events:
            return events

        # Convert the new age events to a DataFrame
        age_df = pl.from_dicts(age_events, schema=events.schema)

        # Combine with the main events and sort
        combined_events = pl.concat([events, age_df]).sort(["subject_id", "time"])
        
        return combined_events