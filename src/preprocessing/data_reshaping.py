import polars as pl
from typing import List
from .base import BasePreprocessor

class DataReshapingPreprocessor(BasePreprocessor):
    """
    A preprocessor to ensure the DataFrame has both 'numeric_value'
    and 'text_value' columns. It creates 'text_value' by casting
    the existing 'numeric_value' column to a string.
    """
    def __init__(self, **kwargs):
        # This preprocessor applies to all data, so it doesn't need real matching values. We provide dummy values to satisfy the parent class's __init__ method.
        super().__init__(matching_type="equals", matching_value="")

    def fit(self, event_files: List[str]) -> None:
        """This preprocessor is rule-based and does not need to be fitted."""
        pass

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Ensures the DataFrame has the required value columns.
        """
        if "numeric_value" not in events.columns:
            raise pl.exceptions.ColumnNotFoundError(
                "The required 'numeric_value' column was not found in the data."
            )

        # Create the 'text_value' column by casting 'numeric_value' to a string.
        # This single step makes the data compatible with all downstream preprocessors.
        events_out = events.with_columns(
            pl.col("numeric_value").cast(pl.Utf8).alias("text_value")
        )

        return events_out