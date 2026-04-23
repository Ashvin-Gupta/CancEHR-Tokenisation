from .base import BasePreprocessor
from typing import List
import polars as pl

class FilterPreprocessor(BasePreprocessor):
    """
    A preprocessor that removes events whose codes match specified criteria.
    Useful for dropping noisy tokens like TIME_OF_DAY or duplicate AGE events.
    """
    def __init__(self, matching_type: str, matching_value: str, invert: bool = False):
        super().__init__(matching_type, matching_value)
        self.invert = invert

    def fit(self, event_files: List[str]) -> None:
        """Rule-based preprocessor, no fitting required."""
        pass

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        mask = None
        if self.matching_type == "starts_with":
            mask = pl.col("code").str.starts_with(self.matching_value)
        elif self.matching_type == "equals":
            mask = pl.col("code") == self.matching_value
        elif self.matching_type == "contains":
            # literal=True makes this a simple string match
            mask = pl.col("code").str.contains(self.matching_value, literal=True)
        elif self.matching_type == "regex":
            # literal=False (default) enables regex matching in Polars
            mask = pl.col("code").str.contains(self.matching_value, literal=False)
            
        if mask is None:
            return events
    
        return events.filter(~mask if not self.invert else mask)