from .base import BasePreprocessor
from typing import List, Set
import polars as pl
from tqdm import tqdm

class TopKCodePreprocessor(BasePreprocessor):
    """
    Keeps only the Top K most frequent codes for a specific prefix (e.g., LAB//).
    """
    def __init__(self, matching_type: str, matching_value: str, k: int = 100):
        super().__init__(matching_type, matching_value)
        self.k = k
        self.top_codes: Set[str] = set()

    def fit(self, event_files: List[str]) -> None:
        counts = pl.DataFrame()
        for file in tqdm(event_files, desc="Calculating Top K"):
            df = pl.read_parquet(file).filter(pl.col("code").str.starts_with(self.matching_value))
            df_counts = df.group_by("code").agg(pl.len().alias("count"))
            counts = pl.concat([counts, df_counts]).group_by("code").agg(pl.col("count").sum())
        
        self.top_codes = set(counts.sort("count", descending=True).limit(self.k)["code"].to_list())
        print(f"Top {self.k} codes for {self.matching_value} identified.")

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        # Keep if it doesn't match the prefix OR if it's in the top_codes list
        return events.filter(
            (~pl.col("code").str.starts_with(self.matching_value)) | 
            (pl.col("code").is_in(list(self.top_codes)))
        )