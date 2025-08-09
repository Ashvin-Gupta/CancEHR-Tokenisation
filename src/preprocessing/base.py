from typing import List, Dict, Any
import polars as pl
import numpy as np
from abc import ABC, abstractmethod
import os
from tqdm import tqdm

class Preprocessor(ABC):
    """
    Base class for all preprocessors with matching functionality

    Args:
        matching_type (str): the type of matching to perform
        matching_value (str): the value to match against
    """
    def __init__(self, matching_type: str, matching_value: str):
        self.matching_type = matching_type
        self.matching_value = matching_value
        
        # validate matching type
        if matching_type not in ["starts_with", "ends_with", "contains", "equals"]:
            raise ValueError(f"Invalid matching type: {matching_type}")
        
        # data storage for codes and their values
        self.data: Dict[str, List[Any]] = {}

        # store the fits for each code
        self.fits = {}
        
    def _match(self, code: str) -> bool:
        """
        Check if a code matches the configured pattern

        Args:
            code (str): the code to check

        Returns:
            bool: True if the code matches the configured pattern, False otherwise
        """
        if self.matching_type == "starts_with":
            return code.startswith(self.matching_value)
        elif self.matching_type == "ends_with":
            return code.endswith(self.matching_value)
        elif self.matching_type == "contains":
            return self.matching_value in code
        elif self.matching_type == "equals":
            return code == self.matching_value
        return False
        
    def fit(self, event_files: List[str]) -> None:
        """
        Train the preprocessor on a list of event files.
        Calls _fit() to fit the preprocessor to the data.

        Args:
            event_files (List[str]): the list of event files to train on
        """
        # Validate all files exist before processing
        for file_path in event_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Event file not found: {file_path}")
            
        # Loop through each event file and populate the data dictionary
        for event_file in tqdm(event_files, desc="Fitting preprocessor", leave=False):
            events = pl.read_parquet(event_file)
            
            for event in events.to_dicts():
                # Check if the event has a numeric value and matches the matching criteria
                if event["numeric_value"] is not None and self._match(event["code"]):

                    # Add the datapoint to the data dictionary
                    if event["code"] not in self.data:
                        self.data[event["code"]] = []
                    self.data[event["code"]].append(event["numeric_value"])
        
        # Call the child class's _fit() method to fit the preprocessor to the data
        self._fit()

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Encode the data in the event files and overwrite numeric_value column.

        Args:
            events (pl.DataFrame): the events to encode

        Returns:
            pl.DataFrame: the events with the encoded numeric_value column
        """
        def encode_row(row):
            code = row["code"]
            numeric_value = row["numeric_value"]
            
            if self._match(code) and numeric_value is not None:
                encoded = self._encode(code, numeric_value)
                return str(encoded) if encoded is not None else str(numeric_value)
            return str(numeric_value) if numeric_value is not None else None

        events = events.with_columns(
            pl.struct(["code", "numeric_value"]).map_elements(encode_row, return_dtype=pl.String).alias("numeric_value")
        )
        return events
            
    @abstractmethod
    def _fit(self, values: np.ndarray) -> Any:
        """
        Fit the preprocessor to the training data, implemented by subclasses

        Args:
            values: array of values to fit the preprocessor to
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _encode(self, values: np.ndarray) -> List[str]:
        """
        Encode values using the fitted preprocessor, implemented by subclasses

        Args:
            values: array of values to encode
        """
        raise NotImplementedError("Subclasses must implement this method")
