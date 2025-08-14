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
        value_column (str): the column containing the numeric values to bin.
    """
    def __init__(self, matching_type: str, matching_value: str, value_column: str):
        self.matching_type = matching_type
        self.matching_value = matching_value
        self.value_column = value_column
        
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

                value = event[self.value_column]
                # Check if the event has a numeric value and matches the matching criteria
                if value is not None and self._match(event["code"]):

                    # map value to float (in case it is a string)
                    value = float(value)

                    print(event["code"], value, type(value))

                    # Add the datapoint to the data dictionary
                    if event["code"] not in self.data:
                        self.data[event["code"]] = []
                    self.data[event["code"]].append(value)
        
        # Call the child class's _fit() method to fit the preprocessor to the data
        self._fit()

    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Encode the data in the event files and overwrite self.value_column column.

        Args:
            events (pl.DataFrame): the events to encode

        Returns:
            pl.DataFrame: the events with the encoded self.value_column column
        """
        def encode_row(row):
            code = row["code"]
            value = row[self.value_column]
            
            if self._match(code) and value is not None:
                value = float(value)
                encoded = self._encode(code, value)
                return str(encoded) if encoded is not None else str(value)
            return str(value) if value is not None else None

        events = events.with_columns(
            pl.struct(["code", self.value_column]).map_elements(encode_row, return_dtype=pl.String).alias(self.value_column)
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
