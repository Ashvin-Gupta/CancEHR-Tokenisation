from typing import List, Dict, Any
import polars as pl
import numpy as np
from abc import ABC, abstractmethod
import os
from tqdm import tqdm

class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors with matching functionality
    
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
        
    def _match(self, code: str) -> bool:
        """
        Check if a code matches the configured pattern

        Args:
            code (str): the code to check

        Returns:
            bool: True if the code matches the configured pattern, False otherwise
        """
        # Handle None or empty codes gracefully
        if code is None or code == "":
            return False
            
        if self.matching_type == "starts_with":
            return code.startswith(self.matching_value)
        elif self.matching_type == "ends_with":
            return code.endswith(self.matching_value)
        elif self.matching_type == "contains":
            return self.matching_value in code
        elif self.matching_type == "equals":
            return code == self.matching_value
        return False
    
    @abstractmethod
    def fit(self, event_files: List[str]) -> None:
        """
        Train the preprocessor on a list of event files.

        Args:
            event_files (List[str]): the list of event files to train on
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Encode the data in the event files.

        Args:
            events (pl.DataFrame): the events to encode

        Returns:
            pl.DataFrame: the events with encoded data
        """
        raise NotImplementedError("Subclasses must implement this method")


class ValuePreprocessor(BasePreprocessor):
    """
    Base class for preprocessors that transform numeric/text values in a specific column
    
    Args:
        matching_type (str): the type of matching to perform
        matching_value (str): the value to match against
        value_column (str): the column containing the values to transform
    """
    def __init__(self, matching_type: str, matching_value: str, value_column: str):
        super().__init__(matching_type, matching_value)
        self.value_column = value_column
        
        # data storage for codes and their values
        self.data: Dict[str, List[Any]] = {}
        
        # store the fits for each code
        self.fits = {}
        
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
                # Check if the event has a value and matches the matching criteria
                if value is not None and self._match(event["code"]):

                    # map value to float (in case it is a string), if it fails skip it as its not a valid value
                    try:
                        value = float(value)
                    except:
                        continue

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
        other_value_column = 'numeric_value' if self.value_column == 'text_value' else 'text_value'

        def encode_row(row):
            code = row["code"]
            value = row[self.value_column]
            other_value = row[other_value_column]

            if value is None:
                return {"value_col_out": None, "other_col_out": other_value}

            # if the code does not match the matching criteria, return the value as a string
            if not self._match(code):
                return {"value_col_out": str(value), "other_col_out": other_value}

            # attempt to map value to float (in case it is a string), if it fails skip it as its not a valid value
            try:
                value = float(value)
            except:
                return {"value_col_out": str(value), "other_col_out": other_value}

            # encode the value
            encoded = self._encode(code, value)
            if encoded is not None:
                return {"value_col_out": str(encoded), "other_col_out": None}
            else:
                return {"value_col_out": str(value), "other_col_out": other_value}

        events = events.with_columns(
            pl.struct(["code", self.value_column, other_value_column])
            .map_elements(encode_row, return_dtype=pl.Struct([
                pl.Field("value_col_out", pl.Utf8),
                pl.Field("other_col_out", events.schema[other_value_column])
            ]))
            .alias("updated_values")
        ).unnest("updated_values").rename({"value_col_out": self.value_column, "other_col_out": other_value_column})
        
        return events
            
    @abstractmethod
    def _fit(self) -> None:
        """
        Fit the preprocessor to the training data, implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _encode(self, code: str, value: float) -> str:
        """
        Encode a value using the fitted preprocessor, implemented by subclasses

        Args:
            code (str): the code 
            value (float): the value to encode
            
        Returns:
            str: the encoded value
        """
        raise NotImplementedError("Subclasses must implement this method")


class CodePreprocessor(BasePreprocessor):
    """
    Base class for preprocessors that transform/enrich codes
    
    Args:
        matching_type (str): the type of matching to perform
        matching_value (str): the value to match against
    """
    def __init__(self, matching_type: str, matching_value: str):
        super().__init__(matching_type, matching_value)
        
    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Encode codes in the events DataFrame by transforming the 'code' column.

        Args:
            events (pl.DataFrame): the events to encode

        Returns:
            pl.DataFrame: the events with transformed codes
        """
        def transform_row(row):
            code = row["code"]
            if self._match(code):
                return self._transform_code(code)
            return code

        events = events.with_columns(
            pl.struct(["code"]).map_elements(transform_row, return_dtype=pl.String).alias("code")
        )
        return events
    
    @abstractmethod
    def _transform_code(self, code: str) -> str:
        """
        Transform a code, implemented by subclasses

        Args:
            code (str): the original code
            
        Returns:
            str: the transformed code
        """
        raise NotImplementedError("Subclasses must implement this method")
