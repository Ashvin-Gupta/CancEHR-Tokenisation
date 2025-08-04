from typing import List
import os
from tqdm import tqdm
import polars as pl
from .base import Preprocessor


def fit_preprocessors_jointly(preprocessors: List[Preprocessor], event_files: List[str]):
    """
    Fit multiple preprocessors jointly by reading through the data files only once.
    
    Args:
        preprocessors (List[Preprocessor]): list of preprocessors to fit
        event_files (List[str]): list of event files to train on
    """
    # Validate all files exist before processing
    for file_path in event_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Event file not found: {file_path}")
    
    # Loop through each event file once and collect data for all preprocessors
    for event_file in tqdm(event_files, desc="Collecting data for all preprocessors"):
        events = pl.read_parquet(event_file)
        
        for event in events.to_dicts():
            # Check each preprocessor to see if this event matches its criteria
            for preprocessor in preprocessors:
                if event["numeric_value"] is not None and preprocessor._match(event["code"]):
                    # Add the datapoint to this preprocessor's data dictionary
                    if event["code"] not in preprocessor.data:
                        preprocessor.data[event["code"]] = []
                    preprocessor.data[event["code"]].append(event["numeric_value"])
    
    # Now fit each preprocessor to its collected data
    for preprocessor in tqdm(preprocessors, desc="Fitting preprocessors", leave=False):
        preprocessor._fit() 