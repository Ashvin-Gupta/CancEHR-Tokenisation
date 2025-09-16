from typing import List
import os
from tqdm import tqdm
import polars as pl
from .base import BasePreprocessor, ValuePreprocessor, CodePreprocessor

def fit_preprocessors_jointly(preprocessors: List[BasePreprocessor], event_files: List[str]) -> None:
    """
    Fit multiple preprocessors jointly by reading through the data files only once.
    Handles ValuePreprocessor, CodePreprocessor, and other BasePreprocessor types.
    Fits the preprocessors in place and does not return anything.
    
    Args:
        preprocessors (List[BasePreprocessor]): list of preprocessors to fit
        event_files (List[str]): list of parquet file paths to train on

    Returns:
        None
    """
    # Validate all files exist before processing
    for file_path in event_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Event file not found: {file_path}")
    
    # Separate preprocessors by type
    value_preprocessors = [p for p in preprocessors if isinstance(p, ValuePreprocessor)]
    code_preprocessors = [p for p in preprocessors if isinstance(p, CodePreprocessor)]
    other_preprocessors = [p for p in preprocessors if not isinstance(p, (ValuePreprocessor, CodePreprocessor))]
    
    # Fit code preprocessors (they don't need to read event files for training data)
    for preprocessor in tqdm(code_preprocessors, desc="Fitting code preprocessors", leave=False):
        preprocessor.fit(event_files)
    
    # Fit other preprocessors (they handle their own fitting logic)
    for preprocessor in tqdm(other_preprocessors, desc="Fitting other preprocessors", leave=False):
        preprocessor.fit(event_files)
    
    # If there are no value preprocessors, we're done
    if not value_preprocessors:
        return
    
    # Loop through each event file once and collect data for all value preprocessors
    for event_file in tqdm(event_files, desc="Collecting data for value preprocessors"):
        events = pl.read_parquet(event_file)
        
        for event in events.to_dicts():
            # Check each value preprocessor to see if this event matches its criteria
            for preprocessor in value_preprocessors:
                value = event[preprocessor.value_column]
                if value is not None and preprocessor._match(event["code"]):
                    # attempt to map value to float (in case it is a string), if it fails skip it as its not a valid value
                    try:
                        value = float(value)
                    except:
                        continue
                    # Add the datapoint to this preprocessor's data dictionary
                    if event["code"] not in preprocessor.data:
                        preprocessor.data[event["code"]] = []
                    preprocessor.data[event["code"]].append(value)
    
    # Now fit each value preprocessor to its collected data
    for preprocessor in tqdm(value_preprocessors, desc="Fitting value preprocessors", leave=False):
        preprocessor._fit() 