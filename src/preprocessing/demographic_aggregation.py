import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from .base import BasePreprocessor
from tqdm import tqdm

class DemographicAggregationPreprocessor(BasePreprocessor):
    """
    A preprocessor that aggregates repeated demographic measurements (BMI, Height, Weight, etc.)
    into single demographic tokens using quantile binning.
    
    For each measurement type:
    1. Finds all matching tokens in patient timeline
    2. Extracts numeric values from specified column
    3. Aggregates using specified method (mean, min, max, median)
    4. Bins aggregated value using quantile bins fitted to training data
    5. Inserts demographic event with timestamp=null
    6. Optionally removes original measurement tokens
    
    Args:
        matching_type (str): Not used for this preprocessor
        matching_value (str): Not used for this preprocessor
        measurements (List[Dict]): List of measurement configurations
    """
    
    def __init__(self, matching_type: str, matching_value: str, measurements: List[Dict[str, Any]]):
        # Initialize base class attributes (not used for matching but required for interface)
        self.matching_type = matching_type
        self.matching_value = matching_value
        
        self.measurements = measurements
        self.quantile_bins = {}  # Will store quantile bins for each measurement
        
        # Validate measurement configurations
        self._validate_measurements()
        
        print(f"DemographicAggregationPreprocessor initialized with {len(measurements)} measurements:")
        for i, measurement in enumerate(measurements):
            token_prefix = measurement.get('token_prefix', 'NO_PREFIX')
            print(f"  {i+1}. {measurement['token_pattern']} → {token_prefix}Qx "
                  f"({measurement['aggregation']}, {measurement['num_bins']} bins)")
    
    def _validate_measurements(self):
        """Validate measurement configurations"""
        required_fields = ["token_pattern", "value_column", "aggregation", "num_bins"]
        valid_aggregations = ["mean", "min", "max", "median"]
        valid_value_columns = ["text_value", "numeric_value"]
        
        for i, measurement in enumerate(self.measurements):
            # Check required fields
            for field in required_fields:
                if field not in measurement:
                    raise ValueError(f"Measurement {i}: missing required field '{field}'")
            
            # Validate aggregation method
            if measurement["aggregation"] not in valid_aggregations:
                raise ValueError(f"Measurement {i}: aggregation must be one of {valid_aggregations}")
            
            # Validate value column
            if measurement["value_column"] not in valid_value_columns:
                raise ValueError(f"Measurement {i}: value_column must be one of {valid_value_columns}")
            
            # Validate num_bins
            if measurement["num_bins"] < 2:
                raise ValueError(f"Measurement {i}: num_bins must be at least 2")
            
            bin_labels = measurement.get("bin_labels")
            if bin_labels is not None:
                if len(bin_labels) != measurement["num_bins"]:
                    raise ValueError(f"Measurement {i}: bin_labels must have length {measurement['num_bins']}")
                
    
    def fit(self, event_files: List[str]) -> None:
        """
        Fit quantile bins for each measurement type based on training data
        
        Args:
            event_files (List[str]): List of parquet file paths to train on
        """
        print(f"Fitting DemographicAggregationPreprocessor on {len(event_files)} files...")
        
        # Collect aggregated values for each measurement type
        measurement_data = {i: [] for i in range(len(self.measurements))}
        
        for file_path in tqdm(event_files, desc="Collecting demographic measurement data"):
            events = pl.read_parquet(file_path)
            
            # Process each subject in this file
            for subject_id in events["subject_id"].unique():
                subject_events = events.filter(pl.col("subject_id") == subject_id)
                
                # Process each measurement type for this subject
                for measurement_idx, measurement in enumerate(self.measurements):
                    aggregated_value = self._aggregate_subject_measurement(subject_events, measurement)
                    if aggregated_value is not None:
                        measurement_data[measurement_idx].append(aggregated_value)
        
        # Compute quantile bins for each measurement
        for measurement_idx, measurement in enumerate(self.measurements):
            values = measurement_data[measurement_idx]
            if len(values) > 0:
                values_array = np.array(values)
                num_bins = measurement["num_bins"]
                
                # Compute quantile edges
                probs = np.linspace(0, 1, num_bins + 1)
                edges = np.quantile(values_array, probs)
                
                self.quantile_bins[measurement_idx] = edges
                
                token_prefix = measurement.get('token_prefix', 'NO_PREFIX')
                print(f"  {token_prefix}Qx: fitted {num_bins} bins from {len(values)} aggregated values")
            else:
                token_prefix = measurement.get('token_prefix', 'NO_PREFIX')
                print(f"  Warning: No valid values found for {token_prefix}Qx")
                self.quantile_bins[measurement_idx] = None
    
    def _aggregate_subject_measurement(self, subject_events: pl.DataFrame, measurement: Dict[str, Any]) -> Optional[float]:
        """
        Find and aggregate measurement values for a single subject
        
        Args:
            subject_events (pl.DataFrame): All events for one subject
            measurement (Dict): Measurement configuration
            
        Returns:
            Optional[float]: Aggregated value or None if no valid measurements found
        """
        # Find matching measurement tokens
        matching_events = subject_events.filter(pl.col("code").str.starts_with(measurement["token_pattern"]))
        
        if matching_events.is_empty():
            return None
        
        # Extract values from specified column
        values = []
        value_column = measurement["value_column"]
        
        for event in matching_events.iter_rows(named=True):
            value = event[value_column]
            if value is not None:
                try:
                    # Convert to float
                    numeric_value = float(value)
                    values.append(numeric_value)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric values
        
        if not values:
            return None
        
        # Apply aggregation
        aggregation = measurement["aggregation"]
        if aggregation == "mean":
            return np.mean(values)
        elif aggregation == "min":
            return np.min(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "median":
            return np.median(values)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def _bin_value(self, value: float, measurement_idx: int) -> Optional[str]:
        """
        Bin a value using the fitted quantile bins
        
        Args:
            value (float): Value to bin
            measurement_idx (int): Index of measurement configuration
            
        Returns:
            Optional[str]: Bin label (e.g., "Q3") or None if no bins available
        """
        if measurement_idx not in self.quantile_bins or self.quantile_bins[measurement_idx] is None:
            return None
        
        edges = self.quantile_bins[measurement_idx]
        bin_index = np.digitize(value, edges[1:-1])  # Exclude first and last edges

        measurement = self.measurements[measurement_idx]
        bin_labels = measurement.get("bin_labels")
        if bin_labels is not None:
            return bin_labels[bin_index]
        
        return f"Q{bin_index + 1}"  # 1-indexed bin labels
    
    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Process events to create demographic tokens and optionally remove original measurements
        
        Args:
            events (pl.DataFrame): Original events DataFrame
            
        Returns:
            pl.DataFrame: Events with demographic tokens added and original measurements optionally removed
        """
        print(f"Processing demographic measurements for {len(events['subject_id'].unique())} subjects...")
        
        # Collect demographic events to add
        demographic_events = []
        
        # Process each subject
        for subject_id in events["subject_id"].unique():
            subject_events = events.filter(pl.col("subject_id") == subject_id)
            
            # Process each measurement type for this subject
            for measurement_idx, measurement in enumerate(self.measurements):
                # Calculate aggregated value
                aggregated_value = self._aggregate_subject_measurement(subject_events, measurement)
                
                if aggregated_value is not None:
                    # Bin the aggregated value
                    bin_label = self._bin_value(aggregated_value, measurement_idx)
                    
                    if bin_label is not None:
                        # Create demographic event
                        token_prefix = measurement.get("token_prefix", "")
                        insert_code = measurement.get("insert_code", True)
                        
                        if insert_code:
                            # Code and value as separate tokens
                            code = token_prefix.rstrip("//") if token_prefix else "DEMOGRAPHIC"
                            text_value = bin_label
                        else:
                            # Combined format with full control over prefix
                            code = "STATIC_DATA_NO_CODE"
                            text_value = f"{token_prefix}{bin_label}"
                        
                        # Create demographic event with full MEDS schema
                        demographic_event = {}
                        demographic_event["subject_id"] = subject_id
                        demographic_event["time"] = None  # Static demographic data
                        demographic_event["code"] = code
                        demographic_event["numeric_value"] = None
                        demographic_event["text_value"] = text_value
                        
                        # Fill in all other columns from original events with None
                        for col in events.columns:
                            if col not in demographic_event:
                                demographic_event[col] = None
                        
                        demographic_events.append(demographic_event)
        
        # Create demographic events DataFrame if any were created
        if demographic_events:
            demographic_df = pl.DataFrame(demographic_events)
            
            # Ensure schema compatibility
            demographic_df = demographic_df.select(events.columns)
            for col in events.columns:
                original_dtype = events.select(col).dtypes[0]
                demographic_df = demographic_df.with_columns(pl.col(col).cast(original_dtype))
        else:
            demographic_df = None
        
        # Remove original measurement tokens if requested
        original_events = events
        for measurement in self.measurements:
            if measurement.get("remove_original_tokens", False):
                # Remove all events matching this token pattern
                events = events.filter(pl.col("code") != measurement["token_pattern"])
        
        removed_count = len(original_events) - len(events)
        print(f"Removed {removed_count} original measurement tokens")
        
        # Combine demographic events with (possibly filtered) original events
        if demographic_df is not None:
            if not events.is_empty():
                combined_events = pl.concat([demographic_df, events], how="vertical")
            else:
                combined_events = demographic_df
        else:
            combined_events = events
        
        # Sort by subject_id and time (null values will sort first)
        combined_events = combined_events.sort(["subject_id", "time"])
        
        # Print summary
        print(f"Added {len(demographic_events)} demographic events")
        
        return combined_events


if __name__ == "__main__":
    # Test the preprocessor
    test_measurements = [
        {
            "token_pattern": "BMI (kg/m2)",
            "value_column": "text_value",
            "aggregation": "median",
            "num_bins": 10,
            "token_prefix": "BMI//",
            "insert_code": False,
            "remove_original_tokens": True
        },
        {
            "token_pattern": "Height (Inches)",
            "value_column": "text_value", 
            "aggregation": "max",
            "num_bins": 10,
            "token_prefix": "HEIGHT//",
            "insert_code": False,
            "remove_original_tokens": False
        }
    ]
    
    preprocessor = DemographicAggregationPreprocessor(
        matching_type="",  # Not used
        matching_value="", # Not used
        measurements=test_measurements
    )
    
    print("DemographicAggregationPreprocessor test completed!")
