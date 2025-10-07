import polars as pl
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from .base import BasePreprocessor
from tqdm import tqdm

class EthosQuantileAgePreprocessor(BasePreprocessor):
    """
    A preprocessor that converts MEDS_BIRTH events into two quantile-based age tokens
    using the Ethos age encoding algorithm. This is an implementation taken from the ETHOS paper (https://www.nature.com/articles/s41746-024-01235-0)
    
    The algorithm provides fine-grained age representation by:
    1. Calculating patient age from MEDS_BIRTH timestamp to reference time
    2. Scaling age to quantile space: age_scaled = age_years * num_quantiles² / 100
    3. Splitting into two components: age_t1 and age_t2
    4. Generating two tokens: Q(age_t1+1) and Q(age_t2+1)
    
    Args:
        matching_type (str): Not used for this preprocessor
        matching_value (str): Not used for this preprocessor  
        reference_time (Union[str, datetime]): Reference time for age calculation ("current" or specific date)
        num_quantiles (int): Number of quantile bins (default: 10)
        prefix (str): Optional prefix for age tokens (default: "AGE_")
        insert_t1_code (bool): Whether to insert code for first age component (default: True)
        insert_t2_code (bool): Whether to insert code for second age component (default: True)
    """
    
    def __init__(self, matching_type: str, matching_value: str, time_unit: str = "years",
                 num_quantiles: int = 10, prefix: str = "AGE_", 
                 insert_t1_code: bool = True, insert_t2_code: bool = True,
                 keep_meds_birth: bool = False):
        # Initialize base class attributes (not used for matching but required for interface)
        self.matching_type = matching_type
        self.matching_value = matching_value
        
        self.time_unit = time_unit
        self.num_quantiles = num_quantiles
        self.prefix = prefix
        self.insert_t1_code = insert_t1_code
        self.insert_t2_code = insert_t2_code
        self.keep_meds_birth = keep_meds_birth
        
        # Validate parameters
        if num_quantiles < 2:
            raise ValueError("num_quantiles must be at least 2")
        if time_unit not in ["years", "days", "hours"]:
            raise ValueError("time_unit must be one of: years, days, hours")
        
        print(f"EthosQuantileAgePreprocessor initialized:")
        print(f"  Time unit: {time_unit}")
        print(f"  Quantiles: {num_quantiles} (providing {num_quantiles**2} age representations)")
        print(f"  Prefix: '{prefix}'")
        print(f"  Insert codes: T1={insert_t1_code}, T2={insert_t2_code}")
        print(f"  Keep MEDS_BIRTH: {keep_meds_birth}")
    
    def fit(self, event_files: List[str]) -> None:
        """
        This preprocessor doesn't need training data as it uses a deterministic algorithm.
        The quantile mapping is based on the mathematical formula, not data distribution.
        
        Args:
            event_files (List[str]): Not used for this preprocessor
        """
        print("EthosQuantileAgePreprocessor fit complete (no training required - uses deterministic algorithm)")
    
    def _calculate_age_from_timeline(self, subject_events: pl.DataFrame) -> float:
        """
        Calculate age from the time delta between MEDS_BIRTH and first real medical event
        
        Args:
            subject_events (pl.DataFrame): All events for a single subject
            
        Returns:
            float: Age in specified time units
        """
        # Find MEDS_BIRTH event
        birth_events = subject_events.filter(pl.col("code") == "MEDS_BIRTH")
        if birth_events.is_empty():
            return 0.0  # No birth event found
        
        birth_time = birth_events["time"].to_list()[0]
        if birth_time is None:
            return 0.0  # No birth timestamp
        
        # Find first real medical event (not demographics, not birth, not null time)
        real_events = subject_events.filter(
            (pl.col("code") != "MEDS_BIRTH") &
            (pl.col("time").is_not_null()) &
            (~pl.col("code").str.starts_with("DEMOGRAPHICS//")) &
            (~pl.col("code").str.starts_with("RACE//")) &
            (~pl.col("code").str.starts_with("MARITAL_STATUS//")) &
            (pl.col("code") != "STATIC_DATA_NO_CODE")
        ).sort("time")
        
        if real_events.is_empty():
            return 0.0  # No real medical events found
        
        first_medical_time = real_events["time"].to_list()[0]
        if first_medical_time is None:
            return 0.0  # No timestamp on first medical event
        
        # Calculate time delta
        if hasattr(birth_time, 'timestamp'):
            birth_timestamp = birth_time.timestamp()
        else:
            birth_timestamp = birth_time
            
        if hasattr(first_medical_time, 'timestamp'):
            medical_timestamp = first_medical_time.timestamp()
        else:
            medical_timestamp = first_medical_time
        
        time_delta_seconds = medical_timestamp - birth_timestamp
        
        # Convert to specified time unit
        if self.time_unit == "years":
            return time_delta_seconds / (365.25 * 24 * 3600)
        elif self.time_unit == "days":
            return time_delta_seconds / (24 * 3600)
        elif self.time_unit == "hours":
            return time_delta_seconds / 3600
        
        return max(0.0, time_delta_seconds / (365.25 * 24 * 3600))  # Default to years
    
    def _encode_age_to_quantiles(self, age_years: float) -> tuple[str, str]:
        """
        Apply the Ethos quantile age encoding algorithm
        
        Args:
            age_years (float): Patient age in years
            
        Returns:
            tuple[str, str]: (age_t1_token, age_t2_token)
        """
        # Step 1: Scale age to quantile space
        age_scaled = age_years * (self.num_quantiles ** 2) / 100
        age_scaled = min(age_scaled, self.num_quantiles ** 2 - 1)  # Cap at max value
        
        # Step 2: Split into two components
        age_t1 = int(age_scaled // self.num_quantiles)  # floor
        age_t2 = int(round(age_scaled % self.num_quantiles))  # round
        
        # Step 3: Handle edge case
        if age_t2 == self.num_quantiles:
            age_t1 += 1
            age_t2 = 0
        
        # Step 4: Generate tokens (1-indexed)
        token1 = f"Q{age_t1 + 1}"
        token2 = f"Q{age_t2 + 1}"
        
        return token1, token2
    
    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Replace MEDS_BIRTH events with two quantile age events based on timeline analysis
        
        Args:
            events (pl.DataFrame): Original events DataFrame
            
        Returns:
            pl.DataFrame: Events DataFrame with MEDS_BIRTH replaced by age quantiles
        """
        # Find MEDS_BIRTH events
        birth_events = events.filter(pl.col("code") == "MEDS_BIRTH")
        
        # Decide whether to keep or remove MEDS_BIRTH events
        if self.keep_meds_birth:
            non_birth_events = events  # Keep all events including MEDS_BIRTH
        else:
            non_birth_events = events.filter(pl.col("code") != "MEDS_BIRTH")  # Remove MEDS_BIRTH
        
        if birth_events.is_empty():
            print("No MEDS_BIRTH events found to process")
            return events
        
        print(f"Processing {len(birth_events)} MEDS_BIRTH events using dynamic age calculation")
        
        # Process each subject separately to calculate age from their timeline
        age_events = []
        processed_subjects = set()
        
        for subject_id in events["subject_id"].unique():
            if subject_id in processed_subjects:
                continue
            processed_subjects.add(subject_id)
            
            # Get all events for this subject
            subject_events = events.filter(pl.col("subject_id") == subject_id)
            
            # Find MEDS_BIRTH event for this subject
            subject_birth_events = subject_events.filter(pl.col("code") == "MEDS_BIRTH")
            if subject_birth_events.is_empty():
                continue  # No birth event for this subject
            
            # Calculate age from timeline
            age_in_time_units = self._calculate_age_from_timeline(subject_events)
            age_t1_token, age_t2_token = self._encode_age_to_quantiles(age_in_time_units)
            
            # Get the birth event to use as template
            birth_event = subject_birth_events.to_dicts()[0]
            
            # Create two age events with same schema as original
            base_event = dict(birth_event)
            
            # Create age_t1 event
            if self.insert_t1_code:
                t1_code = f"{self.prefix}T1"
                t1_value = age_t1_token
            else:
                t1_code = "STATIC_DATA_NO_CODE"
                t1_value = f"{self.prefix}T1//{age_t1_token}" if self.prefix else f"T1//{age_t1_token}"
            
            age_t1_event = base_event.copy()
            age_t1_event["code"] = t1_code
            age_t1_event["text_value"] = t1_value
            age_t1_event["numeric_value"] = None
            age_t1_event["time"] = None  # Set timestamp to None for static age data
            
            # Create age_t2 event  
            if self.insert_t2_code:
                t2_code = f"{self.prefix}T2"
                t2_value = age_t2_token
            else:
                t2_code = "STATIC_DATA_NO_CODE"
                t2_value = f"{self.prefix}T2//{age_t2_token}" if self.prefix else f"T2//{age_t2_token}"
            
            age_t2_event = base_event.copy()
            age_t2_event["code"] = t2_code
            age_t2_event["text_value"] = t2_value
            age_t2_event["numeric_value"] = None
            age_t2_event["time"] = None  # Set timestamp to None for static age data
            
            age_events.extend([age_t1_event, age_t2_event])
        
        # Convert age events to DataFrame if any were created
        if age_events:
            age_df = pl.DataFrame(age_events)
            
            # Ensure schema compatibility
            age_df = age_df.select(events.columns)
            for col in events.columns:
                original_dtype = events.select(col).dtypes[0]
                age_df = age_df.with_columns(pl.col(col).cast(original_dtype))
            
            # Combine with non-birth events
            combined_events = pl.concat([age_df, non_birth_events], how="vertical")
        else:
            combined_events = non_birth_events
        
        # Sort by subject_id and time
        combined_events = combined_events.sort(["subject_id", "time"])
        
        if self.keep_meds_birth:
            print(f"Added {len(age_events)} age quantile events (kept {len(birth_events)} MEDS_BIRTH events)")
        else:
            print(f"Replaced {len(birth_events)} MEDS_BIRTH events with {len(age_events)} age quantile events")
        
        return combined_events


if __name__ == "__main__":
    # Test the preprocessor
    from datetime import datetime
    
    preprocessor = EthosQuantileAgePreprocessor(
        matching_type="",  # Not used
        matching_value="", # Not used
        time_unit="years",
        num_quantiles=10,
        prefix="AGE_",
        insert_t1_code=False,  # Test with codes disabled
        insert_t2_code=False,
        keep_meds_birth=True  # Test keeping MEDS_BIRTH tokens
    )
    
    # Test age encoding
    test_ages = [0, 25, 47, 65, 85, 100]
    print("Testing age encoding:")
    for age in test_ages:
        t1, t2 = preprocessor._encode_age_to_quantiles(age)
        print(f"Age {age:3.0f} → T1={t1}, T2={t2}")
    
    print("\nEthosQuantileAgePreprocessor test completed!")
