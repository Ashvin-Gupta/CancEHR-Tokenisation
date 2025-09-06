import polars as pl
import os
from typing import Dict, List, Any, Optional
from .base import BasePreprocessor
from tqdm import tqdm

class LoadStaticDataPreprocessor(BasePreprocessor):
    """
    A preprocessor that loads static data from a CSV file and inserts it as events
    at the beginning of each subject's timeline with time=null.
    
    This is useful for adding demographic data, static patient characteristics,
    or any other subject-level information that doesn't change over time.
    
    Args:
        matching_type (str): Not used for this preprocessor (all subjects get static data)
        matching_value (str): Not used for this preprocessor  
        csv_filepath (str): Path to the CSV file containing static data
        subject_id_column (str): Column name in CSV that contains subject IDs
        columns (List[Dict]): List of column configurations with mappings and validation
    """
    
    def __init__(self, matching_type: str, matching_value: str, csv_filepath: str, 
                 subject_id_column: str, columns: List[Dict[str, Any]]):
        # Initialize base class attributes directly since matching is not used for static data
        self.matching_type = matching_type  # Not used but kept for interface compatibility
        self.matching_value = matching_value  # Not used but kept for interface compatibility
        
        self.csv_filepath = csv_filepath
        self.subject_id_column = subject_id_column
        self.columns = columns
        self.static_data: Optional[pl.DataFrame] = None
        self.subject_lookup: Dict[Any, Dict[str, str]] = {}
        
        # Validate column configurations
        self._validate_column_configs()
    
    def _validate_column_configs(self):
        """Validate that all column configurations have required fields"""
        for col_config in self.columns:
            if "column_name" not in col_config:
                raise ValueError("Each column configuration must have 'column_name'")
            if "code_template" not in col_config:
                raise ValueError("Each column configuration must have 'code_template'")
            # valid_values and map_invalids_to are optional
    
    def fit(self, event_files: List[str]) -> None:
        """
        Load the static data CSV file and create subject lookup table.
        
        Args:
            event_files (List[str]): Not used for static data loading
        """
        if not os.path.exists(self.csv_filepath):
            raise FileNotFoundError(f"Static data CSV file not found: {self.csv_filepath}")
        
        print(f"Loading static data from: {self.csv_filepath}")
        
        # Load CSV file
        self.static_data = pl.read_csv(self.csv_filepath)
        
        # Validate subject ID column exists
        if self.subject_id_column not in self.static_data.columns:
            raise ValueError(f"Subject ID column '{self.subject_id_column}' not found in CSV. Available columns: {self.static_data.columns}")
        
        # Validate all configured columns exist in CSV
        for col_config in self.columns:
            column_name = col_config["column_name"]
            if column_name not in self.static_data.columns:
                raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {self.static_data.columns}")
        
        # Create subject lookup table
        self._create_subject_lookup()
        
        print(f"Loaded static data for {len(self.subject_lookup)} subjects")
        
        # Print statistics about data values
        self._print_data_statistics()
    
    def _create_subject_lookup(self):
        """Create a lookup dictionary mapping subject_id to processed static data"""
        self.subject_lookup = {}
        
        # Handle multiple rows per subject by taking the first occurrence
        # (could be modified to take most recent which may effect columns like marital_status)
        unique_subjects = self.static_data.group_by(self.subject_id_column).first()
        
        for row in tqdm(unique_subjects.iter_rows(named=True), desc="Processing static data"):
            subject_id = row[self.subject_id_column]
            subject_data = {}
            
            for col_config in self.columns:
                column_name = col_config["column_name"]
                raw_value = row[column_name]
                
                # Clean and map the value
                cleaned_value = self._clean_value(raw_value, col_config)
                subject_data[column_name] = cleaned_value
            
            self.subject_lookup[subject_id] = subject_data
    
    def _clean_value(self, value: Any, col_config: Dict[str, Any]) -> str:
        """
        Clean and map a value according to column configuration
        
        Args:
            value: Raw value from CSV
            col_config: Column configuration dictionary
            
        Returns:
            str: Cleaned and mapped value
        """
        if value is None:
            # Handle null values
            return col_config.get("map_invalids_to", "UNKNOWN")
        
        # Convert to string and strip whitespace
        str_value = str(value).strip().upper()
        
        # Apply custom mappings if provided
        mappings = col_config.get("mappings", {})
        if str_value in mappings:
            str_value = mappings[str_value]
        
        # Check if value is in valid_values list (if provided)
        valid_values = col_config.get("valid_values")
        if valid_values is not None:
            if str_value not in valid_values:
                # Value is not valid, map to default
                return col_config.get("map_invalids_to", "UNKNOWN")
        
        return str_value
    
    def _print_data_statistics(self):
        """Print statistics about the loaded static data"""
        print("\n=== Static Data Statistics ===")
        
        for col_config in self.columns:
            column_name = col_config["column_name"]
            print(f"\n{column_name.upper()} distribution:")
            
            # Count occurrences of each value
            value_counts = {}
            for subject_data in self.subject_lookup.values():
                value = subject_data[column_name]
                value_counts[value] = value_counts.get(value, 0) + 1
            
            # Sort by count descending
            for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.subject_lookup)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
    
    def encode_polars(self, events: pl.DataFrame) -> pl.DataFrame:
        """
        Insert static data events at the beginning of each subject's timeline.
        
        Args:
            events (pl.DataFrame): Original events DataFrame
            
        Returns:
            pl.DataFrame: Events DataFrame with static data events inserted
        """
        if self.static_data is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        # Collect new static data events
        static_events = []
        
        # Get unique subject IDs from the events
        unique_subjects = events.select("subject_id").unique().to_series().to_list()
        
        # Track matching statistics
        subjects_found = 0
        subjects_missing = 0
        
        for subject_id in unique_subjects:
            if subject_id in self.subject_lookup:
                subjects_found += 1
                subject_data = self.subject_lookup[subject_id]
                
                # Create events for each configured column
                for col_config in self.columns:
                    column_name = col_config["column_name"]
                    code_template = col_config["code_template"]
                    value = subject_data[column_name]
                    
                    # Create the event with schema matching the original events DataFrame
                    static_event = {}
                    
                    # Apply value prefix if specified
                    final_value = value
                    if "value_prefix" in col_config and col_config["value_prefix"]:
                        final_value = f"{col_config['value_prefix']}{value}"
                    
                    # Determine if code should be inserted
                    insert_code = col_config.get("insert_code", True)  # Default to True for backward compatibility
                    final_code = code_template if insert_code else "STATIC_DATA_NO_CODE"  # Use placeholder that won't match patterns
                    
                    # Set the key demographic fields
                    static_event["subject_id"] = subject_id
                    static_event["time"] = None  # Static data has null timestamp
                    static_event["code"] = final_code
                    static_event["numeric_value"] = None
                    static_event["text_value"] = final_value
                    
                    # Fill in all other columns from the original events with None
                    for col in events.columns:
                        if col not in static_event:
                            static_event[col] = None
                    static_events.append(static_event)
            else:
                subjects_missing += 1
                
                # Create events with default values for each configured column
                for col_config in self.columns:
                    column_name = col_config["column_name"]
                    code_template = col_config["code_template"]
                    default_value = col_config.get("map_invalids_to", "UNKNOWN")
                    
                    # Create the event with default value and schema matching the original events DataFrame
                    static_event = {}
                    
                    # Apply value prefix to default value if specified
                    final_default_value = default_value
                    if "value_prefix" in col_config and col_config["value_prefix"]:
                        final_default_value = f"{col_config['value_prefix']}{default_value}"
                    
                    # Determine if code should be inserted
                    insert_code = col_config.get("insert_code", True)  # Default to True for backward compatibility
                    final_code = code_template if insert_code else "STATIC_DATA_NO_CODE"  # Use placeholder that won't match patterns
                    
                    # Set the key demographic fields
                    static_event["subject_id"] = subject_id
                    static_event["time"] = None  # Static data has null timestamp
                    static_event["code"] = final_code
                    static_event["numeric_value"] = None
                    static_event["text_value"] = final_default_value
                    
                    # Fill in all other columns from the original events with None
                    for col in events.columns:
                        if col not in static_event:
                            static_event[col] = None
                    static_events.append(static_event)
        
        # Print summary statistics instead of individual warnings
        total_subjects = len(unique_subjects)
        match_percentage = (subjects_found / total_subjects) * 100 if total_subjects > 0 else 0
        print(f"Static data matching: {subjects_found}/{total_subjects} subjects found ({match_percentage:.1f}%), {subjects_missing} using default values")
        
        if not static_events:
            print("No static events to insert")
            return events
        
        # Convert static events to DataFrame with proper schema matching
        if static_events:
            static_df = pl.DataFrame(static_events)
            
            # Ensure columns are in the same order as the original events DataFrame
            static_df = static_df.select(events.columns)
            
            # Ensure schema compatibility by casting columns to match original events
            for col in events.columns:
                original_dtype = events.select(col).dtypes[0]
                static_df = static_df.with_columns(pl.col(col).cast(original_dtype))
            
            # Combine with original events
            combined_events = pl.concat([static_df, events], how="vertical")
        else:
            combined_events = events
        
        # Sort by subject_id and time (null values will sort first)
        combined_events = combined_events.sort(["subject_id", "time"])
        
        print(f"Inserted {len(static_events)} static data events")
        
        return combined_events


if __name__ == "__main__":
    # Test the preprocessor with demographics data
    demographics_preprocessor = LoadStaticDataPreprocessor(
        matching_type="",  # Not used
        matching_value="", # Not used
        csv_filepath="/home/joshua/data/mimic/mimic_iv/mimic_iv/3.1/hosp/admissions.csv",
        subject_id_column="subject_id",
        columns=[
            {
                "column_name": "race",
                "code_template": "DEMOGRAPHICS//RACE",
                "valid_values": ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"],
                "mappings": {
                    "PORTUGUESE": "WHITE",
                    "MULTIPLE RACE/ETHNICITY": "OTHER"
                },
                "map_invalids_to": "RACE_UNKNOWN",
                "value_prefix": "RACE//",
                "insert_code": False  # Only insert "RACE//WHITE", not "DEMOGRAPHICS//RACE"
            },
            {
                "column_name": "marital_status",
                "code_template": "DEMOGRAPHICS//MARITAL_STATUS", 
                "valid_values": ["SINGLE", "WIDOWED", "MARRIED", "DIVORCED"],
                "map_invalids_to": "MARITAL_STATUS_UNKNOWN",
                "value_prefix": "MARITAL_STATUS//",
                "insert_code": True   # Insert both "DEMOGRAPHICS//MARITAL_STATUS" and "MARITAL_STATUS//MARRIED"
            }
        ]
    )
    
    # Test fit
    demographics_preprocessor.fit([])
    
    print("Demographics preprocessor test completed successfully!")
