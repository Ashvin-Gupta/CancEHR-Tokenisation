import polars as pl
import os
from typing import Dict, List, Any, Optional
from .base import CodePreprocessor
from tqdm import tqdm

class CodeEnrichmentPreprocessor(CodePreprocessor):
    """
    A preprocessor that enriches codes with human-readable descriptions using lookup tables.
    When lookup fails, the original code is kept unchanged.
    
    Args:
        matching_type (str): the type of matching to use. Must be one of "starts_with", "ends_with", "contains", "equals"
        matching_value (str): the value to match.
        lookup_file (str): path to the CSV file containing lookup data
        template (str): template string for the enriched code with placeholders for lookup columns
        code_column (str): column name in the lookup file that contains the code/ID to match
        dtypes (Dict[str, Any], optional): data types to override when reading the CSV file
        additional_filters (Dict[str, Any], optional): additional filters to apply during lookup
    """
    def __init__(self, matching_type: str, matching_value: str, lookup_file: str, 
                 template: str, code_column: str, dtypes: Optional[Dict[str, Any]] = None,
                 additional_filters: Optional[Dict[str, Any]] = None):
        super().__init__(matching_type, matching_value)
        
        self.lookup_file = lookup_file
        self.template = template
        self.code_column = code_column
        self.dtypes = self._convert_dtypes(dtypes or {})
        self.additional_filters = additional_filters or {}
        self.lookup_table: Optional[pl.DataFrame] = None
        self.lookup_cache: Dict[Any, str] = {}  # Phase 1: Add simple lookup cache
    
    def _convert_dtypes(self, dtypes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string dtype specifications to Polars types.
        
        Args:
            dtypes: Dictionary of column names to dtype specifications
            
        Returns:
            Dictionary with converted Polars dtypes
        """
        converted = {}
        for col, dtype in dtypes.items():
            if isinstance(dtype, str):
                if dtype.lower() in ["string", "str", "utf8"]:
                    converted[col] = pl.Utf8
                elif dtype.lower() in ["int", "int64", "integer"]:
                    converted[col] = pl.Int64
                elif dtype.lower() in ["float", "float64"]:
                    converted[col] = pl.Float64
                else:
                    # If we don't recognize it, keep as is
                    converted[col] = dtype
            else:
                converted[col] = dtype
        return converted
    
    def fit(self, event_files: List[str]) -> None:
        """
        Load the lookup table from the CSV file.
        
        Args:
            event_files (List[str]): the list of event files (not used for code enrichment, but required by interface)
        """
        if not os.path.exists(self.lookup_file):
            raise FileNotFoundError(f"Lookup file not found: {self.lookup_file}")
        
        print(f"Loading lookup table from: {self.lookup_file}")
        
        # Read CSV with specified dtypes if provided
        if self.dtypes:
            self.lookup_table = pl.read_csv(self.lookup_file, dtypes=self.dtypes)
        else:
            self.lookup_table = pl.read_csv(self.lookup_file)
        
        # Validate that the code column exists
        if self.code_column not in self.lookup_table.columns:
            raise ValueError(f"Code column '{self.code_column}' not found in lookup file. Available columns: {self.lookup_table.columns}")
        
        print(f"Loaded lookup table with {len(self.lookup_table)} entries")
        
        # Optimization: Pre-filter lookup table during fit
        if self.additional_filters:
            for filter_column, filter_value in self.additional_filters.items():
                self.lookup_table = self.lookup_table.filter(pl.col(filter_column) == filter_value)
            print(f"Applied filters, table now has {len(self.lookup_table)} entries")
        
        # Optimization: Create simple lookup cache
        self._create_simple_cache()
    
    def _create_simple_cache(self):
        """
        Phase 1 Optimization 2: Create a simple dictionary cache for faster lookups
        """
        self.lookup_cache = {}
        
        # Convert to dictionary for O(1) lookups
        for row in tqdm(self.lookup_table.iter_rows(named=True), desc=f"Creating lookup cache for {self.lookup_file}"):
            key = row[self.code_column]
            try:
                enriched = self.template.format(**row)
                self.lookup_cache[key] = enriched
            except KeyError:
                continue  # Skip rows that don't match template
        
        print(f"Created lookup cache with {len(self.lookup_cache)} entries")
    
    def _extract_code_id(self, code: str) -> str:
        """
        Extract the ID/code part from a structured code string.
        For example: 
        - "LAB//51237//unit" -> "51237" (parts[1])
        - "DIAGNOSIS//ICD//10//E7800" -> "E7800" (parts[3])
        
        Args:
            code (str): the full code string
            
        Returns:
            str: the extracted ID
        """
        # Split by "//" and determine which part contains the ID
        parts = code.split("//")
        
        if len(parts) < 2:
            # If no "//" structure, return the whole code
            return code
        
        # Handle different code formats
        if code.startswith("DIAGNOSIS//ICD//"):
            # For ICD codes: DIAGNOSIS//ICD//9//CODE or DIAGNOSIS//ICD//10//CODE
            # The actual ICD code is in the last part (parts[3])
            if len(parts) >= 4:
                return parts[3]
            else:
                return code  # Fallback to original if format is unexpected
        else:
            # For other codes like LAB: LAB//ID//unit
            # The ID is in parts[1]
            return parts[1]
    
    def _transform_code(self, code: str) -> str:
        """
        Transform a code by looking it up in the lookup cache first, then fallback to table lookup.
        If lookup fails, returns the original code unchanged.
        
        Args:
            code (str): the original code
            
        Returns:
            str: the enriched code or original code if lookup fails
        """
        if self.lookup_table is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        # Extract the ID from the code
        code_id = self._extract_code_id(code)
        
        try:
            # Convert code_id to appropriate type for lookup
            if self.lookup_table[self.code_column].dtype == pl.Int64:
                code_id = int(code_id)
            elif self.lookup_table[self.code_column].dtype == pl.Float64:
                code_id = float(code_id)
            # else keep as string (including for pl.Utf8)
        except ValueError:
            # If conversion fails, return original code
            return code
        
        # Check cache first (O(1) lookup)
        if code_id in self.lookup_cache:
            return self.lookup_cache[code_id]
        
        # Fallback to original method for codes not in cache
        # This handles edge cases that might not be in the pre-computed cache
        lookup_result = self.lookup_table.filter(pl.col(self.code_column) == code_id)
        
        if lookup_result.is_empty():
            # Code not found in lookup table, return original
            return code
        
        # Get the first matching row as a dictionary
        row = lookup_result.to_dicts()[0]
        
        # Apply the template with the lookup data
        try:
            enriched_code = self.template.format(**row)
            # Add to cache for future use
            self.lookup_cache[code_id] = enriched_code
            return enriched_code
        except KeyError as e:
            raise ValueError(f"Template references column '{e.args[0]}' which is not in the lookup file. Available columns: {list(row.keys())}")


if __name__ == "__main__":
    # Test the preprocessor with LAB codes
    lab_enricher = CodeEnrichmentPreprocessor(
        matching_type="starts_with",
        matching_value="LAB//",
        lookup_file="/home/joshua/data/mimic/mimic_iv/mimic_iv/3.1/hosp/d_labitems.csv",
        template="<LAB> {label} ({category}) </LAB>",
        code_column="itemid"
    )
    
    # Test fit
    lab_enricher.fit([])  # No event files needed for code enrichment
    
    # Test transform
    test_codes = [
        "LAB//51237//mg/dL",
        "LAB//51274//sec", 
        "LAB//99999//unknown"  # This should fallback to original
    ]
    
    for code in test_codes:
        if lab_enricher._match(code):
            enriched = lab_enricher._transform_code(code)
            print(f"{code} -> {enriched}")
        else:
            print(f"{code} -> no match") 