from src.postprocessing.base import Postprocessor
import pandas as pd
from typing import List, Dict, Any

class NaturalLanguageTranslationPostprocessor(Postprocessor):
    """
    Translates structured medical codes to natural language.
    Should be applied AFTER binning but BEFORE tokenization.
    """
    
    def __init__(self, medical_lookup_filepath1: str, medical_lookup_filepath2: str, 
                 lab_lookup_filepath: str, 
                 region_lookup_filepath: str):
        """
        Args:
            medical_lookup_filepath1: Path to medical lookup CSV for tranlating key events from lit review
            medical_lookup_filepath2: Path to medical lookup CSV for translating Read Codes
            lab_lookup_filepath: Path to lab lookup CSV
            region_lookup_filepath: Path to region lookup CSV
        """
        # Load lookups
        medical_df1 = pd.read_csv(medical_lookup_filepath1)
        self.medical_lookup1 = pd.Series(
            medical_df1['term'].values, 
            index=medical_df1['code'].astype(str).str.upper()
        ).to_dict()
        
        medical_df2 = pd.read_csv(medical_lookup_filepath2)
        self.medical_lookup2 = pd.Series(
            medical_df2['term'].values, 
            index=medical_df2['code'].astype(str).str.upper()
        ).to_dict()
        
        lab_df = pd.read_csv(lab_lookup_filepath)
        self.lab_lookup = pd.Series(
            lab_df['term'].values, 
            index=lab_df['code'].astype(str).str.upper()
        ).to_dict()
        
        region_df = pd.read_csv(region_lookup_filepath)
        self.region_lookup = pd.Series(
            region_df['Description'].values, 
            index=region_df['regionid'].astype(str).str.upper()
        ).to_dict()
    
    def _translate_code(self, code: str, binned_value: str = None) -> str:
        """
        Translate a single code to natural language.
        
        Args:
            code: The structured code (e.g., "MEDICAL//E11.9")
            binned_value: The binned value (e.g., "Q3", "high")
        Returns:
            Natural language string, or None to skip
        """
        if not isinstance(code, str, ):
            return None
        
        try:
            # Time intervals
            if code.startswith('<time_interval_'):
                time_part = code.split('_')[-1].strip('>')
                return time_part
            
            # Age
            elif code.startswith('AGE: '):
                return code
            
            # BMI with binning
            elif code.startswith('MEDICAL//BMI'):
                concept = code.split('//')[1]
                if binned_value and binned_value in ['very low', 'low', 'normal', 'high', 'very high']:
                    return f"{concept} {binned_value}"
                return concept
            
            # Medical codes with potential binning
            elif code.startswith('MEDICAL//'):
                raw_code = code.split('//')[1].upper()
                description = self.medical_lookup1.get(raw_code)
                if description is None:
                    description = self.medical_lookup2.get(raw_code, raw_code.replace('_', ' ').title())

                if 'bp_' in code.lower() and binned_value in ['low', 'normal', 'high']:
                    return f"<EVT> {description} {binned_value}"
                
                return description
            
            # Measurements with binning
            elif code.startswith('MEASUREMENT//'):
                raw_code = code.split('//')[1].upper()
                description = self.medical_lookup1.get(raw_code)
                if description is None:
                    description = self.medical_lookup2.get(raw_code, raw_code.replace('_', ' ').title())
                
                if binned_value and binned_value in ['low', 'normal', 'high']:
                    return f"<EVT> {description} {binned_value}"
                
                return description
            
            # Labs with binning
            elif code.startswith('LAB//'):
                raw_code = code.split('//')[1].upper()
                description = self.lab_lookup.get(raw_code, raw_code.replace('_', ' ').title())

                if binned_value and binned_value in ['low', 'normal', 'high']:
                    return f"<EVT> {description} {binned_value}"
                
                return description
            
            # Gender/Ethnicity
            elif code.startswith(('GENDER//', 'ETHNICITY//')):
                parts = code.split('//')
                return f"{parts[0]} {parts[1]}"
            
            # Region
            elif code.startswith('REGION//'):
                parts = code.split('//')
                return f"{parts[0]} {self.region_lookup.get(parts[1], parts[1])}"
            
            # Quantile indicators (might be standalone or part of previous code)
            elif code.startswith('Q') and len(code) <= 4 and code[1:].isdigit():
                return f"Quantile {code[1:]}"
            
            # Binning indicators (handled as part of previous code)
            elif code in ['low', 'normal', 'high', 'very low', 'very high']:
                # These should be consumed by the previous measurable code
                return None
            
            # Special tokens to skip
            elif code in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return code  # Keep special tokens as-is
            
            else:
                # Unknown format, keep as-is
                return code
                
        except Exception as e:
            print(f"Warning: Failed to translate '{code}': {e}")
            return code

    def _encode(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate codes in a single subject's events to natural language.
        
        Args:
            datapoint: Dict with "subject_id" and "event_list"
            
        Returns:
            Modified datapoint with translated codes
        """
        event_list = datapoint["event_list"]
        
        # Process events, looking ahead for binning indicators
        new_event_list = []
        i = 0
        
        while i < len(event_list):
            event = event_list[i]
            code = event["code"]
            
            # Check if this is a measurable concept that might have binning
            is_measurable = (
                code.startswith('LAB//') or 
                code.startswith('MEASUREMENT//') or 
                code.startswith('MEDICAL//BMI') or
                code.startswith('MEDICAL//bp_')
            )
            
            # Look ahead for binning indicator (n+1)
            binned_value = None
            skip_next = False
            
            if is_measurable and i + 1 < len(event_list):
                next_code = event_list[i + 1]["code"]
                # Check if next code is a binning indicator
                if next_code in ['low', 'normal', 'high', 'very low', 'very high']:
                    binned_value = next_code
                    skip_next = True
            
            # Translate the code (with optional binned value)
            translated = self._translate_code(code, binned_value)
            
            # Only add if translation produced something meaningful
            if translated is not None:
                new_event = event.copy()
                new_event["code"] = translated
                new_event_list.append(new_event)
            
            # Move to next event (skip binning indicator if we consumed it)
            if skip_next:
                i += 2  # Skip both current and next
            else:
                i += 1  # Just move to next
        
        datapoint["event_list"] = new_event_list
        return datapoint
    