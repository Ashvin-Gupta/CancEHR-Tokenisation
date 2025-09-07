from typing import List, Dict, Any
from .base import Postprocessor

class DemographicSortOrderPostprocessor(Postprocessor):
    """
    A postprocessor that sorts demographic tokens at the beginning of each subject's timeline
    according to specified token patterns.
    
    This ensures deterministic ordering of demographic tokens (race, age, BMI, etc.)
    that all have timestamp=null or 0.
    
    Args:
        token_patterns (List[str]): List of token patterns in desired order
    """
    
    def __init__(self, token_patterns: List[str]):
        super().__init__()
        self.token_patterns = token_patterns
        
        # Create pattern to priority mapping
        self.pattern_priorities = {}
        for priority, pattern in enumerate(token_patterns):
            self.pattern_priorities[pattern] = priority
        
        print(f"DemographicSortOrderPostprocessor: Token sort order:")
        for i, pattern in enumerate(token_patterns):
            print(f"  {i+1}. {pattern}")
        print(f"  (Unmatched tokens will have priority {len(token_patterns)})")
    
    def _get_token_priority(self, event: Dict[str, Any]) -> int:
        """
        Get the sort priority for an event based on its text_value
        
        Args:
            event (Dict): Event dictionary
            
        Returns:
            int: Sort priority (lower = earlier in sequence)
        """
        # Check text_value for demographic tokens
        text_value = event.get("text_value", "")
        if text_value:
            for pattern, priority in self.pattern_priorities.items():
                if text_value.startswith(pattern):
                    return priority
        
        # Check code as fallback
        code = event.get("code", "")
        if code:
            for pattern, priority in self.pattern_priorities.items():
                if code.startswith(pattern):
                    return priority
        
        # Token doesn't match any pattern, give it lowest priority (sorts last)
        return len(self.pattern_priorities)
    
    def _is_demographic_event(self, event: Dict[str, Any]) -> bool:
        """
        Check if an event is a demographic event (timestamp is null/0)
        
        Args:
            event (Dict): Event dictionary
            
        Returns:
            bool: True if demographic event
        """
        timestamp = event.get("timestamp", None)
        return timestamp is None or timestamp == 0
    
    def _encode(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sort demographic tokens at the beginning of the event timeline
        
        Args:
            datapoint (Dict): Subject datapoint with event_list
            
        Returns:
            Dict: Subject datapoint with sorted event_list
        """
        if "event_list" not in datapoint:
            return datapoint
        
        event_list = datapoint["event_list"]
        
        # Separate demographic events from regular events
        demographic_events = []
        regular_events = []
        
        for event in event_list:
            if self._is_demographic_event(event):
                demographic_events.append(event)
            else:
                regular_events.append(event)
        
        # Sort demographic events by priority
        demographic_events.sort(key=lambda event: (
            self._get_token_priority(event),  # Primary: demographic priority
            event.get("text_value", ""),      # Secondary: alphabetical for same priority
            event.get("code", "")             # Tertiary: code for tie-breaking
        ))
        
        # Combine sorted demographic events with regular events
        sorted_event_list = demographic_events + regular_events
        
        # Return updated datapoint
        result = datapoint.copy()
        result["event_list"] = sorted_event_list
        
        return result


if __name__ == "__main__":
    # Test the postprocessor
    postprocessor = DemographicSortOrderPostprocessor(
        token_patterns=["BMI//", "GENDER//", "MARITAL_STATUS//", "AGE_T", "RACE//"]
    )
    
    # Create test data
    test_datapoint = {
        "subject_id": 12345,
        "event_list": [
            {"timestamp": None, "code": "STATIC_DATA_NO_CODE", "text_value": "RACE//WHITE"},
            {"timestamp": None, "code": "STATIC_DATA_NO_CODE", "text_value": "AGE_T1//Q3"},
            {"timestamp": None, "code": "STATIC_DATA_NO_CODE", "text_value": "BMI//Q5"},
            {"timestamp": None, "code": "STATIC_DATA_NO_CODE", "text_value": "GENDER//F"},
            {"timestamp": None, "code": "STATIC_DATA_NO_CODE", "text_value": "MARITAL_STATUS//MARRIED"},
            {"timestamp": 1000, "code": "LAB//51237//mg/dL", "text_value": None},
        ]
    }
    
    print("\nBefore sorting:")
    for event in test_datapoint["event_list"]:
        print(f"  {event.get('text_value', event.get('code', 'N/A'))}")
    
    # Apply postprocessor
    result = postprocessor._encode(test_datapoint)
    
    print("\nAfter sorting:")
    for event in result["event_list"]:
        print(f"  {event.get('text_value', event.get('code', 'N/A'))}")
    
    print("\nDemographicSortOrderPostprocessor test completed!")
