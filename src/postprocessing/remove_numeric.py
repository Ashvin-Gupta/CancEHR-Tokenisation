import re
from typing import List, Dict, Any
from .base import Postprocessor

class RemoveNumericPostprocessor(Postprocessor):
    """
    A postprocessor that removes events where numeric_value or text_value 
    contains purely numeric strings (e.g., "123", "45.6", "-10.5").
    
    This is a safety net to catch any numeric values that weren't properly binned
    during preprocessing. In theory, if the data was clean, there would be no
    filtering here, but sometimes events such as "non-drinker" can have a 
    numerical value associated with them for some reason.
    
    Events with non-numeric text values (e.g., "low", "normal", "high", "Q1") 
    are preserved.
    """
    
    def __init__(self):
        super().__init__()
        # Pattern to match purely numeric strings (integers, floats, negative numbers)
        self.numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        print("RemoveNumericPostprocessor initialized: Will remove events with purely numeric values")
    
    def _is_purely_numeric(self, value: Any) -> bool:
        """
        Check if a value is purely numeric (contains only digits, decimal point, and optional minus sign)
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value is purely numeric, False otherwise
        """
        if value is None:
            return False
        
        value_str = str(value).strip()
        if not value_str:
            return False
        
        # Check if it matches the numeric pattern
        return bool(self.numeric_pattern.match(value_str))
    
    def _encode(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove events where numeric_value or text_value contains purely numeric strings
        
        Args:
            datapoint: Dictionary containing subject_id and event_list
            
        Returns:
            Dictionary with updated event_list containing filtered events
        """
        if "event_list" not in datapoint:
            return datapoint
        
        event_list = datapoint["event_list"]
        filtered_events = []
        removed_count = 0
        
        for event in event_list:
            should_remove = False
            
            # Check if numeric_value is purely numeric
            if event.get("numeric_value") is not None:
                if self._is_purely_numeric(event["numeric_value"]):
                    should_remove = True
                    removed_count += 1
            
            # Check if text_value is purely numeric
            if not should_remove and event.get("text_value") is not None:
                if self._is_purely_numeric(event["text_value"]):
                    should_remove = True
                    removed_count += 1
            
            # Keep the event if it shouldn't be removed
            if not should_remove:
                filtered_events.append(event)
        
        # Update the datapoint
        result = datapoint.copy()
        result["event_list"] = filtered_events
        
        return result

if __name__ == "__main__":
    # Test the postprocessor
    postprocessor = RemoveNumericPostprocessor()
    
    # Create test data
    test_datapoint = {
        "subject_id": 12345,
        "event_list": [
            {"code": "LAB//123", "numeric_value": "123", "text_value": None, "timestamp": None},
            {"code": "LAB//456", "numeric_value": None, "text_value": "456.7", "timestamp": None},
            {"code": "LAB//789", "numeric_value": "low", "text_value": None, "timestamp": None},  # Should keep
            {"code": "LAB//012", "numeric_value": None, "text_value": "normal", "timestamp": None},  # Should keep
            {"code": "LAB//345", "numeric_value": "Q1", "text_value": None, "timestamp": None},  # Should keep
            {"code": "LAB//678", "numeric_value": "-10.5", "text_value": None, "timestamp": None},  # Should remove
            {"code": "LAB//901", "numeric_value": None, "text_value": "AGE_20-24", "timestamp": None},  # Should keep
        ]
    }
    
    print("\nBefore filtering:")
    for i, event in enumerate(test_datapoint["event_list"]):
        print(f"  {i}: code={event['code']}, numeric={event.get('numeric_value')}, text={event.get('text_value')}")
    
    # Apply postprocessor
    result = postprocessor._encode(test_datapoint)
    
    print("\nAfter filtering:")
    for i, event in enumerate(result["event_list"]):
        print(f"  {i}: code={event['code']}, numeric={event.get('numeric_value')}, text={event.get('text_value')}")
    
    print("\nRemoveNumericPostprocessor test completed!")

