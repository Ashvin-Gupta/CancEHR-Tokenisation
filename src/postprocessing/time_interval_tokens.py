import numpy as np
from typing import List, Dict, Any
import datetime
import polars as pl
from .base import Postprocessor
import numpy as np

class TimeIntervalPostprocessor(Postprocessor):
    def __init__(self, interval_tokens: Dict[str, Dict[str, int]]):
        """
        Postprocessor that inserts time interval tokens between events in the event_list.

        Args:
            interval_tokens: Dictionary containing the interval tokens and their minimum and maximum values in minutes.
        """
        super().__init__()

        # Store interval-specific parameters, converting minutes to timedelta objects
        self.interval_tokens = {}
        for interval_name, interval_info in interval_tokens.items():
            converted_info = {}
            if "min" in interval_info:
                converted_info["min"] = datetime.timedelta(minutes=interval_info["min"])
            if "max" in interval_info:
                converted_info["max"] = datetime.timedelta(minutes=interval_info["max"])
            self.interval_tokens[interval_name] = converted_info
    
    def _encode(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode time intervals between events in the event_list.
        
        Args:
            datapoint: Dictionary containing subject_id and event_list
            
        Returns:
            Dictionary with updated event_list containing time interval tokens
        """
        event_list = datapoint["event_list"]
        new_event_list = []
        
        # Add the first event
        if event_list:
            new_event_list.append(event_list[0])
        
        # Process subsequent events
        for current_event in event_list[1:]:
            current_timestamp = current_event["timestamp"]
            
            # Skip events with None timestamps for time interval calculation
            if current_timestamp is None:
                new_event_list.append(current_event)
                continue
            
            # Find the last event with a non-None timestamp
            last_timestamp = None
            for prev_event in reversed(new_event_list):
                if prev_event["timestamp"] is not None and prev_event["code"] != "MEDS_BIRTH":
                    last_timestamp = prev_event["timestamp"]
                    break
            
            # If we found a previous timestamp, calculate time difference
            if last_timestamp is not None:
                time_delta = current_timestamp - last_timestamp
                
                # Check if time delta matches any interval
                for interval_name, interval_info in self.interval_tokens.items():
                    min_time = interval_info["min"]
                    max_time = interval_info.get("max", None)
                    
                    if max_time is None:
                        # Open-ended interval (e.g., "1h-")
                        condition = time_delta >= min_time
                    else:
                        # Bounded interval (e.g., "5m-15m")
                        condition = min_time <= time_delta <= max_time
                    
                    if condition:
                        # Insert time interval token
                        interval_token = {
                            'code': f'<time_interval_{interval_name}>',
                            'timestamp': current_timestamp,
                            'numeric_value': None,
                            'text_value': None,
                            'unit': None
                        }
                        new_event_list.append(interval_token)
                        break
            
            # Add the current event
            new_event_list.append(current_event)
        
        # Update the datapoint
        datapoint["event_list"] = new_event_list
        return datapoint

if __name__ == "__main__":

    # Example time intervals (now using integer minutes)
    time_intervals = {
        "5m-15m": {
            "min": 5,
            "max": 15
        },
        "15m-30m": {
            "min": 15,
            "max": 30
        },
        "30m-1h": {
            "min": 30,
            "max": 60
        },
        "1h-": {
            "min": 60
        }
    }

    TIPT = TimeIntervalPostprocessor(interval_tokens=time_intervals)

    # Test with realistic data
    datapoint = {
        'subject_id': 123,
        'event_list': [
            {'code': 'code1', 'timestamp': None, 'numeric_value': None, 'text_value': None},
            {'code': 'code2', 'timestamp': datetime.datetime(2081, 1, 1, 0, 0), 'numeric_value': None, 'text_value': None},
            {'code': 'code3', 'timestamp': datetime.datetime(2129, 3, 16, 0, 0), 'numeric_value': None, 'text_value': None},
            {'code': 'code4', 'timestamp': datetime.datetime(2129, 3, 16, 23, 40), 'numeric_value': None, 'text_value': None},
            {'code': 'code5', 'timestamp': datetime.datetime(2129, 3, 17, 0, 58), 'numeric_value': None, 'text_value': None},
            {'code': 'code6', 'timestamp': datetime.datetime(2129, 3, 17, 5, 46), 'numeric_value': None, 'text_value': None}
        ]
    }

    print("Original datapoint:")
    for i, event in enumerate(datapoint["event_list"]):
        print(f"  {i}: {event['code']} - {event['timestamp']}")

    encoded_datapoint = TIPT._encode(datapoint)

    print("\nEncoded datapoint:")
    for i, event in enumerate(encoded_datapoint["event_list"]):
        print(f"  {i}: {event['code']} - {event['timestamp']}")