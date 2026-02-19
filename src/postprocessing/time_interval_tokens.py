import numpy as np
from typing import List, Dict, Any, Optional
import datetime
import polars as pl
from .base import Postprocessor
import numpy as np

class TimeIntervalPostprocessor(Postprocessor):
    def __init__(
        self,
        interval_tokens: Dict[str, Dict[str, int]],
        use_dynamic_bucketing: bool = False,
        wrap_token: bool = True,
        dataset: str = "CPRD",
    ):
        """
        Postprocessor that inserts time interval tokens between events in the event_list.

        Args:
            interval_tokens: Dictionary containing the interval tokens and their minimum and maximum values in minutes.
                            Ignored if use_dynamic_bucketing is True.
            use_dynamic_bucketing: If True, uses dynamic bucketing:
                                   - < 1 week: buckets into days (1d, 2d, 3d, etc.)
                                   - >= 1 week: buckets into weeks (1w, 2w, 32w, 45w, etc.)
        """
        super().__init__()
        self.use_dynamic_bucketing = use_dynamic_bucketing
        self.wrap_token = wrap_token
        self.dataset = (dataset or "CPRD").strip().upper()
        if self.dataset not in {"CPRD", "MIMIC"}:
            raise ValueError(f"Unsupported dataset mode: {dataset}. Expected 'CPRD' or 'MIMIC'.")

        # Always define this attribute for safety
        self.interval_tokens: Dict[str, Dict[str, datetime.timedelta]] = {}

        if not use_dynamic_bucketing:
            # Store interval-specific parameters, converting minutes to timedelta objects
            for interval_name, interval_info in interval_tokens.items():
                converted_info = {}
                if "min" in interval_info:
                    converted_info["min"] = datetime.timedelta(minutes=interval_info["min"])
                if "max" in interval_info:
                    converted_info["max"] = datetime.timedelta(minutes=interval_info["max"])
                self.interval_tokens[interval_name] = converted_info
    
    def _get_dynamic_bucket_name(self, time_delta: datetime.timedelta) -> Optional[str]:
        """
        Calculate dynamic bucket name based on time delta.
        
        Args:
            time_delta: Time difference between events
            
        Returns:
            Bucket name string (e.g., "1d", "3d", "1w", "32w")
        """
        total_seconds = time_delta.total_seconds()
        if total_seconds <= 0:
            return None

        # Dataset-specific dynamic bucketing
        if self.dataset == "MIMIC":
            total_minutes = total_seconds / 60.0
            if total_minutes < 5.0:
                return None  # ignore sub-5min gaps

            if total_minutes < 60.0:
                minutes = int(total_minutes)  # floor
                minutes = max(5, min(59, minutes))
                return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"

            total_hours = total_seconds / 3600.0
            if total_hours < 24.0:
                hours = int(total_hours)  # floor
                hours = max(1, min(23, hours))
                return f"{hours} hour" if hours == 1 else f"{hours} hours"

            total_days = total_seconds / 86400.0
            if total_days < 7.0:
                days = int(total_days)  # floor
                days = max(1, min(6, days))
                return f"{days} day" if days == 1 else f"{days} days"

            weeks = int(total_days / 7.0)  # floor
            weeks = max(1, weeks)
            return f"{weeks} week" if weeks == 1 else f"{weeks} weeks"

        # Default CPRD behavior (keep as-is): no token for sub-day gaps, then days (<7), else weeks
        total_days = total_seconds / 86400.0
        if total_days < 1.0:
            return None

        if total_days < 7.0:
            days = int(round(total_days))
            days = max(1, min(6, days))
            return f"{days} day" if days == 1 else f"{days} days"

        weeks = int(round(total_days / 7.0))
        weeks = max(1, weeks)
        return f"{weeks} week" if weeks == 1 else f"{weeks} weeks"

    
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

                if self.use_dynamic_bucketing:
                    # Use dynamic bucketing
                    interval_name = self._get_dynamic_bucket_name(time_delta)
                    if interval_name is not None:
                        code = f"<time_interval_{interval_name}>" if self.wrap_token else f"TIME {interval_name}"
                        interval_token = {
                            'code': code,
                            'timestamp': current_timestamp,
                            'numeric_value': None,
                            'text_value': None,
                            'unit': None
                        }
                        new_event_list.append(interval_token)
                    # else: no interval token for sub-day gaps
                else:
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
                            code = f"<time_interval_{interval_name}>" if self.wrap_token else interval_name
                            interval_token = {
                                'code': code,
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