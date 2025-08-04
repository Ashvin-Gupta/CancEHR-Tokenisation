from typing import List, Dict, Any
import polars as pl
from src.preprocessing.base import Preprocessor
from abc import ABC, abstractmethod
import datetime

class Tokenizer(ABC):
    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.unknown_token = "<unknown>"
        self.vocab = pl.DataFrame(schema={"token": pl.Int64, "str": pl.String, "count": pl.Int64})
        self.special_tokens = {
            self.unknown_token: 0,
            "<start>": 1,
            "<end>": 2,
        }

    def _process_events(self, events: pl.DataFrame) -> List[str]:
        """Convert events DataFrame into list of strings for tokenization"""
        processed_events = []
        
        for subject_id, subject_events in events.group_by("subject_id", maintain_order=True):

            event_list = []
            for event in subject_events.to_dicts():
                event = {
                    "code": event["code"],
                    "timestamp": event["time"],
                    "numeric_value": event["numeric_value"],
                    "text_value": event["text_value"]
                }
                event_list.append(event)

            processed_events.append(
                {
                    "subject_id": subject_id,
                    "event_list": event_list,
                }
            )

        return processed_events
    
    def _events_to_lists(self, events: List[dict]) -> Dict[str, List[str]]:
        """
        Convert a list of events to a string.

        Args:
            events (List[dict]): the events to convert to a string

        Returns:
            str: the string representation of the events
        """
        result = {
            "strings": [],
            "timestamps": []
        }

        def format_timestamp(timestamp: Any) -> float:
            """Format a timestamp to a float"""
            if timestamp is None:
                return 0
            elif isinstance(timestamp, datetime.datetime):
                return timestamp.timestamp()
            else:
                return timestamp

        # Loop through each subject in the events list
        for subject in events:

            strings = ["<start>"]
            timestamps = [0]

            for event in subject["event_list"]:

                # Add event start token if specified
                if self.insert_event_tokens:
                    strings.append("<event>")
                    timestamps.append(format_timestamp(event["timestamp"]))
                
                # Add the code
                strings.append(event["code"])
                timestamps.append(format_timestamp(event["timestamp"]))
                
                # Add the numeric value if specified
                if event["numeric_value"] is not None:
                    if self.insert_numeric_tokens:
                        strings.append("<numeric>")
                        timestamps.append(format_timestamp(event["timestamp"]))

                        strings.append(str(event["numeric_value"]))
                        timestamps.append(format_timestamp(event["timestamp"]))

                        strings.append("</numeric>")
                        timestamps.append(format_timestamp(event["timestamp"]))
                    else:
                        strings.append(str(event["numeric_value"]))
                        timestamps.append(format_timestamp(event["timestamp"]))
                
                # Add the text value if specified
                if event["text_value"] is not None:
                    if self.insert_text_tokens:
                        strings.append("<text>")
                        timestamps.append(format_timestamp(event["timestamp"]))

                        strings.append(str(event["text_value"]))
                        timestamps.append(format_timestamp(event["timestamp"]))

                        strings.append("</text>")
                        timestamps.append(format_timestamp(event["timestamp"]))
                    else:
                        strings.append(str(event["text_value"]))
                        timestamps.append(format_timestamp(event["timestamp"]))
                
                if self.insert_event_tokens:
                    strings.append("</event>")
                    timestamps.append(format_timestamp(event["timestamp"]))

            strings.append("<end>")
            timestamps.append(format_timestamp(event["timestamp"]))

            result["strings"].append(strings)
            result["timestamps"].append(timestamps)

        return result

    @abstractmethod
    def train(self, events: pl.DataFrame, preprocessors: List[Preprocessor]) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def encode(self, events: pl.DataFrame, preprocessors: List[Preprocessor], allow_unknown: bool = False) -> List[int]:
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self) -> str:
        return f"Tokenizer: {self.tokenizer_name}\n" \
               f"Vocab size: {len(self.vocab)}\n"