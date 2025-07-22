from typing import List
import polars as pl
from src.preprocessing.base import Preprocessor
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.unknown_token = "<unknown>"
        self.vocab = pl.DataFrame(schema={"token": pl.Int64, "str": pl.String, "count": pl.Int64})
        self.special_tokens = {
            self.unknown_token: 0
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
    
    def _events_to_str(self, events: List[dict]) -> str:
        """
        Convert a list of events to a string.

        Args:
            events (List[dict]): the events to convert to a string

        Returns:
            str: the string representation of the events
        """
        strs = []
        for subject in events:
            subject_str = ""
            for event in subject["event_list"]:
                if self.insert_event_tokens:
                    subject_str += "<event> "
                
                subject_str += event["code"]
                
                if event["numeric_value"] is not None:
                    if self.insert_numeric_tokens:
                        subject_str += f" <numeric> {event['numeric_value']} </numeric>"
                    else:
                        subject_str += f" {event['numeric_value']}"
                
                if event["text_value"] is not None:
                    if self.insert_text_tokens:
                        subject_str += f" <text> {event['text_value']} </text>"
                    else:
                        subject_str += f" {event['text_value']}"
                
                if self.insert_event_tokens:
                    subject_str += " </event> "
                else:
                    subject_str += " "

            strs.append(subject_str)

        return strs

    @abstractmethod
    def train(self, events: pl.DataFrame, preprocessors: List[Preprocessor]) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def encode(self, events: pl.DataFrame, preprocessors: List[Preprocessor], allow_unknown: bool = False) -> List[int]:
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self) -> str:
        return f"Tokenizer: {self.tokenizer_name}\n" \
               f"Vocab size: {len(self.vocab)}\n"