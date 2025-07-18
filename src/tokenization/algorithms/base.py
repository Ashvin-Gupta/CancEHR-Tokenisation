from typing import List
import polars as pl

class Tokenizer:
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

    def train(self, events: pl.DataFrame) -> None:
        pass

    def encode(self, events: pl.DataFrame, allow_unknown: bool = False) -> List[int]:
        pass

    def __str__(self) -> str:
        return f"Tokenizer: {self.tokenizer_name}\n" \
               f"Vocab size: {len(self.vocab)}\n"