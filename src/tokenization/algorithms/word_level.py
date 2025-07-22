from src.tokenization.algorithms.base import Tokenizer
from typing import List, Tuple
from collections import Counter, defaultdict
import polars as pl
import gc
import os
from tqdm import tqdm
from src.preprocessing.base import Preprocessor

class WordLevelTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 10000, insert_event_tokens: bool = True, insert_numeric_tokens: bool = True, insert_text_tokens: bool = True):
        super().__init__("word_level")
        self.vocab_size = vocab_size
        self.insert_event_tokens = insert_event_tokens
        self.insert_numeric_tokens = insert_numeric_tokens
        self.insert_text_tokens = insert_text_tokens
        self.vocab = pl.DataFrame(schema={"token": pl.Int64, "str": pl.String, "count": pl.Int64})

        # Add special tokens to the vocabulary
        special_token_rows = [
            {"token": idx, "str": token, "count": 0}
            for token, idx in self.special_tokens.items()
        ]
        self.vocab = pl.concat([
            self.vocab,
            pl.DataFrame(special_token_rows)
        ])

        self.vocab_map = {
            token: idx for idx, token in enumerate(self.special_tokens.keys())
        }

    def train(self, event_files: List[str], preprocessors: List[Preprocessor]) -> None:
        """
        Train the tokenizer using a list of parquet files containing MEDS events.
        The vocabulary will be limited to the most frequent tokens up to vocab_size,
        while preserving special tokens added during initialization.

        Args:
            event_files: List of paths to parquet files containing events
            preprocessors: List of preprocessors to apply to the events
        Returns:
            None
        """
        if not event_files:
            raise ValueError("event_files list cannot be empty")
        
        # Validate all files exist before processing
        for file_path in event_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Event file not found: {file_path}")
        
        print(f"Training tokenizer on {len(event_files)} files...")
        
        # Count frequencies of all codes across all files
        code_counts = Counter()
        
        for i, file_path in tqdm(enumerate(event_files), total=len(event_files)):
            
            try:
                # Read one file at a time for memory efficiency
                events = pl.read_parquet(file_path)

                # Apply preprocessors to the events
                if len(preprocessors) > 0:
                    for preprocessor in preprocessors:
                        events = preprocessor.encode_polars(events)
                
                # Process events for this file
                processed_events = self._process_events(events)
                subject_strs = self._events_to_str(processed_events)

                # Update counts for this file
                for subject_str in subject_strs:
                    code_counts.update(subject_str.split())
                
                # Explicitly delete large objects and force garbage collection
                del events, processed_events, subject_strs
                gc.collect()
                
            except Exception as e:
                raise RuntimeError(f"Error processing file {file_path}: {str(e)}")
        
        print(f"Found {len(code_counts)} unique tokens across all files")
        
        # Remove special tokens from counts as they're already in vocab
        for special_token in self.special_tokens.keys():
            code_counts.pop(special_token, None)
        
        # Sort codes by frequency and take top vocab_size (minus special tokens)
        remaining_slots = self.vocab_size - len(self.special_tokens)
        sorted_codes = sorted(code_counts.items(), key=lambda x: (-x[1], x[0]))[:remaining_slots]
        
        print(f"Selected top {len(sorted_codes)} tokens for vocabulary (excluding special tokens)")
        
        # Calculate count of excluded tokens for <unknown>
        included_tokens = set(code for code, _ in sorted_codes)
        excluded_count = sum(count for code, count in code_counts.items() if code not in included_tokens)
        
        # Update <unknown> token count
        self.vocab = self.vocab.with_columns(
            pl.when(pl.col("str") == self.unknown_token)
            .then(excluded_count)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        
        # Create vocabulary rows for learned tokens
        learned_vocab_rows = [
            {"token": idx + len(self.special_tokens), "str": code, "count": count}
            for idx, (code, count) in enumerate(sorted_codes)
        ]

        # Combine special tokens with learned tokens
        self.vocab = pl.concat([
            self.vocab,
            pl.DataFrame(learned_vocab_rows)
        ])
        
        # Update vocab map with learned tokens
        self.vocab_map.update({row["str"]: row["token"] for row in learned_vocab_rows})
        
        print(f"Training complete! Vocabulary size: {len(self.vocab)}")

    def encode(self, event_filepath: str, preprocessors: List[Preprocessor], allow_unknown: bool = False) -> List[dict]:
        """
        Encode a dataframe of events into their corresponding token IDs.
        
        Args:
            event_filepath: Path to parquet file containing events
            preprocessors: List of preprocessors to apply to the events
            allow_unknown: Whether to use <unknown> token for out-of-vocabulary words
            
        Returns:
            List of dictionaries with subject_id, tokens, and timestamps for each subject
        """
        if self.vocab.is_empty():
            raise ValueError("Tokenizer is not trained yet.")
        
        events = pl.read_parquet(event_filepath)
        
        # Apply preprocessors to the events
        if len(preprocessors) > 0:
            for preprocessor in preprocessors:
                events = preprocessor.encode_polars(events)
        
        processed_events = self._process_events(events)
        subject_strs = self._events_to_str(processed_events)

        timelines = []

        for subject_data, subject_str in zip(processed_events, subject_strs):
            tokens = []
            timestamps = []
            
            # Split the string into individual tokens
            str_tokens = subject_str.split()
            
            # Get base timestamp (first event with an actual timestamp)
            base_timestamp = None
            for event in subject_data["event_list"]:
                if event["timestamp"] is not None:
                    base_timestamp = event["timestamp"]
                    break
            
            # If no events have timestamps, use 0 as base
            if base_timestamp is None:
                base_timestamp = 0
            
            # Track current event index for timestamp mapping
            event_idx = 0
            
            for token_str in str_tokens:
                if token_str in self.vocab_map:
                    tokens.append(self.vocab_map[token_str])
                elif allow_unknown:
                    tokens.append(self.vocab_map[self.unknown_token])
                else:
                    raise ValueError(f"Token '{token_str}' not found in vocabulary.")
                
                # For timestamp, we need to map tokens back to events
                if self.insert_event_tokens:
                    # Use event tokens for timestamp mapping
                    if token_str == "<event>":
                        # Starting a new event, use current event timestamp
                        if event_idx < len(subject_data["event_list"]):
                            event_timestamp = subject_data["event_list"][event_idx]["timestamp"]
                            if event_timestamp is not None:
                                current_timestamp = (event_timestamp - base_timestamp).total_seconds()
                            else:
                                current_timestamp = 0  # Use 0 for events without timestamps
                        else:
                            current_timestamp = 0
                    elif token_str == "</event>":
                        # Ending current event, advance to next
                        event_idx += 1
                        # Keep the current timestamp for the closing token
                    else:
                        # Use current event timestamp for all tokens within the event
                        if event_idx < len(subject_data["event_list"]):
                            event_timestamp = subject_data["event_list"][event_idx]["timestamp"]
                            if event_timestamp is not None:
                                current_timestamp = (event_timestamp - base_timestamp).total_seconds()
                            else:
                                current_timestamp = 0  # Use 0 for events without timestamps
                        else:
                            current_timestamp = 0
                else:
                    # Without event tokens, we need to track events differently
                    # Check if this token is a code (starts with event codes we expect)
                    if (token_str not in ['<numeric>', '</numeric>', '<text>', '</text>'] and 
                        not token_str.replace('.', '').replace('-', '').isdigit() and
                        event_idx < len(subject_data["event_list"]) and
                        token_str == subject_data["event_list"][event_idx]["code"]):
                        # This is the start of a new event
                        event_timestamp = subject_data["event_list"][event_idx]["timestamp"]
                        if event_timestamp is not None:
                            current_timestamp = (event_timestamp - base_timestamp).total_seconds()
                        else:
                            current_timestamp = 0
                        event_idx += 1  # Move to next event after processing the code
                    # For all other tokens (numeric values, text, special tokens), keep current timestamp
                
                timestamps.append(current_timestamp)
                
            timelines.append(
                {
                    "subject_id": subject_data["subject_id"][0],
                    "tokens": tokens,
                    "timestamps": timestamps
                }
            )

        return timelines

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs into their corresponding event strings.

        Args:
            tokens (List[int]): list of token IDs

        Returns:
            str: string of event strings

        Raises:
            ValueError: if any token ID is not found in the vocabulary
        """
            
        # Check all tokens first to provide a complete error message
        invalid_tokens = []
        for token in tokens:
            if self.vocab.filter(pl.col("token") == token).is_empty():
                invalid_tokens.append(token)
        
        if invalid_tokens:
            raise ValueError(
                f"Found {len(invalid_tokens)} invalid token(s) not in vocabulary: {invalid_tokens}. "
                f"Valid token range is 0 to {len(self.vocab) - 1}"
            )
            
        # If all tokens are valid, proceed with decoding
        decoded_tokens = []
        for token in tokens:
            result = self.vocab.filter(pl.col("token") == token)["str"]
            decoded_tokens.extend(result.to_list())
        
        str_tokens = " ".join(decoded_tokens)

        return str_tokens


if __name__ == "__main__":
    import polars as pl
    from src.preprocessing.quantile_bin import QuantileBinPreprocessor
    import os
    
    dir = "/home/joshua/data/mimic_meds/mimic_iv_meds/MEDS_cohort/data/train"
    files = os.listdir(dir)
    files = [os.path.join(dir, file) for file in files]

    train_files = files[:3]
    test_file = files[3]
    
    quantile_bin_preprocessor = QuantileBinPreprocessor(matching_type="starts_with", matching_value="LAB", k=10)

    quantile_bin_preprocessor.fit(train_files)

    tokenizer = WordLevelTokenizer(vocab_size=2000, insert_event_tokens=True, insert_numeric_tokens=True, insert_text_tokens=True)
    tokenizer.train(train_files, [quantile_bin_preprocessor])

    test_tokens = tokenizer.encode(test_file, [quantile_bin_preprocessor], allow_unknown=True)

    print(test_tokens[0])

    print(tokenizer.decode(test_tokens[0]["tokens"]))
    
