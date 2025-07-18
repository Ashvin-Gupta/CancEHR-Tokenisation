from src.tokenization.algorithms.base import Tokenizer
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import polars as pl

class BPE(Tokenizer):
    def __init__(self, vocab_size: int = 10_000):
        super().__init__("bpe")
        self.vocab_size = vocab_size
        self.vocab = pl.DataFrame(schema={"token": pl.Int64, "str": pl.String, "count": pl.Int64})
        self.merges = {}  # Store the learned BPE merges

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

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs in the vocabulary."""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        """Merge all occurrences of the pair in the vocabulary."""
        merged = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged.append(new_word)
        return merged

    def train(self, events: pl.DataFrame) -> None:
        """
        Train the BPE tokenizer using a polars dataframe of MEDS events.
        The vocabulary will be limited to the most frequent tokens up to vocab_size,
        while preserving special tokens added during initialization.
        """
        processed_events = self._process_events(events)
        
        # Initialize vocabulary with individual characters
        char_counts = Counter()
        for event in processed_events:
            for code in event["event_list"]:
                char_counts.update(code)
        
        # Remove special tokens from counts
        for special_token in self.special_tokens.keys():
            char_counts.pop(special_token, None)
        
        # Create initial vocabulary with individual characters
        vocab_rows = [
            {"token": idx + len(self.special_tokens), "str": char, "count": count}
            for idx, (char, count) in enumerate(char_counts.most_common())
        ]
        
        # Convert all codes to character sequences
        words = []
        for event in processed_events:
            for code in event["event_list"]:
                if code not in self.special_tokens:
                    words.append(list(code))
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.special_tokens) - len(vocab_rows)
        for _ in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Merge the pair
            words = self._merge_pair(best_pair, words)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            vocab_rows.append({
                "token": len(vocab_rows) + len(self.special_tokens),
                "str": merged_token,
                "count": pairs[best_pair]
            })
        
        # Update vocabulary
        self.vocab = pl.concat([
            self.vocab,
            pl.DataFrame(vocab_rows)
        ])
        
        # Update vocab map
        self.vocab_map.update({row["str"]: row["token"] for row in vocab_rows})

    def _encode_word(self, word: str) -> List[int]:
        """
        Encode a single word using BPE by finding longest matching substrings.
        
        Args:
            word: The word to encode
            
        Returns:
            List of token IDs
        """
        if word in self.vocab_map:
            return [self.vocab_map[word]]
        
        # Get all vocab strings sorted by length (longest first)
        vocab_strings = sorted([row["str"] for row in self.vocab.to_dicts() 
                              if row["str"] not in self.special_tokens], 
                             key=len, reverse=True)
        
        tokens = []
        i = 0
        
        while i < len(word):
            # Try to find the longest matching substring starting at position i
            matched = False
            for vocab_token in vocab_strings:
                if word[i:].startswith(vocab_token):
                    tokens.append(self.vocab_map[vocab_token])
                    i += len(vocab_token)
                    matched = True
                    break
            
            if not matched:
                # If no substring matches, use the single character or unknown token
                char = word[i]
                if char in self.vocab_map:
                    tokens.append(self.vocab_map[char])
                else:
                    tokens.append(self.vocab_map[self.unknown_token])
                i += 1
                
        return tokens

    def encode(self, events: pl.DataFrame, allow_unknown: bool = False) -> List[dict]:
        """
        Encode a dataframe of events into their corresponding token IDs using BPE.
        
        Args:
            events: DataFrame of events to encode
            allow_unknown: Whether to use unknown token for unseen tokens
            
        Returns:
            List of dictionaries containing subject_id, tokens, and timestamps
        """
        if self.vocab.is_empty():
            raise ValueError("Tokenizer is not trained yet.")
            
        processed_events = self._process_events(events)
        timelines = []

        for subject_data in processed_events:
            tokens = []
            timestamps = []
            for idx, event_code in enumerate(subject_data["event_list"]):
                if event_code in self.special_tokens:
                    tokens.append(self.vocab_map[event_code])
                else:
                    # Encode the word using BPE
                    word_tokens = self._encode_word(event_code)
                    tokens.extend(word_tokens)
                timestamps.append(subject_data["event_timestamps"][idx] - subject_data["event_timestamps"][0])
                
            timelines.append({
                "subject_id": subject_data["subject_id"],
                "tokens": tokens,
                "timestamps": timestamps
            })

        return timelines

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs into their corresponding event strings.

        Args:
            tokens: list of token IDs

        Returns:
            string of event strings

        Raises:
            ValueError: if any token ID is not found in the vocabulary
        """
        if not isinstance(tokens, list):
            raise TypeError(f"Expected list of tokens, got {type(tokens)}")
            
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
        return " ".join(decoded_tokens)


if __name__ == "__main__":
    import polars as pl
    tokenizer = BPE(vocab_size=100)
    corpus = pl.read_parquet("/home/joshua/data/mimic/meds/mimiciv_demo/2.2/data/events.parquet")
    tokenizer.train(corpus)
    print(tokenizer.vocab)
    tokenizer.vocab.write_csv("vocab.csv")


    x = corpus.filter(pl.col("subject_id") == 10000032)
    x_text = " ".join([code for code in x['code']])
    tokens = tokenizer.encode(x, allow_unknown=True)[0]['tokens']
    print(tokens)
    print(tokenizer.decode(tokens))
    print(x_text)
    # tokens = tokenizer.encode(corpus[0], allow_unknown=True)

    # text = ""
    # for t in tokens:
    #     text += tokenizer.decode(t['tokens']) + " "
    # print(text)
    # tokens = tokenizer.encode(corpus, allow_unknown=True)
    # for t in tokens:
    #     print(t)
    #     print(tokenizer.decode(t['tokens']))
