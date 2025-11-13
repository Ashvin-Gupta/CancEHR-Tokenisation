from src.tokenization.algorithms.base import Tokenizer
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import polars as pl
import gc
from tqdm import tqdm
from src.preprocessing.base import BasePreprocessor
from src.postprocessing.base import Postprocessor

class BPETokenizer(Tokenizer):
    """
    A tokenizer that uses Byte Pair Encoding (BPE) to tokenize events.
    BPE learns subword units by iteratively merging the most frequent 
    consecutive token pairs.
    
    Args:
        vocab_size (int): the size of the vocabulary.
        insert_event_tokens (bool): whether to insert event tokens.
        insert_numeric_tokens (bool): whether to insert numeric tokens.
        insert_text_tokens (bool): whether to insert text tokens.
        end_of_word_suffix (str): suffix to mark end of words (default: "</w>")
    """
    def __init__(self, vocab_size: int = 1000, 
                 insert_event_tokens: bool = True, 
                 insert_numeric_tokens: bool = True, 
                 insert_text_tokens: bool = True,
                 end_of_word_suffix: str = "</w>") -> None:
        super().__init__("bpe", vocab_size, insert_event_tokens, 
                        insert_numeric_tokens, insert_text_tokens)
        
        self.end_of_word_suffix = end_of_word_suffix
        self.merges = []  # List of (pair, new_token) tuples in order of learning
        self.merge_ranks = {}  # Maps pair tuples to their merge priority
        
        # Initialize vocab_map with special tokens
        self.vocab_map = {
            token: idx for token, idx in self.special_tokens.items()
        }
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Split a word into characters with end-of-word marker."""
        if not word:
            return []
        chars = list(word)
        chars[-1] = chars[-1] + self.end_of_word_suffix
        return chars
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get all consecutive pairs in a tokenized word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs
    
    def _merge_pair(self, word: List[str], pair: Tuple[str, str], 
                    replacement: str) -> List[str]:
        """Merge all occurrences of a pair in a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def train(self, event_files: List[str], 
              preprocessors: List[BasePreprocessor], 
              postprocessors: List[Postprocessor]) -> None:
        """
        Train the BPE tokenizer using a list of parquet files.
        
        Algorithm:
        1. Initialize vocabulary with all characters
        2. Count frequencies of all token pairs
        3. Merge the most frequent pair
        4. Repeat until vocab_size is reached
        """
        if not event_files:
            raise ValueError("event_files list cannot be empty")
        
        print(f"Training BPE tokenizer on {len(event_files)} files...")
        
        # Step 1: Collect all words and their frequencies
        word_freqs = Counter()
        
        for file_path in tqdm(event_files, desc="Collecting words"):
            events = pl.read_parquet(file_path)
            
            if len(preprocessors) > 0:
                for preprocessor in preprocessors:
                    events = preprocessor.encode_polars(events)
            
            processed_events = self._process_events(events)
            print(processed_events)
            
            if len(postprocessors) > 0:
                for postprocessor in postprocessors:
                    processed_events = postprocessor.encode(processed_events)
            
            subject_strs = self._events_to_lists(processed_events)["strings"]
            
            for string_list in subject_strs:
                for word in string_list:
                    if word not in self.special_tokens:
                        word_freqs[word] += 1
            
            del events, processed_events, subject_strs
            gc.collect()
        
        print(f"Found {len(word_freqs)} unique words")
        
        # Step 2: Initialize vocabulary with characters
        # Tokenize all words into characters
        split_words = {word: self._tokenize_word(word) 
                      for word in word_freqs.keys()}
        
        # Get all unique characters
        char_vocab = set()
        for tokens in split_words.values():
            char_vocab.update(tokens)
        
        # Initialize vocabulary: special tokens + characters
        next_token_id = len(self.special_tokens)
        for char in sorted(char_vocab):
            self.vocab_map[char] = next_token_id
            next_token_id += 1
        
        print(f"Initial character vocabulary size: {len(char_vocab)}")
        
        # Step 3: Learn merges
        num_merges = self.vocab_size - len(self.vocab_map)
        
        for merge_idx in tqdm(range(num_merges), desc="Learning merges"):
            # Count all pairs across all words (weighted by word frequency)
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                word_tokens = split_words[word]
                pairs = self._get_pairs(word_tokens)
                for pair in pairs:
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                print(f"No more pairs to merge. Stopping at {len(self.vocab_map)} tokens.")
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Create merged token
            merged_token = ''.join(best_pair)
            
            # Add to vocabulary
            self.vocab_map[merged_token] = next_token_id
            next_token_id += 1
            
            # Record the merge
            self.merges.append((best_pair, merged_token))
            self.merge_ranks[best_pair] = merge_idx
            
            # Update all words with this merge
            for word in split_words:
                split_words[word] = self._merge_pair(
                    split_words[word], best_pair, merged_token
                )
        
        print(f"Training complete! Learned {len(self.merges)} merges")
        print(f"Final vocabulary size: {len(self.vocab_map)}")
        
        # Build the vocab DataFrame for compatibility
        vocab_rows = [
            {"token": token_id, "str": token_str, "count": 0}
            for token_str, token_id in self.vocab_map.items()
        ]
        self.vocab = pl.DataFrame(vocab_rows)
    
    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word using learned BPE merges."""
        if word in self.special_tokens:
            return [word]
        
        # Start with character-level tokenization
        word_tokens = self._tokenize_word(word)
        
        if len(word_tokens) == 1:
            return word_tokens
        
        # Apply merges in order
        while len(word_tokens) > 1:
            # Find the highest priority pair that exists in the word
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break
            
            # Find pair with lowest merge rank (earliest merge)
            bigram = min(pairs, 
                        key=lambda pair: self.merge_ranks.get(pair, float('inf')))
            
            # If no valid merge found, stop
            if bigram not in self.merge_ranks:
                break
            
            # Apply the merge
            merged_token = ''.join(bigram)
            word_tokens = self._merge_pair(word_tokens, bigram, merged_token)
        
        return word_tokens
    
    def encode(self, event_filepath: str, 
               preprocessors: List[BasePreprocessor], 
               postprocessors: List[Postprocessor]) -> List[dict]:
        """Encode events using BPE tokenization."""
        if not self.vocab_map or not self.merges:
            raise ValueError("Tokenizer is not trained yet.")
        
        events = pl.read_parquet(event_filepath)
        
        if len(preprocessors) > 0:
            for preprocessor in preprocessors:
                events = preprocessor.encode_polars(events)
        
        processed_events = self._process_events(events)
        
        if len(postprocessors) > 0:
            for postprocessor in postprocessors:
                processed_events = postprocessor.encode(processed_events)
        
        subject_ids = [x["subject_id"][0] for x in processed_events]
        subject_event_lists = self._events_to_lists(processed_events)
        
        timelines = []
        for subject_id, subject_strings, subject_timestamps in zip(
            subject_ids, subject_event_lists["strings"], 
            subject_event_lists["timestamps"]):
            
            # Tokenize each word with BPE
            tokens = []
            expanded_timestamps = []
            for word, timestamp in zip(subject_strings, subject_timestamps):
                bpe_tokens = self._encode_word(word)
                for bpe_token in bpe_tokens:
                    token_id = self.vocab_map.get(
                        bpe_token, 
                        self.vocab_map[self.unknown_token]
                    )
                    tokens.append(token_id)
                    expanded_timestamps.append(timestamp)
            
            timelines.append({
                "subject_id": subject_id,
                "tokens": tokens,
                "timestamps": expanded_timestamps
            })
        
        return timelines
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        # Create reverse mapping
        id_to_token = {v: k for k, v in self.vocab_map.items()}
        
        decoded_tokens = []
        for token_id in tokens:
            if token_id not in id_to_token:
                raise ValueError(f"Token ID {token_id} not in vocabulary")
            decoded_tokens.append(id_to_token[token_id])
        
        # Join tokens and clean up word boundaries
        result = ''.join(decoded_tokens)
        result = result.replace(self.end_of_word_suffix, ' ')
        
        return result.strip()