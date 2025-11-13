from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import polars as pl
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src.postprocessing.base import Postprocessor
from src.preprocessing.base import BasePreprocessor
from src.tokenization.algorithms.base import Tokenizer


class HFBPETokenizer(Tokenizer):
    """
    Hugging Face BPE tokenizer that plugs into the existing tokenization pipeline.
    """
    def __init__(
        self,
        vocab_size: int = 10_000,
        insert_event_tokens: bool = True,
        insert_numeric_tokens: bool = True,
        insert_text_tokens: bool = True,
        tokenizer_dir: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            tokenizer_name="hf_bpe",
            vocab_size=vocab_size,
            insert_event_tokens=insert_event_tokens,
            insert_numeric_tokens=insert_numeric_tokens,
            insert_text_tokens=insert_text_tokens,
        )

        default_specials = [
            self.unknown_token,
            self.start_token,
            self.end_token,
        ]
        if self.insert_event_tokens:
            default_specials.extend([self.event_start_token, self.event_end_token])
        if self.insert_numeric_tokens:
            default_specials.extend([self.numeric_start_token, self.numeric_end_token])
        if self.insert_text_tokens:
            default_specials.extend([self.text_start_token, self.text_end_token])

        self.special_tokens_list = list(dict.fromkeys((special_tokens or []) + default_specials))
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self._hf_tokenizer: Optional[PreTrainedTokenizerFast] = None

    # --------------------------------------------------------------------- #
    # Training                                                              #
    # --------------------------------------------------------------------- #
    def train(
        self,
        event_files: List[str],
        preprocessors: List[BasePreprocessor],
        postprocessors: List[Postprocessor],
    ) -> None:
        """
        Train a BPE tokenizer using Hugging Face's `tokenizers` library.
        """
        if self.tokenizer_dir and (self.tokenizer_dir / "tokenizer.json").exists():
            self._load_existing_tokenizer(self.tokenizer_dir)
            return

        hf_core_tokenizer = HFTokenizer(models.BPE(unk_token=self.unknown_token))
        hf_core_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=True,
        )

        hf_core_tokenizer.train_from_iterator(
            iterator=self._training_iterator(event_files, preprocessors, postprocessors),
            trainer=trainer,
        )

        self._hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=hf_core_tokenizer,
            unk_token=self.unknown_token,
            pad_token=None,
        )

        if self.tokenizer_dir:
            self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            self._hf_tokenizer.save_pretrained(self.tokenizer_dir)

        self._update_vocab_dataframe()

    def _training_iterator(
        self,
        event_files: List[str],
        preprocessors: List[BasePreprocessor],
        postprocessors: List[Postprocessor],
    ) -> Iterator[str]:
        for strings, _timestamps in self._yield_subject_sequences(event_files, preprocessors, postprocessors):
            yield " ".join(strings)

    # --------------------------------------------------------------------- #
    # Encoding                                                              #
    # --------------------------------------------------------------------- #
    def encode(
        self,
        event_filepath: str,
        preprocessors: List[BasePreprocessor],
        postprocessors: List[Postprocessor],
    ) -> List[dict]:
        """
        Encode one parquet file into token-id timelines (with timestamps).
        """
        if self._hf_tokenizer is None:
            if not self.tokenizer_dir:
                raise ValueError("Tokenizer has not been trained yet.")
            self._load_existing_tokenizer(self.tokenizer_dir)

        timelines = []
        for subject_id, strings, timestamps in self._yield_subject_sequences(
            [event_filepath], preprocessors, postprocessors, include_subject_ids=True
        ):
            encoding = self._hf_tokenizer(
                strings,
                is_split_into_words=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )

            encoded_ids = encoding.ids
            encoded_word_ids = encoding.word_ids()

            expanded_timestamps = []
            for word_id in encoded_word_ids:
                if word_id is None:
                    expanded_timestamps.append(0.0)
                else:
                    expanded_timestamps.append(timestamps[word_id])

            timelines.append(
                {
                    "subject_id": subject_id,
                    "tokens": encoded_ids,
                    "timestamps": expanded_timestamps,
                }
            )

        return timelines

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #
    def _yield_subject_sequences(
        self,
        event_files: Iterable[str],
        preprocessors: List[BasePreprocessor],
        postprocessors: List[Postprocessor],
        include_subject_ids: bool = False,
    ) -> Iterator:
        for file_path in event_files:
            events = pl.read_parquet(file_path)

            for preprocessor in preprocessors or []:
                events = preprocessor.encode_polars(events)

            processed_events = self._process_events(events)

            for postprocessor in postprocessors or []:
                processed_events = postprocessor.encode(processed_events)

            subject_lists = self._events_to_lists(processed_events)

            for idx, (strings, timestamps) in enumerate(
                zip(subject_lists["strings"], subject_lists["timestamps"])
            ):
                if include_subject_ids:
                    subject_id = processed_events[idx]["subject_id"][0]
                    yield subject_id, strings, timestamps
                else:
                    yield strings, timestamps

    def _load_existing_tokenizer(self, directory: Path) -> None:
        self._hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
        self._update_vocab_dataframe()

    def _update_vocab_dataframe(self) -> None:
        vocab_items = sorted(self._hf_tokenizer.get_vocab().items(), key=lambda kv: kv[1])
        self.vocab = pl.DataFrame(
            {
                "token": [idx for _token, idx in vocab_items],
                "str": [token for token, _idx in vocab_items],
                "count": [0] * len(vocab_items),
            }
        )