import pickle
from pathlib import Path
from typing import List

import polars as pl
from transformers import PreTrainedTokenizerFast


# ---- CONFIG (edit these paths) ----
# One of your encoded .pkl timelines
PKL_PATH = Path("/home/ashvingupta/Documents/PhD/Projects/ehr-tokenisation/artifacts/tokenizers/experiments/vocab_5000/train_0.pkl")  # change if needed

# Directory of the trained tokenizer (matches the vocab_4000 you’re inspecting)
TOKENIZER_DIR = Path("/home/ashvingupta/Documents/PhD/Projects/ehr-tokenisation/artifacts/tokenizers/experiments/vocab_5000")

# A single translated parquet file produced by the MIMIC translation step
TRANSLATED_PARQUET = Path(
    "/home/ashvingupta/Documents/MIMIC-IV/mimic_meds_ed/groups/a-faisal/edisonliu/MIMIC-IV-preprocessed/EHR_3.1_ED_2.2/translated/test/0.parquet"
)

TEXT_COLUMN = "text_sequence_with_time"

N_EXAMPLES = 3
N_TOKENS_TO_SHOW = 40
N_PARQUET_CHARS_TO_SHOW = 100  # Number of chars to show from the parquet file


def load_timelines(pkl_path: Path) -> List[dict]:
    pkl_path = pkl_path.expanduser().resolve()
    with pkl_path.open("rb") as f:
        timelines = pickle.load(f)
    return timelines


def main() -> None:
    tokenizer_dir = TOKENIZER_DIR.expanduser().resolve()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"Loaded tokenizer from {tokenizer_dir}")

    timelines = load_timelines(PKL_PATH)
    print(f"Loaded {len(timelines)} timelines from {PKL_PATH}")

    translated_path = TRANSLATED_PARQUET.expanduser().resolve()
    print(f"Loading translated parquet from {translated_path}")

    # Only read the first N_PARQUET_CHARS_TO_SHOW characters from the parquet file
    # by reading a small number of rows, just for a preview (for efficiency)
    try:
        # Read only the first row (or as needed) from the parquet file
        preview_df = pl.read_parquet(
            translated_path, columns=[TEXT_COLUMN], n_rows=1
        )
        if preview_df.height > 0:
            preview_text = preview_df[TEXT_COLUMN][0]
            if preview_text is not None:
                s = str(preview_text)
                print("First few characters from translated parquet (first 100 chars):", repr(s[:N_PARQUET_CHARS_TO_SHOW]))
            else:
                print("First few characters from translated parquet: <None>")
        else:
            print("First few characters from translated parquet: <no rows in file>")
    except Exception as e:
        print(f"Could not read preview from parquet file: {e}")

    # The rest of the logic (timelines, tokens, decoding, etc.) remains as before
    for i, tl in enumerate(timelines[:N_EXAMPLES]):
        print("\n" + "=" * 80)
        print(f"Example {i}  subject_id={tl['subject_id']}")
        token_ids = tl["tokens"]
        print(f"Num tokens: {len(token_ids)}")

        # Print raw token IDs (added for debug)
        print("Token IDs:", token_ids[:N_TOKENS_TO_SHOW])

        # Show raw token strings
        sample_ids = token_ids[:N_TOKENS_TO_SHOW]
        tokens = tokenizer.convert_ids_to_tokens(sample_ids)
        print("Raw tokens:", tokens)

        # Show decoded text
        decoded = tokenizer.decode(sample_ids, clean_up_tokenization_spaces=False)
        print("Decoded text:", repr(decoded))
        print("Decoded text (first 100 chars):", repr(decoded[:100]))

        # We are no longer showing original translated text matched by subject_id,
        # since we just want to preview characters from the parquet

if __name__ == "__main__":
    main()

