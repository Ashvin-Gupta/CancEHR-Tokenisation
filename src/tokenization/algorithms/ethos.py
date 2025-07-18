import polars as pl
from src.tokenization.algorithms.base import Tokenizer
from typing import List

def main(input_dir: str):
    
    df = pl.scan_parquet(input_dir + "/0.parquet").collect()
    print(df.shape)
    print(df.head())

    # print the df where subject_id = 10007275
    print(df.filter(pl.col("subject_id") == 10007275).head())

    # print df where numeric_value is not null
    print(df.filter(pl.col("numeric_value").is_not_null()).head())

class EthosTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 10_000):
        super().__init__("ethos")
        self.vocab_size = vocab_size
        self.vocab = pl.DataFrame(schema={"token": pl.Int64, "str": pl.String, "count": pl.Int64})

    def train(self, events: pl.DataFrame) -> None:
        
        processed_events = self._process_events(events)

        x = processed_events[0]

        print(x)

        # print(x["event_list"])
        # print(x["event_timestamps"])
        # print(x["event_text"])
    
    def encode(self, events: pl.DataFrame, allow_unknown: bool = False) -> List[dict]:
        pass
    
    def decode(self, tokens: List[int]) -> str:
        pass

if __name__ == "__main__":
    import polars as pl
    INPUT_DIR = "/home/joshua/data/mimic_meds/mimic_iv_meds/MEDS_cohort/data/train"

    df = pl.scan_parquet(INPUT_DIR + "/0.parquet").collect()

    tokenizer = EthosTokenizer()
    tokenizer.train(df)
    # print(tokenizer.vocab)
    
