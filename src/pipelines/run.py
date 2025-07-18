import yaml
from src.tokenization import WordLevelTokenizer, BPE
import os
import polars as pl
import pickle
import json
from datetime import datetime
import shutil
from tqdm import tqdm

DATASET_DIRS = ["train", "tuning", "held_out"]

def run_pipeline(config: dict):

    # Validate all the expected fields are present in the config
    validate_config(config)

    # Check is save directory already exists, and ask if the user wants to overwrite it
    if os.path.exists(os.path.join(config["save_path"], config["name"])):
        overwrite = input(f"Save directory {os.path.join(config['save_path'], config['name'])} already exists. Overwrite? (y/n): ")
        if overwrite != "y":
            print("Exiting...")
            return
        else:
            print("Overwriting...")
            shutil.rmtree(os.path.join(config["save_path"], config["name"]))

    # create save_path directory if it doesn't exist
    if not os.path.exists(os.path.join(config["save_path"], config["name"])):
        os.makedirs(os.path.join(config["save_path"], config["name"]))

    # Create subdirectories for each dataset
    for dataset in DATASET_DIRS:
        os.makedirs(os.path.join(config["save_path"], config["name"], dataset), exist_ok=True)

    # Check data is valid and load it
    data_files = gather_data_files(config["data"]["path"])
    print(f"Found {len(data_files['train'])} train files, {len(data_files['tuning'])} tuning files, and {len(data_files['held_out'])} held out files")

    # Load tokenizer
    if config["tokenization"]["tokenizer"] == "word_level":
        tokenizer = WordLevelTokenizer(
            vocab_size=config["tokenization"]["vocab_size"],
            insert_event_tokens=config["tokenization"]["insert_event_tokens"],
            insert_numeric_tokens=config["tokenization"]["insert_numeric_tokens"],
            insert_text_tokens=config["tokenization"]["insert_text_tokens"]
        )
    else:
        raise ValueError(f"Tokenizer {config['tokenization']['tokenizer']} not supported")
    
    # Fit tokenizer to train data
    tokenizer.train(data_files["train"])

    # encode train data
    encode_files(tokenizer, data_files["train"], os.path.join(config["save_path"], config["name"], "train"))

    # encode tuning data
    encode_files(tokenizer, data_files["tuning"], os.path.join(config["save_path"], config["name"], "tuning"))

    # encode held out data
    encode_files(tokenizer, data_files["held_out"], os.path.join(config["save_path"], config["name"], "held_out"))

    # store a copy of the config file
    with open(os.path.join(config["save_path"], config["name"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # store metadata
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(config["save_path"], config["name"], "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # store the vocab file as a csv
    tokenizer.vocab.write_csv(os.path.join(config["save_path"], config["name"], "vocab.csv"))

def validate_config(config: dict):
    """
    Validate the config file.
    """
    
    # Check tokenizer is valid
    if config["tokenization"]["tokenizer"] not in ["word_level", "bpe"]:
        raise ValueError(f"Tokenizer {config['tokenization']['tokenizer']} not supported")
    
    # Check data path is valid and files exist
    if config["data"]["path"] is None:
        raise ValueError("Data path is not set")
    
    dataset_dirs= os.listdir(config["data"]["path"])
    
    if not all(dir in dataset_dirs for dir in DATASET_DIRS):
        raise ValueError(f"Data path does not contain all expected directories: {DATASET_DIRS}")
    
    # Check data files exist
    for dir in DATASET_DIRS:
        dataset_dir_files = os.listdir(os.path.join(config["data"]["path"], dir))
        if not all(file.endswith(".parquet") for file in dataset_dir_files):
            raise ValueError(f"Data path does not contain all expected files: {dataset_dir_files}")

def gather_data_files(data_path: str):
    """
    Gather all data files from a data directory.
    Directory should be structured as:
    data_path/
    ├── train/
    │   ├── 0.parquet
    │   ├── 1.parquet
    │   └── ...
    ├── tuning/
    │   ├── 0.parquet
    │   ├── 1.parquet
    │   └── ...
    └── held_out/
        ├── 0.parquet
        ├── 1.parquet
        └── ...
    """
    data_files = {x: [] for x in DATASET_DIRS}
    
    for dir in DATASET_DIRS:
        dataset_dir_files = os.listdir(os.path.join(data_path, dir))
        for file in dataset_dir_files:
            if file.endswith(".parquet"):
                data_files[dir].append(os.path.join(data_path, dir, file))

    return data_files

def encode_files(tokenizer, event_files, save_path: str):

    for file in tqdm(event_files):
        encoded_data = tokenizer.encode(file, allow_unknown=True)
        with open(os.path.join(save_path, os.path.basename(file).replace(".parquet", ".pkl")), "wb") as f:
            pickle.dump(encoded_data, f)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str)

    args = parser.parse_args()

    with open(args.config_filepath, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config)