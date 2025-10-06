import yaml
from src.tokenization import WordLevelTokenizer
import os
import polars as pl
import pickle
import json
from datetime import datetime
import shutil
from tqdm import tqdm
from typing import List
from src.preprocessing.base import BasePreprocessor
from src.postprocessing.base import Postprocessor
from src.preprocessing import QuantileBinPreprocessor, CodeEnrichmentPreprocessor, LoadStaticDataPreprocessor, EthosQuantileAgePreprocessor, DemographicAggregationPreprocessor
from src.preprocessing.code_truncation import CodeTruncationPreprocessor
from src.postprocessing import TimeIntervalPostprocessor, DemographicSortOrderPostprocessor
from src.preprocessing.utils import fit_preprocessors_jointly

DATASET_DIRS = ["train", "tuning", "held_out"]

def run_pipeline(config: dict, run_name: str, overwrite: bool = False):
    """
    Run a tokenization pipeline end to end.

    Args:
        config (dict): the config file
        run_name (str): the name of the run
        overwrite (bool): whether to overwrite the save directory if it exists
    """

    # Validate all the expected fields are present in the config
    validate_config(config)

    run_directory = os.path.join(config["save_path"], run_name)

    # Check if save directory already exists
    if os.path.exists(run_directory):
        if overwrite:
            print(f"Overwriting existing directory: {run_directory}")
            shutil.rmtree(run_directory)
        else:
            print(f"Error: Save directory {run_directory} already exists. Use the --overwrite flag to overwrite it.")
            return

    # Create save_path directory
    os.makedirs(run_directory)

    # ... (rest of the run_pipeline function remains the same)
    # Create subdirectories for each dataset
    for dataset in DATASET_DIRS:
        os.makedirs(os.path.join(config["save_path"], run_name, dataset), exist_ok=True)

    # Check data is valid and load it
    data_files = gather_data_files(config["data"]["path"])
    print(f"Found {len(data_files['train'])} train files, {len(data_files['tuning'])} tuning files, and {len(data_files['held_out'])} held out files")

    data_files['train'] = data_files['train']
    data_files['tuning'] = data_files['tuning']
    data_files['held_out'] = data_files['held_out']

    # Create preprocessors
    preprocessors = []
    if "preprocessing" in config:
        for preprocessing_config in config["preprocessing"]:
            if preprocessing_config["type"] == "code_truncation":
                preprocessor = CodeTruncationPreprocessor(
                    matching_type=preprocessing_config["matching_type"],
                    matching_value=preprocessing_config["matching_value"],
                )
            elif preprocessing_config["type"] == "quantile_bin":
                preprocessor = QuantileBinPreprocessor(
                    matching_type=preprocessing_config["matching_type"],
                    matching_value=preprocessing_config["matching_value"],
                    k=preprocessing_config["k"],
                    value_column=preprocessing_config["value_column"]
                )
            elif preprocessing_config["type"] == "code_enrichment":
                preprocessor = CodeEnrichmentPreprocessor(
                    matching_type=preprocessing_config["matching_type"],
                    matching_value=preprocessing_config["matching_value"],
                    lookup_file=preprocessing_config["lookup_file"],
                    template=preprocessing_config["template"],
                    code_column=preprocessing_config["code_column"],
                    dtypes=preprocessing_config.get("dtypes", None),
                    additional_filters=preprocessing_config.get("additional_filters", None)
                )
            elif preprocessing_config["type"] == "load_static_data":
                preprocessor = LoadStaticDataPreprocessor(
                    matching_type="",  # Not used for static data
                    matching_value="", # Not used for static data
                    csv_filepath=preprocessing_config["csv_filepath"],
                    subject_id_column=preprocessing_config["subject_id_column"],
                    columns=preprocessing_config["columns"]
                )
            elif preprocessing_config["type"] == "ethos_quantile_age":
                preprocessor = EthosQuantileAgePreprocessor(
                    matching_type="",  # Not used for age processing
                    matching_value="", # Not used for age processing
                    time_unit=preprocessing_config.get("time_unit", "years"),
                    num_quantiles=preprocessing_config.get("num_quantiles", 10),
                    prefix=preprocessing_config.get("prefix", "AGE_"),
                    insert_t1_code=preprocessing_config.get("insert_t1_code", True),
                    insert_t2_code=preprocessing_config.get("insert_t2_code", True),
                    keep_meds_birth=preprocessing_config.get("keep_meds_birth", False)
                )
            elif preprocessing_config["type"] == "demographic_aggregation":
                preprocessor = DemographicAggregationPreprocessor(
                    matching_type="",  # Not used for demographic aggregation
                    matching_value="", # Not used for demographic aggregation
                    measurements=preprocessing_config["measurements"]
                )
            else:
                raise ValueError(f"Preprocessor {preprocessing_config['type']} not supported")
            
            preprocessors.append(preprocessor)

    # Fit all preprocessors jointly
    if preprocessors:
        fit_preprocessors_jointly(preprocessors, data_files["train"])

    # Load postprocessors
    postprocessors = []
    if "postprocessing" in config:
        for postprocessing_config in config["postprocessing"]:
            if postprocessing_config["type"] == "time_interval":
                postprocessor = TimeIntervalPostprocessor(postprocessing_config["interval_tokens"])
            elif postprocessing_config["type"] == "demographic_sort_order":
                postprocessor = DemographicSortOrderPostprocessor(postprocessing_config["token_patterns"])
            else:
                raise ValueError(f"Postprocessor {postprocessing_config['type']} not supported")
            
            postprocessors.append(postprocessor)

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
    tokenizer.train(data_files["train"], preprocessors, postprocessors)

    # encode train data
    encode_files(tokenizer, data_files["train"], os.path.join(run_directory, "train"), preprocessors, postprocessors)

    # encode tuning data
    encode_files(tokenizer, data_files["tuning"], os.path.join(run_directory, "tuning"), preprocessors, postprocessors)

    # encode held out data
    encode_files(tokenizer, data_files["held_out"], os.path.join(run_directory, "held_out"), preprocessors, postprocessors)

    # store a copy of the config file
    with open(os.path.join(run_directory, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # store metadata
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "config": config,
    }
    with open(os.path.join(run_directory, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # store the vocab file as a csv
    tokenizer.vocab.write_csv(os.path.join(run_directory, "vocab.csv"))

def validate_config(config: dict):
    """
    Validate the config has all the expected fields.
    Raises a ValueError if any expected fields are missing.

    Args:
        config (dict): the config file
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

    Args:
        data_path (str): the path to the data directory

    Returns:
        dict: a dictionary of dataset directories and their files
    """
    data_files = {x: [] for x in DATASET_DIRS}
    
    for dir in DATASET_DIRS:
        dataset_dir_files = os.listdir(os.path.join(data_path, dir))
        for file in dataset_dir_files:
            if file.endswith(".parquet"):
                data_files[dir].append(os.path.join(data_path, dir, file))

    return data_files

def encode_files(tokenizer, event_files: List[str], save_path: str, preprocessors: List[BasePreprocessor] = None, postprocessors: List[Postprocessor] = None):
    """
    Encode a list of event files and save them as pickle files.

    Args:
        tokenizer (Tokenizer): the tokenizer to use
        event_files (List[str]): the list of event files to encode
        save_path (str): the path to save the encoded files
        preprocessors (List[Preprocessor]): the list of preprocessors to use
    """
    for file in tqdm(event_files):
        encoded_data = tokenizer.encode(file, preprocessors, postprocessors)
        with open(os.path.join(save_path, os.path.basename(file).replace(".parquet", ".pkl")), "wb") as f:
            pickle.dump(encoded_data, f)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the save directory if it exists.")

    args = parser.parse_args()

    with open(args.config_filepath, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config, args.run_name, args.overwrite)