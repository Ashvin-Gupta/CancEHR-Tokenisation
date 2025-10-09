import polars as pl
import pickle
import os
from tqdm import tqdm

# --- CONFIGURE YOUR FILE PATHS HERE ---

# 1. Path to the directory containing your RAW .parquet files (the input to the pipeline)
#    This should be the same as the 'data.path' in your YAML config.
RAW_DATA_DIR = "/data/scratch/qc25022/liver/final_cleaned_debug/tuning/"

# 2. Path to the directory containing your final TOKENIZED .pkl files
#    This is the output directory from your tokenization run.
TOKENIZED_DATA_DIR = "/data/scratch/qc25022/liver/tokenised_data/cprd_test/tuning/"

# --- VALIDATION SCRIPT ---

def get_ids_from_parquet(directory: str) -> set:
    """Reads all subject_ids from all .parquet files in a directory."""
    subject_ids = set()
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"Warning: No .parquet files found in '{directory}'")
        return subject_ids
        
    for file_path in tqdm(parquet_files, desc="Reading raw Parquet files"):
        df = pl.read_parquet(file_path)
        subject_ids.update(df["subject_id"].unique().to_list())
        
    return subject_ids

def get_ids_from_pkl(directory: str) -> set:
    """Reads all subject_ids from all .pkl files in a directory."""
    subject_ids = set()
    pkl_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"Warning: No .pkl files found in '{directory}'")
        return subject_ids
        
    for file_path in tqdm(pkl_files, desc="Reading tokenized PKL files"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            for patient in data:
                subject_ids.add(patient['subject_id'])
                
    return subject_ids

if __name__ == "__main__":
    print("--- Starting Patient ID Validation ---")
    
    # Get the set of IDs from the source and the output
    raw_subject_ids = get_ids_from_parquet(RAW_DATA_DIR)
    tokenized_subject_ids = get_ids_from_pkl(TOKENIZED_DATA_DIR)
    
    print("\n--- Validation Summary ---")
    print(f"Found {len(raw_subject_ids)} unique subject IDs in the raw data.")
    print(f"Found {len(tokenized_subject_ids)} unique subject IDs in the tokenized output.")
    
    # Compare the sets to find any discrepancies
    missing_from_output = raw_subject_ids - tokenized_subject_ids
    extra_in_output = tokenized_subject_ids - raw_subject_ids
    
    print("\n--- Results ---")
    if not missing_from_output and not extra_in_output:
        print("✅ Success! All subject IDs from the raw data are present in the tokenized output.")
    else:
        if missing_from_output:
            print(f"🚨 Error: {len(missing_from_output)} subject IDs are MISSING from the tokenized output.")
            print("These patients were in the raw data but did not get processed:")
            print(list(missing_from_output))
        
        if extra_in_output:
            print(f"🚨 Warning: {len(extra_in_output)} subject IDs are in the output but not in the raw data (this is unusual).")
            print(list(extra_in_output))
            
    print("\n--- End of Validation ---")
