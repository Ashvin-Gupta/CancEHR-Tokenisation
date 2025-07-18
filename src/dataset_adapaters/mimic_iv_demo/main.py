import os
import polars as pl
import shutil
import meds
import json
from datetime import datetime

def main(input_dir, output_dir):
    
    # Process input and output directories
    process_input_dir(input_dir)
    process_output_dir(output_dir)

    events = []

    # Patient events
    patients = pl.read_csv(os.path.join(input_dir, "hosp", "patients.csv.gz"), infer_schema_length=0)
    birth_year = pl.col('anchor_year').cast(pl.Int32) - pl.col('anchor_age').cast(pl.Int32)
    birth_event = patients.select(subject_id=pl.col('subject_id').cast(pl.Int64), code=pl.lit(meds.birth_code), time=pl.datetime(birth_year, 1, 1))
    gender_event = patients.select(subject_id=pl.col('subject_id').cast(pl.Int64), code='Gender/' + pl.col('gender'), time=pl.datetime(birth_year, 1, 1))
    death_event = patients.select(subject_id=pl.col('subject_id').cast(pl.Int64), code=pl.lit(meds.death_code), time=pl.col('dod').str.to_datetime()).filter(pl.col('time').is_not_null())

    events.extend([birth_event, gender_event, death_event])

    # Procedure events
    procedures = pl.read_csv(os.path.join(input_dir, "hosp", "procedures_icd.csv.gz"), infer_schema_length=0)
    procedure_event = procedures.select(
                                    # We need code and time
                                    subject_id=pl.col('subject_id').cast(pl.Int64),
                                    code='ICD' + pl.col('icd_version') + '/' + pl.col('icd_code'),
                                    time=pl.col('chartdate').str.to_datetime(),

                                    # We can also include other information
                                    seq_num = pl.col('seq_num').cast(pl.Int64),
                                    hadm_id = pl.col('hadm_id').cast(pl.Int64),
                                    )

    events.append(procedure_event)

    # Organise events by subject_id
    events_table = pl.concat(events, how='diagonal')
    events_table = events_table.sort(pl.col('subject_id'), pl.col('time'))
    
    # Save events table to output directory
    events_table.write_parquet(os.path.join(output_dir, "data", "events.parquet"))

    # Save metadata to json
    metadata = {
        "dataset_name": "mimic_iv_demo",
        "dataset_description": "MIMIC-IV-Demo dataset",
        "dataset_version": "2.2",
        "dataset_url": "https://physionet.org/content/mimic-iv-demo/2.2/",
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_dir, "metadata", "metadata.json"), "w") as f:
        json.dump(metadata, f)

def process_input_dir(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory {input_dir} is not a directory")

    # check hosp and icu folders exist
    hosp_dir = os.path.join(input_dir, "hosp")
    icu_dir = os.path.join(input_dir, "icu")
    if not os.path.exists(hosp_dir):
        raise FileNotFoundError(f"Hosp directory {hosp_dir} does not exist")
    if not os.path.exists(icu_dir):
        raise FileNotFoundError(f"ICU directory {icu_dir} does not exist")

def process_output_dir(output_dir):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist")
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory {output_dir} is not a directory")
    
    # Ask user before emptying output directory
    if os.listdir(output_dir):
        response = input(f"Output directory {output_dir} is not empty. Do you want to empty it? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            raise ValueError("Output directory must be empty to proceed")
    
    # Create data and metadata folders
    data_dir = os.path.join(output_dir, "data")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(data_dir, exist_ok=False)
    os.makedirs(metadata_dir, exist_ok=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/mimic_iv_demo")
    parser.add_argument("--output_dir", type=str, default="data/mimic_iv_demo")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)