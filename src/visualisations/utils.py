import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import gc
import datetime

def time_delta_to_str(time_delta: float):
    """
    Convert a time delta in seconds to a readable format showing the time unit.
    
    Args:
        time_delta (float): Time delta in seconds
        
    Returns:
        str: Formatted string (e.g., "+2.3 years", "+45 days", "+3 hours")
    """
    if time_delta == 0:
        return "T0"
    
    # Convert to different units and find the most appropriate one
    years = time_delta / (365.25 * 24 * 3600)
    days = time_delta / (24 * 3600)
    hours = time_delta / 3600
    minutes = time_delta / 60
    
    if years >= 1:
        return f"+{years:.1f} years"
    elif days >= 1:
        return f"+{days:.0f} days"
    elif hours >= 1:
        return f"+{hours:.0f} hours"
    elif minutes >= 1:
        return f"+{minutes:.0f} minutes"
    else:
        return f"+{time_delta:.0f} seconds"

def calculate_subject_id_to_ehr_shard_mappings(ehr_dataset_dir: str):
    """
    Calculate a mapping dictionary from subject id to the file path of the ehr shard that contains the data for that subject.

    Args:
        ehr_dataset_dir (str): the directory containing the ehr shards

    Returns:
        dict: a mapping dictionary from subject id to the file path of the ehr shard that contains the data for that subject
    """
    patient_to_ehr_shard_mappings = {}

    pickle_files = [os.path.join(ehr_dataset_dir, file) for file in os.listdir(ehr_dataset_dir) if file.endswith(".pkl")]
    for pickle_file in tqdm(pickle_files, desc="Calculating patient to ehr shard mappings"):
        with open(pickle_file, "rb") as f:
                file_data = pkl.load(f)
                for datapoint in file_data:
                    patient_to_ehr_shard_mappings[datapoint['subject_id']] = pickle_file
    return patient_to_ehr_shard_mappings

def calculate_subject_id_to_clinical_note_shard_mappings(clinical_note_dir: str):
    """
    Calculate a mapping dictionary from subject id to the file path of the clinical note shard that contains the data for that subject.

    Args:
        clinical_note_dir (str): the directory containing the clinical note shards

    Returns:
        dict: a mapping dictionary from subject id to the file path of the clinical note shard that contains the data for that subject
    """
    patient_to_clinical_note_shard_mappings = {}
    pickle_files = [os.path.join(clinical_note_dir, file) for file in os.listdir(clinical_note_dir) if file.endswith(".pkl")]
    for pickle_file in tqdm(pickle_files, desc="Calculating patient to clinical note shard mappings"):
        with open(pickle_file, "rb") as f:
            file_data = pkl.load(f)
            for datapoint in file_data:
                patient_to_clinical_note_shard_mappings[datapoint['subject_id']] = pickle_file
    return patient_to_clinical_note_shard_mappings

def get_subject_id_filepath_lookup_table(ehr_dataset_dir: str, clinical_note_dir: str = None):
    """
    Get the subject id to data file lookup table.

    Args:
        ehr_dataset_dir (str): the directory containing the ehr shards
        clinical_note_dir (str): the directory containing the clinical note shards

    Returns:
        pd.DataFrame: a dataframe with the subject id and the file path of the data file that contains the data for that subject
    """

    subject_id_to_ehr_shard_mappings = calculate_subject_id_to_ehr_shard_mappings(ehr_dataset_dir)
    subject_id_to_clinical_note_shard_mappings = calculate_subject_id_to_clinical_note_shard_mappings(clinical_note_dir) if clinical_note_dir is not None else None

    data = []
    for subject_id in subject_id_to_ehr_shard_mappings.keys():
        data.append({
            'subject_id': subject_id,
            'ehr_shard_filepath': subject_id_to_ehr_shard_mappings[subject_id],
            'clinical_note_shard_filepath': subject_id_to_clinical_note_shard_mappings[subject_id] if subject_id_to_clinical_note_shard_mappings is not None and subject_id in subject_id_to_clinical_note_shard_mappings else None
        })

    df = pd.DataFrame(data, columns=['subject_id', 'ehr_shard_filepath', 'clinical_note_shard_filepath'])

    return df

def get_get_subject_ehr_datapoint(subject_id: str, ehr_shard_filepath: str):
    """
    Get the ehr datapoint for a single subject.

    Args:
        subject_id: the id of the subject
        ehr_shard_filepath: the filepath of the ehr shard that contains the data for that subject

    Returns:
        dict: the datapoint for the subject
    """
    with open(ehr_shard_filepath, "rb") as f:
        ehr_data = pkl.load(f)
        for datapoint in ehr_data:
            if datapoint['subject_id'] == subject_id:
                return datapoint

def get_get_subject_clinical_note_datapoint(subject_id: str, clinical_note_shard_filepath: str):
    """
    Get the clinical note datapoint for a single subject.

    Args:
        subject_id: the id of the subject
        clinical_note_shard_filepath: the filepath of the clinical note shard that contains the data for that subject

    Returns:
        dict: the datapoint for the subject
    """
    with open(clinical_note_shard_filepath, "rb") as f:
        clinical_note_data = pkl.load(f)
        for datapoint in clinical_note_data:
            if datapoint['subject_id'] == subject_id:
                return datapoint

def get_subject_datapoint(subject_id: str, ehr_shard_filepath: str, clinical_note_shard_filepath: str):
    """
    Get the datapoint for a single subject.

    Args:
        subject_id: the id of the subject
        ehr_shard_filepath: the filepath of the ehr shard that contains the data for that subject
        clinical_note_shard_filepath: the filepath of the clinical note shard that contains the data for that subject

    Returns:
        dict: the datapoint for the subject
    """
    ehr_data = get_get_subject_ehr_datapoint(subject_id, ehr_shard_filepath)

    if clinical_note_shard_filepath is not None:
        clinical_note_data = get_get_subject_clinical_note_datapoint(subject_id, clinical_note_shard_filepath)
    else:
        clinical_note_data = None
    
    datapoint = {
        'subject_id': int(subject_id),
        'ehr_data': {
            'token_ids': ehr_data['tokens'],
            'timestamps': ehr_data['timestamps']
        },
        'clinical_note_data': clinical_note_data['notes'] if clinical_note_data is not None else None
    }
    return datapoint

if __name__ == "__main__":
    # EHR_PATIENT_TO_SHARD_MAPPINGS = calculate_patient_to_ehr_shard_mappings("/home/joshua/data/mimic/mimic_iv/meds/mimic_iv_meds/tokenized_data/ethos_timetokens/tuning")
    # print(len(EHR_PATIENT_TO_SHARD_MAPPINGS))

    # CLINICAL_NOTE_PATIENT_TO_SHARD_MAPPINGS = calculate_patient_to_clinical_note_shard_mappings("/home/joshua/data/mimic/mimic_iv/tokenized_notes")
    # print(len(CLINICAL_NOTE_PATIENT_TO_SHARD_MAPPINGS))

    # df = get_subject_id_filepath_lookup_table("/home/joshua/data/mimic/mimic_iv/meds/mimic_iv_meds/tokenized_data/ethos_timetokens/tuning", "/home/joshua/data/mimic/mimic_iv/tokenized_notes")
    # print(df.info())
    # print(df.head())

    # subject_datapoint(10032725, "/home/joshua/data/mimic/mimic_iv/meds/mimic_iv_meds/tokenized_data/ethos_timetokens/tuning/19.pkl", "/home/joshua/data/mimic/mimic_iv/tokenized_notes/radiology_shard_0000.pkl")
    pass