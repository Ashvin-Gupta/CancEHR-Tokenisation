import pandas as pd
import pickle
import os
from tqdm import tqdm
import json

class NarrativeGenerator:
    """
    A class to generate a natural language dataset from a tokenized EHR event stream.

    This class reads the output of the tokenization pipeline (.pkl files and vocab)
    and uses lookup tables to translate structured event codes into human-readable
    phrases suitable for fine-tuning a large language model.
    """
    def __init__(self, config: dict):
        """
        Initializes the generator by loading all necessary mapping files from the config.
        
        Args:
            config (dict): A dictionary containing all the file paths and parameters.
        """
        self.config = config
        self._load_mappings()

    def _load_mappings(self):
        """Loads vocabulary, translation lookups, and patient labels into memory."""
        print("Loading all necessary mappings...")
        
        # Load the vocabulary to map token IDs back to their string representations
        vocab_df = pd.read_csv(self.config['vocab_file'])
        self.id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        # Load the medical and lab code translation tables
        medical_lookup_df = pd.read_csv(self.config['medical_lookup_file'])
        self.medical_lookup = pd.Series(medical_lookup_df['term'].values, index=medical_lookup_df['code'].astype(str).str.upper()).to_dict()

        lab_lookup_df = pd.read_csv(self.config['lab_lookup_file'])
        self.lab_lookup = pd.Series(lab_lookup_df['term'].values, index=lab_lookup_df['code'].astype(str).str.upper()).to_dict()

        # Load the patient cancer labels
        labels_df = pd.read_csv(self.config['labels_file'])
        self.subject_to_label_map = pd.Series(labels_df['site'].values, index=labels_df['subject_id']).to_dict()

        # Convert the 'site' column to a binary label (1 for cancer, 0 for control)
        labels_df['label'] = labels_df['site'].apply(lambda x: 1 if x.lower() == 'liver' else 0)
        self.subject_to_label_map = pd.Series(labels_df['label'].values, index=labels_df['subject_id']).to_dict()
        
        print("Mappings loaded successfully.")

    def _translate_token(self, token_string: str) -> str:
        """
        Translates a single structured token string into a human-readable phrase.
        
        Args:
            token_string (str): The token string to translate (e.g., "MEDICAL//hypertension").
            
        Returns:
            str: The translated, human-readable phrase.
        """
        if not isinstance(token_string, str):
            return ""

        if token_string.startswith('<time_interval_'):
            time_part = token_string.split('_')[-1].strip('>')
            return f"{time_part}"
        
        elif token_string.startswith('MEDICAL//'):
            code = token_string.split('//')[1].upper()
            description = self.medical_lookup.get(code, code.replace('_', ' ').title())
            return f"{description}"
            
        elif token_string.startswith('LAB//'):
            code = token_string.split('//')[1].upper()
            description = self.lab_lookup.get(code, code.replace('_', ' ').title())
            return f"Lab {description}"

        elif token_string.startswith(('BMI//', 'HEIGHT//', 'WEIGHT//')):
            parts = token_string.split('//')
            return f"{parts[0]}: {parts[1]}."

        elif token_string.startswith(('GENDER//', 'ETHNICITY//')):
            parts = token_string.split('//')
            # return f"{parts[0].title()}: {parts[1]}."
            return f"{parts[1]}"
        
        elif token_string.startswith('REGION//'):
            parts = token_string.split('//')
            return f"{parts[0].title()}: {parts[1]}"

        elif token_string.startswith('Q') and len(token_string) <= 4 and token_string[1:].isdigit():
            return f"{token_string[1:]}"

        elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
            return ""
            
        else:
            # Default case for any other tokens (like raw numbers that weren't binned)
            return f"{token_string}."

    def generate(self):
        """
        The main method to process all .pkl files, generate the narratives,
        and write the final dataset to a .txt file.
        """
        tokenized_data_dir = self.config['tokenized_data_dir']
        output_txt_file = self.config['output_txt_file']
        
        print(f"Starting narrative generation from data in: {tokenized_data_dir}")

        SIX_MONTHS_IN_SECONDS = 6 * 30.44 * 24 * 60 * 60
        
        with open(output_txt_file, 'w') as f_out:
            pkl_files = [os.path.join(tokenized_data_dir, f) for f in os.listdir(tokenized_data_dir) if f.endswith('.pkl')]
            
            if not pkl_files:
                print(f"Warning: No .pkl files found in {tokenized_data_dir}")
                return

            for pkl_file in tqdm(pkl_files, desc="Processing files"):
                with open(pkl_file, 'rb') as f_in:
                    patient_data_list = pickle.load(f_in)
                    
                    for patient in patient_data_list:
                        subject_id = patient['subject_id']
                        token_ids = patient['tokens']
                        timestamps = patient['timestamps']
                        
                        label = self.subject_to_label_map.get(subject_id)
                        if label is None:
                            continue 

                        if label == 1: # If this is a cancer case
                            # Find the last valid timestamp in the trajectory
                            last_timestamp = max(t for t in timestamps if t is not None and t > 0)
                            
                            if last_timestamp:
                                cutoff_timestamp = last_timestamp - SIX_MONTHS_IN_SECONDS
                                
                                # Filter tokens and timestamps to keep only events before the cutoff
                                truncated_token_ids = []
                                for i, ts in enumerate(timestamps):
                                    # Keep static events (ts=0) and events before the cutoff
                                    if ts == 0 or ts < cutoff_timestamp:
                                        truncated_token_ids.append(token_ids[i])
                                token_ids = truncated_token_ids
                        
                        string_codes = [self.id_to_token_map.get(tid, "") for tid in token_ids]
                        
                        translated_phrases = [self._translate_token(code) for code in string_codes]
                        
                        final_phrases = [phrase for phrase in translated_phrases if phrase]
                        
                        narrative = " ".join(final_phrases)
                        
                        # Create the JSON object for this patient
                        json_line = {
                            "text": narrative,
                            "label": label
                        }
                        
                        # Write the JSON object as a new line in the output file
                        f_out.write(json.dumps(json_line) + '\n')

        print(f"\n✅ Narrative generation complete. Dataset saved to: {output_txt_file}")