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
        
        # Step 1: Determine the string label for each patient
        def determine_label(row):
            if row['is_case'] == 0:
                return 'Control'
            else:
                return row['site']
        
        labels_df['string_label'] = labels_df.apply(determine_label, axis=1)

        # Step 2: Create a mapping from string labels to integer IDs
        unique_labels = sorted([label for label in labels_df['string_label'].unique() if label != 'Control'])
        self.label_to_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        self.label_to_id_map['Control'] = 0
        
        print("Created the following label-to-ID mapping:")
        print(self.label_to_id_map)
        
        # Step 3: Map the string labels to their new integer IDs
        labels_df['label_id'] = labels_df['string_label'].map(self.label_to_id_map)
        self.subject_to_label_map = pd.Series(labels_df['label_id'].values, index=labels_df['subject_id']).to_dict()
        
        # Step 4: Add cancer date to the mapping
        labels_df['cancerdate'] = pd.to_datetime(labels_df['cancerdate'], errors='coerce')
        self.subject_to_cancer_date_map = pd.Series(labels_df['cancerdate'].values, index=labels_df['subject_id']).to_dict()
        # self.subject_to_cancer_date_map = {k: v.strftime('%Y-%m-%d') if pd.notna(v) else None for k, v in self.subject_to_cancer_date_map.items()}

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
        
        elif token_string.startswith('AGE_'):
            return f"{token_string}"
        
        elif token_string.startswith('MEDICAL//'):
            code = token_string.split('//')[1].upper()
            description = self.medical_lookup.get(code, code.replace('_', ' ').title())
            return f"{description}"
        
        elif token_string.startswith('MEASUREMENT//'):
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
            return f"Unknown"

    def generate(self):
        """
        Main method to process .pkl files for train, tuning, and held_out splits,
        and write a separate .jsonl file for each.
        """
        tokenized_data_root = self.config['tokenized_data_root']
        output_narrative_dir = self.config['output_narrative_dir']
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_narrative_dir, exist_ok=True)
        
        print(f"Starting narrative generation...")
        print(f"Reading from: {tokenized_data_root}")
        print(f"Writing to:   {output_narrative_dir}")

        # Loop through each of the data splits
        for split in ["train", "tuning", "held_out"]:
            print(f"\n--- Processing split: {split} ---")
            
            input_dir = os.path.join(tokenized_data_root, split)
            output_file = os.path.join(output_narrative_dir, f"{split}.jsonl")
            
            with open(output_file, 'w') as f_out:
                pkl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pkl')]
                
                if not pkl_files:
                    print(f"Warning: No .pkl files found for split '{split}' in {input_dir}")
                    continue

                for pkl_file in tqdm(pkl_files, desc=f"Processing {split} files"):
                    with open(pkl_file, 'rb') as f_in:
                        patient_data_list = pickle.load(f_in)
                        
                        for patient in patient_data_list:
                            subject_id = patient['subject_id']
                            
                            label = self.subject_to_label_map.get(subject_id)
                            if pd.isna(label):
                                continue
                            
                            token_ids = patient['tokens']
                            string_codes = [self.id_to_token_map.get(tid, "") for tid in token_ids]
                            translated_phrases = [self._translate_token(code) for code in string_codes]
                            final_phrases = [phrase for phrase in translated_phrases if phrase]
                            narrative = ", ".join(final_phrases)
                            
                            json_line = {
                                "subject_id": subject_id,
                                "text": narrative,
                                "label": int(label),
                                "timestamps": patient['timestamps'],
                                "cancer_date": cancer_date.isoformat() if pd.notna(cancer_date) else None
                                # "cancer_date": self.subject_to_cancer_date_map.get(subject_id, None)
                            }
                            
                            f_out.write(json.dumps(json_line) + '\n')

        print(f"\n✅ Narrative generation complete. Datasets saved in: {output_narrative_dir}")