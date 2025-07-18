import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

# Files that depend on other files before they can be processed
DEPENDENCIES = {
    "hosp/diagnoses_icd.csv": ["hosp/admissions.csv"],
    "hosp/drgcodes.csv": ["hosp/admissions.csv"], 
    "hosp/patients.csv": ["hosp/admissions.csv"],
}

# Files that need special processing
SPECIAL_PROCESSING_FILES = [
    "hosp/d_icd_diagnoses.csv",
    "hosp/d_icd_procedures.csv",
]

def discover_input_files(input_dir: Path) -> List[Path]:
    """Discover all input files recursively."""
    logger.info(f"Discovering files in {input_dir}")
    
    # Combine subdirectory and root files, remove duplicates
    all_files = list(set(
        list(input_dir.rglob("*/*.*")) + list(input_dir.rglob("*.*"))
    ))
    
    logger.info(f"Found {len(all_files)} files")
    return all_files

def process_independent_files(
    files: List[Path], 
    input_dir: Path, 
    output_dir: Path
) -> Tuple[List[str], Set[str]]:
    """Process files that don't need dependencies."""
    logger.info("Processing independent files...")
    
    files_needing_dependencies = []
    processed_files = set()
    
    for file_path in files:
        file_key = str(file_path.relative_to(input_dir))
        output_path = output_dir / file_path.relative_to(input_dir)
        
        # Skip if already processed
        if output_path.exists():
            processed_files.add(file_key)
            continue
        
        # Check what type of processing is needed
        if file_key in DEPENDENCIES:
            files_needing_dependencies.append(file_key)
        elif file_key in SPECIAL_PROCESSING_FILES:
            # Will be handled in special processing pass
            continue
        else:
            # File needs no processing - copy
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Copying {file_path} -> {output_path}")
                shutil.copy2(file_path, output_path)
                processed_files.add(file_key)
            except Exception as e:
                logger.error(f"Failed to process {file_key}: {e}")
    
    logger.info(f"Processed {len(processed_files)} independent files")
    return files_needing_dependencies, processed_files

def process_dependent_files(
    dependent_files: List[str], 
    input_dir: Path, 
    output_dir: Path
) -> Set[str]:
    """Process files that depend on other files."""
    logger.info("Processing dependent files...")
    
    processed_files = set()
    
    # Group files by their dependencies to minimize loading
    dependency_groups: Dict[str, List[str]] = {}
    for file_key in dependent_files:
        dep_key = DEPENDENCIES[file_key][0]  # Assume single dependency for simplicity
        dependency_groups.setdefault(dep_key, []).append(file_key)
    
    # Process each dependency group
    for dependency_key, dependent_file_list in dependency_groups.items():
        dependency_path = input_dir / dependency_key
        
        if not dependency_path.exists():
            logger.error(f"Dependency file not found: {dependency_path}")
            continue
            
        logger.info(f"Loading dependency: {dependency_key}")
        # Placeholder for actual data loading (would use polars/pandas)
        dependency_data = f"LOADED_{dependency_key}"
        
        # Process all files that depend on this data
        for file_key in dependent_file_list:
            try:
                output_path = output_dir / file_key.replace('.csv', '.parquet')
                
                if output_path.exists():
                    processed_files.add(file_key)
                    continue
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing {file_key} with dependency {dependency_key}")
                
                # Placeholder for actual processing logic
                # In real implementation: load file, apply transformation, save as parquet
                output_path.write_text(f"Processed {file_key} with {dependency_key}")
                processed_files.add(file_key)
                
            except Exception as e:
                logger.error(f"Failed to process dependent file {file_key}: {e}")
    
    logger.info(f"Processed {len(processed_files)} dependent files")
    return processed_files

def process_special_files(input_dir: Path, output_dir: Path) -> Set[str]:
    """Process files that need special handling."""
    logger.info("Processing special files...")
    
    processed_files = set()
    
    for file_key in SPECIAL_PROCESSING_FILES:
        try:
            input_path = input_dir / file_key
            output_path = output_dir / file_key.replace('.csv', '.parquet')
            
            if not input_path.exists():
                logger.warning(f"Special file not found: {input_path}")
                continue
                
            if output_path.exists():
                processed_files.add(file_key)
                continue
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Special processing: {file_key}")
            # Placeholder for special processing (e.g., ICD code formatting)
            output_path.write_text(f"Special processed {file_key}")
            processed_files.add(file_key)
            
        except Exception as e:
            logger.error(f"Failed to process special file {file_key}: {e}")
    
    logger.info(f"Processed {len(processed_files)} special files")
    return processed_files

def main(input_dir: Path, output_dir: Path, do_overwrite: bool = False) -> None:
    """Perform pre-MEDS data wrangling for MIMIC-IV.
    
    Three-pass processing algorithm:
    1. Process independent files (copy)
    2. Process dependent files (with transformations)
    3. Process special files (ICD codes, etc.)
    """
    logger.info(f"Starting MIMIC-IV pre-processing: {input_dir} -> {output_dir}")
    
    # Check if already complete
    done_file = output_dir / ".done"
    if done_file.exists() and not do_overwrite:
        logger.info("Processing already complete. Use do_overwrite=True to reprocess.")
        return
    
    # Create output directory and discover files
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = discover_input_files(input_dir)
    
    # Three-pass processing
    dependent_files, processed_independent = process_independent_files(
        all_files, input_dir, output_dir
    )
    processed_dependent = process_dependent_files(dependent_files, input_dir, output_dir)
    processed_special = process_special_files(input_dir, output_dir)
    
    # Summary and completion
    total_processed = len(processed_independent) + len(processed_dependent) + len(processed_special)
    logger.info(f"Processing complete: {len(processed_independent)} independent, "
                f"{len(processed_dependent)} dependent, {len(processed_special)} special "
                f"({total_processed} total)")
    
    # Mark completion
    done_file.write_text(f"Processing completed at {datetime.now()}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    load_dotenv()

    print(os.getenv("MIMIC_IV_INPUT_DIR"))
    print(os.getenv("MIMIC_IV_PRE_MEDS_DIR"))
    
    mimic_iv_dir = Path(os.getenv("MIMIC_IV_INPUT_DIR"))
    mimic_iv_pre_meds_dir = Path(os.getenv("MIMIC_IV_PRE_MEDS_DIR"))

    print(mimic_iv_dir, mimic_iv_pre_meds_dir)
    
    main(
        input_dir=mimic_iv_dir, 
        output_dir=mimic_iv_pre_meds_dir,
        do_overwrite=False
    )