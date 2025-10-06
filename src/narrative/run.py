import argparse
import yaml
from src.narrative.generator import NarrativeGenerator

if __name__ == "__main__":
    """
    Main entry point for the narrative generation pipeline.
    
    This script reads a configuration file, initializes the NarrativeGenerator,
    and runs the generation process to create a natural language dataset for LLM fine-tuning.
    """
    parser = argparse.ArgumentParser(description="Generate a natural language dataset from tokenized EHR data.")
    parser.add_argument(
        "--config_filepath", 
        type=str, 
        required=True, 
        help="Path to the narrative generation config YAML file."
    )
    args = parser.parse_args()

    # Load the configuration from the specified YAML file
    print(f"Loading configuration from: {args.config_filepath}")
    with open(args.config_filepath, "r") as f:
        config = yaml.safe_load(f)

    # Create an instance of the generator with the loaded config
    generator = NarrativeGenerator(config)
    
    # Run the main generation process
    generator.generate()