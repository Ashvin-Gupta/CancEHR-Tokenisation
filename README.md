# Electronic Health Record (EHR) Tokenization

A Python library for tokenizing electronic health record (EHR) data in MEDS format, designed for machine learning applications in healthcare. Edited for CPRD data.

## Features

- **Flexible Tokenization**: Word-level tokenization with configurable vocabulary sizes
- **Data Preprocessing**: 
  - Quantile binning for numerical values (lab results, vitals)
  - Code enrichment with human-readable descriptions via lookup tables
- **Time-aware Processing**: Time interval tokens for capturing temporal patterns
- **MEDS Format Support**: Built for the MEDS (Medical Event Data Standard) format
- **Configurable Pipelines**: YAML-based configuration for reproducible experiments

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```bash
python -m src.pipelines.run --config_filepath src/pipelines/config/nightingale_no_code_enrich.yaml --run_name my_experiment
```

## Pipeline Components

### Preprocessing
- **Quantile Binning**: Discretizes continuous values (lab results, vitals) into quantile-based bins
- **Code Enrichment**: Enriches medical codes with human-readable descriptions using lookup tables

### Tokenization
- **Word-Level Tokenizer**: Creates vocabularies from medical events with configurable size limits
- Support for event, numeric, and text tokens

### Postprocessing
- **Time Interval Tokens**: Adds temporal context by binning time intervals between events

## Configuration

Configure your tokenization pipeline using YAML files (see `src/pipelines/config/` for examples):

```yaml
data:
  path: "/path/to/meds/data"

preprocessing:
  - type: "quantile_bin"
    matching_type: "starts_with"
    matching_value: "LAB"
    value_column: "numeric_value"
    k: 10

tokenization:
  tokenizer: "word_level"
  vocab_size: 5500
  insert_event_tokens: False
```

## Data Format

Expects MEDS-formatted data with the following structure:
```
data/
├── train/
│   ├── 0.parquet
│   └── ...
├── tuning/
│   └── ...
└── held_out/
    └── ...
```

## Dependencies

- `polars`: Fast DataFrame operations
- `meds`: Medical Event Data Standard
- `tqdm`: Progress tracking

Built and tested with MIMIC-IV data.
