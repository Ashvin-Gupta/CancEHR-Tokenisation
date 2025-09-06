# Preprocessing Components

Preprocessors transform and enrich raw EHR data before tokenization. They use pattern matching to target specific medical codes and apply transformations to either the codes themselves or their associated values.

## Available Preprocessors

### QuantileBinPreprocessor
**Purpose**: Converts continuous numerical values into quantile-based categorical bins.

**Use Cases**: Lab values, vital signs, any continuous measurements that benefit from discretization.

**Configuration:**
```yaml
- type: "quantile_bin"
  matching_type: "starts_with"
  matching_value: "LAB"
  value_column: "numeric_value"
  k: 10
```

**How it works**: Collects values during training, computes quantile boundaries, then maps new values to bins (Q0, Q1, Q2, etc.).

**Example**: `LAB//51237//mg/dL` with `numeric_value=145.2` → `numeric_value="Q7"`

---

### CodeEnrichmentPreprocessor
**Purpose**: Enriches medical codes with human-readable descriptions using lookup tables.

**Use Cases**: Lab codes → names, diagnosis codes → descriptions, drug codes → generic names.

**Configuration:**
```yaml
- type: "code_enrichment"
  matching_type: "starts_with"
  matching_value: "LAB//"
  lookup_file: "/path/to/d_labitems.csv"
  template: "<LAB> {label} ({category}) </LAB>"
  code_column: "itemid"
```

**How it works**: Loads CSV lookup table, creates cache for fast retrieval, replaces codes using string templates.

**Example**: `LAB//51237//mg/dL` → `<LAB> Glucose (Chemistry) </LAB>`

---

### LoadStaticDataPreprocessor
**Purpose**: Loads static subject-level data from CSV files and inserts it as events at the beginning of each subject's timeline.

**Use Cases**: Demographics (race, marital status), insurance information, admission details, or any subject-level characteristics that don't change over time.

**Configuration:**
```yaml
- type: "load_static_data"
  csv_filepath: "/path/to/admissions.csv"
  subject_id_column: "subject_id"
  columns:
    - column_name: "race"
      code_template: "DEMOGRAPHICS//RACE"
      valid_values: ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]
      mappings:
        "PORTUGUESE": "WHITE"
        "MULTIPLE RACE/ETHNICITY": "OTHER"
      map_invalids_to: "RACE_UNKNOWN"
      value_prefix: "RACE//"        # NEW: Adds prefix to values
      insert_code: false            # NEW: Don't insert code into token stream
    - column_name: "marital_status"
      code_template: "DEMOGRAPHICS//MARITAL_STATUS"
      valid_values: ["SINGLE", "WIDOWED", "MARRIED", "DIVORCED"]
      map_invalids_to: "MARITAL_STATUS_UNKNOWN"
      value_prefix: "MARITAL_STATUS//"  # NEW: Adds prefix to values
      insert_code: true                 # NEW: Insert code into token stream
```

**How it works**: Loads CSV data during training, creates subject lookup table with value cleaning/mapping, then inserts static events with `time=null` at the start of each subject's timeline. Subjects not found in the CSV receive default values specified by `map_invalids_to`.

**Example**: For subject 12345 with the above config, inserts events like:
- `code=null`, `text_value="RACE//WHITE"`, `time=null` (race: no code, prefixed value)
- `code="DEMOGRAPHICS//MARITAL_STATUS"`, `text_value="MARITAL_STATUS//MARRIED"`, `time=null` (marital: both code and prefixed value)

**Features:**
- **Value Mapping**: Transform values using custom mappings (e.g., "PORTUGUESE" → "WHITE")
- **Validation**: Specify valid values and map invalid ones to defaults
- **Configurable Codes**: Use any code template format you prefer
- **Value Prefixes**: Add prefixes to values (e.g., "WHITE" → "RACE//WHITE")
- **Code Toggle**: Control whether codes are inserted into token stream per field
- **Multiple Columns**: Process multiple static fields from the same CSV
- **Missing Subject Handling**: Subjects not found in static data get default values from `map_invalids_to` with warnings

---

### EthosQuantileAgePreprocessor
**Purpose**: Converts MEDS_BIRTH events into two quantile-based age tokens using the Ethos age encoding algorithm.

**Use Cases**: Fine-grained age representation for medical timeline models, providing num_quantiles² different age encodings.

**Configuration:**
```yaml
- type: "ethos_quantile_age"
  time_unit: "years"              # Time unit for age calculation (years/days/hours)
  num_quantiles: 10               # Number of quantile bins (provides 100 age representations)
  prefix: "AGE_"                  # Optional prefix for tokens
  insert_t1_code: false           # Whether to insert T1 component code
  insert_t2_code: false           # Whether to insert T2 component code
  keep_meds_birth: false          # Whether to keep original MEDS_BIRTH token
```

**How it works**: 
1. Calculates age as time delta between MEDS_BIRTH and first real medical event
2. Applies Ethos algorithm: `age_scaled = age_delta * num_quantiles² / 100`
3. Splits into components: `age_t1 = floor(age_scaled / num_quantiles)`, `age_t2 = round(age_scaled % num_quantiles)`
4. Replaces MEDS_BIRTH with two quantile events at timestamp=null

**Example**: 
- Time delta: 47 years between birth and first medical event
- Algorithm: `age_scaled = 47`, `age_t1 = 4`, `age_t2 = 7`
- Output events:
  - `code=null`, `text_value="AGE_T1//Q5"`, `time=null` (if insert_t1_code=false)
  - `code=null`, `text_value="AGE_T2//Q8"`, `time=null` (if insert_t2_code=false)

**Features:**
- **Deterministic**: Same age always produces same tokens
- **Fine-grained**: Provides quadratic precision scaling with quantile count
- **Configurable**: Control codes, prefixes, and quantile resolution
- **Static Timing**: Age events use timestamp=null to mark them as metadata/static context
- **Flexible Birth Handling**: Option to keep or replace original MEDS_BIRTH tokens

---

### DemographicAggregationPreprocessor
**Purpose**: Aggregates repeated demographic measurements (BMI, Height, Weight, etc.) into single demographic tokens using quantile binning.

**Use Cases**: Convert multiple measurements of the same type into a single representative demographic value, useful for vital signs, anthropometric data, or any repeated measurements.

**Configuration:**
```yaml
- type: "demographic_aggregation"
  measurements:
    - token_pattern: "BMI (kg/m2)"
      value_column: "text_value"
      aggregation: "median"           # mean, min, max, median
      num_bins: 10
      demographic_code: "BMI"
      prefix: ""                      # Optional prefix for demographic code
      insert_code: false              # Whether to insert demographic code
      remove_original_tokens: true    # Remove original BMI measurements
    - token_pattern: "Height (Inches)"
      value_column: "text_value"
      aggregation: "max"              # Use tallest height measurement
      num_bins: 10
      demographic_code: "HEIGHT"
      prefix: "DEMO_"
      insert_code: false
      remove_original_tokens: false   # Keep original height measurements
```

**How it works**: 
1. Finds all matching measurement tokens for each subject (e.g., all "BMI (kg/m2)" events)
2. Extracts numeric values and applies aggregation method
3. Fits quantile bins during training using all subjects' aggregated values
4. Creates demographic events with quantile-binned values at timestamp=null
5. Optionally removes original measurement tokens from timeline

**Example**: 
- Subject has BMI measurements: [22.1, 23.5, 22.8, 23.0]
- Median aggregation: 22.9
- Quantile binning: Q6 (if median falls in 6th quantile)
- Output: `code=null`, `text_value="BMI//Q6"`, `time=null`
- Original BMI tokens removed if `remove_original_tokens: true`

**Features:**
- **Multiple Measurements**: Process different measurement types in one preprocessor
- **Flexible Aggregation**: Choose mean, min, max, or median per measurement type
- **Data-Driven Binning**: Quantile bins fitted to actual training data distribution
- **Original Token Control**: Keep or remove original measurements per measurement type
- **Configurable Output**: Control codes, prefixes, and bin counts independently

---

## Pattern Matching

All preprocessors support flexible pattern matching:
- `starts_with`: Match codes beginning with pattern
- `ends_with`: Match codes ending with pattern  
- `contains`: Match codes containing pattern
- `equals`: Exact match

## Pipeline Integration

Configure multiple preprocessors in YAML:

```yaml
preprocessing:
  - type: "quantile_bin"
    matching_type: "starts_with"
    matching_value: "LAB"
    value_column: "numeric_value"
    k: 10
  - type: "code_enrichment"
    matching_type: "starts_with"
    matching_value: "LAB//"
    lookup_file: "lookups/labs.csv"
    template: "<LAB> {label} </LAB>"
    code_column: "itemid"
```

Preprocessors are automatically applied during tokenization in the order specified.
