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
