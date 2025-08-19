# Postprocessing Components

Postprocessors operate on tokenized sequences after initial tokenization, adding temporal context or other sequence-level enhancements to improve downstream model performance.

## Available Postprocessors

### TimeIntervalPostprocessor
**Purpose**: Inserts time interval tokens between events to capture temporal patterns in patient timelines.

**Use Cases**: Help models understand timing relationships, capture clinical workflow patterns, add temporal context.

**Configuration:**
```yaml
postprocessing:
  - type: "time_interval"
    interval_tokens:
      "5m-15m":
        min: 5
        max: 15
      "15m-45m":
        min: 15
        max: 45
      "2h-3h":
        min: 120
        max: 180
      "1d-":  # Open-ended interval
        min: 1440
```

**How it works**: Calculates time differences between consecutive events and inserts appropriate interval tokens (e.g., `<time_interval_5m-15m>`) before events.

**Example:**
```
Original: [ADMISSION, LAB//Glucose (30min later), MEDICATION (5h later)]
After:    [ADMISSION, <time_interval_15m-45m>, LAB//Glucose, <time_interval_5h-8h>, MEDICATION]
```

**Key Features:**
- Handles null timestamps gracefully
- Supports open-ended intervals (e.g., "1d-" for 1 day or more)
- Adds tokens to vocabulary for each defined interval

## Pipeline Integration

Postprocessors are automatically applied during tokenization:

```yaml
postprocessing:
  - type: "time_interval"
    interval_tokens:
      "immediate": {"min": 0, "max": 5}
      "short": {"min": 5, "max": 60}
      "medium": {"min": 60, "max": 480}
      "long": {"min": 480, "max": 1440}
      "extended": {"min": 1440}
```

**Considerations:**
- Each interval adds one token to vocabulary
- Time interval tokens increase sequence length
- Choose intervals based on your data distribution and clinical needs
