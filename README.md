# DBTokenizer

A tokenizer for long-format database and EHR tabular data, designed to convert
structured clinical records into dense token sequences suitable for autoregressive models.

## Input Schema

Every input DataFrame must have exactly five columns:

| Column | Type | Description |
|---|---|---|
| `id` | int | Patient / entity identifier |
| `time` | datetime | Timestamp of the event |
| `class` | str | Event category (e.g. `lab`, `medication`, `vital`) |
| `text_value` | str \| null | Text label or free-text content |
| `numeric_value` | float \| null | Associated numeric measurement |

Rows are sorted by `(id, time)` during encoding. Each patient's sequence is
bookended by `<|sos|>` / `<|eos|>` tokens.

---

## Two-Stage Workflow

```python
from tokenizer import DBTokenizer

tok = DBTokenizer(...)   # configure
tok.train(df)            # learn vocab, numeric transforms, time-delta fits
ids, vals = tok.encode(df)  # produce token sequences
```

### Training (`train`)

1. **Birth-date extraction** — rows matching `(birth_date_class, birth_date_text_value)` provide per-patient birth dates for age tokens.
2. **Text-mode resolution** — each `class` is assigned `"bpe"` or `"concept"` text mode (auto-detected by cardinality threshold, or explicitly overridden).
3. **Numeric transform fitting** — per `(class, text_value)` group:
   - ≤ `level_threshold` distinct values → categorical **level** tokens (`<|L0|>`, `<|L1|>`, …)
   - Otherwise → quantile **bins** (discrete mode) or distribution **scaling** (continuous mode)
4. **Time-delta fitting** — separate transforms for inpatient (`delta_time_ip`) and outpatient (`delta_time_op`) time gaps.
5. **Vocabulary construction** — special tokens + concept tokens + (optionally) BPE merges.

### Encoding (`encode`)

Returns `(ids, vals)`:
- `ids`: flat list of integer token IDs
- `vals`: parallel list of floats (continuous mode) or `None` (discrete mode)

---

## Text Tokenization Modes

### Concept Mode

Each unique `text_value` becomes a single vocabulary token. Best for columns
with low cardinality (lab names, vital signs, medication routes).

### BPE Mode

Free text is tokenized with byte-pair encoding. Training uses
[rustbpe](https://github.com/karpathy/rustbpe); inference uses
[tiktoken](https://github.com/openai/tiktoken) for fast batch encoding.

By default, `text_mode_default="auto"` selects BPE for classes with more than
`text_mode_threshold` (default 64) unique text values, and concept for the rest.

### Per-Class and Per-Pair Overrides

```python
tok = DBTokenizer(
    text_mode_overrides={
        "medications": "bpe",                         # class-level
        ("notes", "discharge_summary"): "concept",    # pair-level
    },
)
```

---

## Numeric Tokenization

### Level Tokens

When a `(class, text_value)` group has ≤ `level_threshold` distinct numeric
values, each value maps to a categorical token `<|L0|>` … `<|Ln|>`.

### Discrete Bins

For higher-cardinality numeric groups, quantile-based bins produce tokens
`<|Q0|>` … `<|Q{n_bins}|>`.

### Continuous Scaling

Distribution-aware scaling (normal, lognormal, gamma, minmax) transforms each
value into a standardised float. The token stream carries a `<|NUM|>` marker
and the scaled float is stored in the parallel `vals` array.

### Fused vs Factored Placement

| Mode | Token Stream Example |
|---|---|
| **Factored** | `<|lab|>`, `hemoglobin`, `<|L4|>` |
| **Fused** | `<|lab|>`, `hemoglobin::L4` |

Fused mode merges the text token and its numeric level/bin into a single
compound token (using `::` as delimiter). This reduces sequence length but
is only available for concept-mode text groups. BPE-mode groups automatically
fall back to factored placement.

---

## Time Tokens

### Time Deltas

Two token families track inter-event time gaps:

- `<|delta_time_ip|>` — inpatient (between admission and discharge)
- `<|delta_time_op|>` — outpatient (default)

In **discrete** mode, time deltas are binned: `<|delta_time_op_Q3|>`.
In **continuous** mode, a scaled float is emitted alongside the delta token.

Inpatient / outpatient state is tracked automatically from
`admission_classes` and `discharge_classes`.

### Milestones

Configurable calendar-boundary markers injected when the time gap crosses
a milestone epoch. Uses a Kalman-filter style: only the *most recent*
milestone is emitted (not every intermediate one).

| Frequency | Token Example | Config |
|---|---|---|
| Week of year | `<\|ms_week_12\|>` | `milestone_op="week"` |
| Month | `<\|ms_month_3\|>` | `milestone_op="month"` |
| Daily | `<\|ms_day_5\|>` | `milestone_ip="daily"` |
| 12-hour shift | `<\|ms_12hr_1\|>` | `milestone_ip="12hr"` |
| 8-hour shift | `<\|ms_8hr_2\|>` | `milestone_ip="8hr"` |
| None | *(suppressed)* | `milestone_op="none"` |

### Age Tokens

`<|age_N|>` is emitted once per patient or whenever the patient's integer age
changes. Birth dates are extracted from rows where `class == birth_date_class`
and `text_value == birth_date_text_value`.

---

## Save / Load

```python
tok.save("path/to/tokenizer")
# writes tokenizer.json + tokenizer.bpe (if BPE enabled)

tok2 = DBTokenizer.load("path/to/tokenizer")
```

JSON stores all configuration and learned parameters. The BPE encoding object
is pickled separately (tiktoken `Encoding`).

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode_default` | str | `"auto"` | `"auto"`, `"bpe"`, or `"concept"` |
| `text_mode_overrides` | dict | `None` | Per-class or per-(class, text_value) overrides |
| `text_mode_threshold` | int | `64` | Auto mode: classes with > threshold unique values use BPE |
| `num_type` | str | `"discrete"` | `"discrete"` or `"continuous"` |
| `num_seq` | str | `"factored"` | `"fused"` or `"factored"` |
| `n_bins` | int | `10` | Number of quantile bins (discrete) |
| `level_threshold` | int | `10` | Max distinct values for level tokens |
| `bin_clip_min` | float | `1.0` | Lower percentile clip for binning |
| `bin_clip_max` | float | `99.0` | Upper percentile clip for binning |
| `continuous_clip_min` | float | `1.0` | Lower percentile clip for scaling |
| `continuous_clip_max` | float | `99.0` | Upper percentile clip for scaling |
| `distributions` | dict | `None` | Per-group distribution overrides for continuous mode |
| `final_vocab_size` | int | `4096` | Target BPE vocab size (including specials) |
| `split_pattern` | str | GPT-4 pattern | Regex for BPE pre-tokenization |
| `bpe_training_sample` | int\|float | `None` | Subsample BPE training texts |
| `bpe_training_seed` | int | `42` | RNG seed for training subsampling |
| `admission_classes` | list | `[]` | Class names that mark hospital admission |
| `discharge_classes` | list | `[]` | Class names that mark hospital discharge |
| `milestone_ip` | str | `"daily"` | Inpatient milestone frequency |
| `milestone_op` | str | `"week"` | Outpatient milestone frequency |
| `milestone_shift_start` | int | `7` | Hour for shift-based milestone start |
| `birth_date_class` | str | `"demographic"` | Class name for birth-date rows |
| `birth_date_text_value` | str | `"birth_date"` | text_value for birth-date rows |

---

## Dependencies

- **polars** — DataFrame processing
- **numpy** — numeric transforms
- **rustbpe** — BPE training (optional, only for BPE text mode)
- **tiktoken** — BPE inference (optional, only for BPE text mode)

Install BPE dependencies:

```bash
pip install tiktoken
# rustbpe: build from source (https://github.com/karpathy/rustbpe)
```

---

## Additional Methods

| Method | Description |
|---|---|
| `encode_debug(df)` | Returns the DataFrame with a `tokens` column showing human-readable token strings per row |
| `decode(ids)` | Decode a list of token IDs back to token strings |
| `decode_token(id)` | Decode a single token ID |
| `scale(params, x)` | Apply learned scaling transform |
| `unscale(params, x)` | Invert scaling transform |
