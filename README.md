# DBTokenizer

A tokenizer for long-format database and EHR tabular data, designed to convert
structured clinical records into dense token sequences suitable for autoregressive models.

## Installation

```bash
pip install dbtoken
```

Optional extras:

```bash
pip install "dbtoken[bpe]"      # BPE training (rustbpe) + inference (tiktoken)
pip install "dbtoken[metrics]"  # bits-per-row evaluation metrics (requires PyTorch)
pip install "dbtoken[dev]"      # development dependencies
```

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
from dbtoken import DBTokenizer

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
4. **Time-delta fitting** — one distribution per temporal scale band (see [Temporal Scales](#temporal-scales)).
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
        "admission": "bpe",               # class-level: force BPE for class "admission"
        ("notes", "N/A"): "concept",      # pair-level: force concept for "N/A" in "notes"
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

Equal-mass percentile edges collapse when values cluster (a point mass larger
than `1 / n_bins` makes several edges land on the same number). Training snaps
each cut onto a gap between consecutive unique values, then reallocates leftover
cuts by splitting the heaviest remaining bin. The result is as many strictly
increasing edges as the data can support, up to `n_bins - 1`. Time-delta bins
use the same procedure.

### Continuous Scaling

Distribution-aware scaling (normal, lognormal, gamma, minmax) transforms each
value into a standardised float. The token stream carries a `<|NUM|>` marker
and the scaled float is stored in the parallel `vals` array.

### Fused vs Factored Placement

| Mode | Token Stream Example |
|---|---|
| **Factored** | `<\|lab\|>`, `hemoglobin`, `<\|L4\|>` |
| **Fused** | `<\|lab\|>`, `hemoglobin::L4` |

Fused mode merges the text token and its numeric level/bin into a single
compound token (using `::` as delimiter). This reduces sequence length but
is only available for concept-mode text groups. BPE-mode groups automatically
fall back to factored placement.

---

## Time Tokens

### Temporal Scales

Time-delta distributions are split by user-specified thresholds. Each band
gets its own fitted distribution and token family. This cleanly handles data
with multiple time scales (e.g. short inpatient gaps vs. long outpatient gaps)
without any hard-coded state logic.

```python
# Single global distribution (default — no thresholds)
tok = DBTokenizer()
# → tokens: <|delta_time_0|>

# Two bands split at 24 hours
tok = DBTokenizer(time_scales=[86400.0])
# → tokens: <|delta_time_0|>  (≤ 24 h)
#           <|delta_time_1|>  (> 24 h)

# Named bands
tok = DBTokenizer(
    time_scales=[86400.0],
    time_scale_names=["short", "long"],
)
# → tokens: <|delta_time_short|>, <|delta_time_long|>
```

In **discrete** mode, the delta is also binned: `<|delta_time_short|>` `<|Q3|>`,
or as a single fused token `<|delta_time_short_Q3|>`.  
In **continuous** mode, a scaled float is emitted alongside the marker token.

### Milestones

Calendar-boundary markers are injected when a time gap crosses an epoch
boundary. Uses a Kalman-filter style: only the *most recent* milestone is
emitted (not every intermediate one). Milestone granularity is configured
per state (see [State Machine](#state-machine)).

| Granularity | Token Example | Config Value |
|---|---|---|
| Week of year | `<\|ms_week_12\|>` | `"week"` |
| Month | `<\|ms_month_3\|>` | `"month"` |
| Day of week | `<\|ms_day_5\|>` | `"daily"` |
| 12-hour shift | `<\|ms_12hr_1\|>` | `"12hr"` |
| 8-hour shift | `<\|ms_8hr_2\|>` | `"8hr"` |
| None | *(suppressed)* | `"none"` |

The `milestone_shift_start` parameter (default `7`, i.e. 07:00) sets the
hour-of-day at which shift-based boundaries (`8hr`, `12hr`) begin.

### Age Tokens

`<|age_N|>` is emitted once per patient or whenever the patient's integer age
changes. Birth dates are extracted from rows where `class == birth_date_class`
and `text_value == birth_date_text_value`.

---

## State Machine

A lightweight state machine controls which milestone granularity is used at
each point in a patient's sequence. State does **not** affect time-delta
distributions (those are purely threshold-based).

```python
tok = DBTokenizer(
    # State definitions
    state_transitions={
        "inpatient":  ["admission"],   # entering "admission" event → inpatient state
        "outpatient": ["discharge"],   # entering "discharge" event → outpatient state
    },
    initial_state="outpatient",        # state before the first transition

    # Milestone granularity per state
    milestone_per_state={
        "inpatient":  "8hr",   # 8-hour shift markers during admission
        "outpatient": "week",  # weekly markers otherwise
    },
    milestone_shift_start=7,           # shifts start at 07:00
)
```

**Rules:**
- `state_transitions` maps state name → list of `class` values that trigger entry into that state.
- Transitions use **last-trigger-wins**: if a row's `class` matches multiple states, the last match in dict iteration order wins.
- The state is reset to `initial_state` at the start of each patient.
- State transition happens **after** the current row's time-delta and milestone are emitted, so the triggering event itself is encoded under the previous state.
- States not listed in `milestone_per_state` silently use `"none"` (no milestones).

**Minimal example — milestones only, no time-scale split:**

```python
tok = DBTokenizer(
    milestone_per_state={"default": "week"},
)
```

---

## Save / Load

```python
tok.save("path/to/tokenizer")
# writes tokenizer.json (+ tokenizer.bpe if BPE enabled)

tok2 = DBTokenizer.load("path/to/tokenizer")
```

JSON stores all configuration and learned parameters. The BPE encoding object
is saved separately (tiktoken `Encoding`).

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode_default` | str | `"auto"` | `"auto"`, `"bpe"`, or `"concept"` |
| `text_mode_overrides` | dict | `None` | Per-class or per-`(class, text_value)` mode overrides |
| `text_mode_threshold` | int | `64` | Auto mode: classes with > threshold unique values use BPE |
| `num_type` | str | `"discrete"` | `"discrete"` or `"continuous"` |
| `num_seq` | str | `"factored"` | `"fused"` or `"factored"` |
| `n_bins` | int | `10` | Number of quantile bins (discrete mode) |
| `level_threshold` | int | `10` | Max distinct values for level tokens |
| `bin_clip_min` | float | `1.0` | Lower percentile clip for binning |
| `bin_clip_max` | float | `99.0` | Upper percentile clip for binning |
| `continuous_clip_min` | float | `1.0` | Lower percentile clip for scaling |
| `continuous_clip_max` | float | `99.0` | Upper percentile clip for scaling |
| `distributions` | dict | `None` | Per-group distribution overrides for continuous mode |
| `final_vocab_size` | int | `4096` | Target BPE vocab size (including specials) |
| `split_pattern` | str | GPT-4 pattern | Regex for BPE pre-tokenization |
| `bpe_training_sample` | int\|float | `None` | Subsample BPE training texts |
| `bpe_training_seed` | int | `42` | RNG seed for BPE training subsampling |
| `time_scales` | list[float] | `None` | Sorted thresholds (seconds) splitting time-delta bands, e.g. `[86400.0]` |
| `time_scale_names` | list[str] | `None` | Names for each band; length must equal `len(time_scales) + 1` |
| `state_transitions` | dict | `None` | Maps state name → list of `class` values that trigger it |
| `initial_state` | str | `"default"` | Starting state for each patient |
| `milestone_per_state` | dict | `None` | Maps state name → milestone granularity (`"week"`, `"month"`, `"daily"`, `"12hr"`, `"8hr"`, `"none"`) |
| `milestone_shift_start` | int | `7` | Hour-of-day for shift-based milestone boundaries |
| `birth_date_class` | str | `"demographic"` | Class name for birth-date rows |
| `birth_date_text_value` | str | `"birth_date"` | `text_value` for birth-date rows |

---

## Dependencies

- **polars** — DataFrame processing
- **numpy** — numeric transforms
- **rustbpe** — BPE training (optional, only for BPE text mode)
- **tiktoken** — BPE inference (optional, only for BPE text mode)

Install BPE support:

```bash
pip install "dbtoken[bpe]"
```

---

## API Reference

| Method | Description |
|---|---|
| `train(df)` | Fit vocab, numeric transforms, and time-delta distributions |
| `encode(df)` | Encode DataFrame → `(ids, vals)` |
| `encode_debug(df)` | Returns the DataFrame with a `tokens` column of human-readable strings per row |
| `decode(ids)` | Decode a list of token IDs back to token strings |
| `decode_token(id)` | Decode a single token ID |
| `decode_to_dataframe(ids, vals, reference_time, start_patient_id)` | Reconstruct a DataFrame from a token stream |
| `classify_token_ids(ids)` | Classify each token ID into a type string (`"sos"`, `"eos"`, `"time_delta"`, `"class"`, `"concept"`, `"Q"`, `"L"`, `"milestone"`, `"age"`, etc.) |
| `get_numeric_context(ids)` | Return the fitted numeric params dict for each token position that needs density adjustment |
| `save(path)` | Serialise to `<path>.json` (+ `<path>.bpe` if BPE enabled) |
| `load(path)` | Class method — deserialise from disk |
| `scale(params, x)` | Apply learned scaling transform |
| `unscale(params, x)` | Invert scaling transform |

---

## Development

```bash
git clone https://github.com/AJLozaLab/dbtoken.git
cd dbtoken
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```