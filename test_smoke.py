"""Smoke tests for DBTokenizer (v2 — milestones, birth dates, EOS, pair overrides, BPE)."""
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from tokenizer import DBTokenizer

try:
    import rustbpe
    import tiktoken
    HAS_BPE = True
except ImportError:
    HAS_BPE = False

# ════════════════════════════════════════════════════════════════════════════
# Shared synthetic data builder
# ════════════════════════════════════════════════════════════════════════════

def _make_df(include_birth=True, include_admit=False):
    """Build a tiny two-patient DataFrame.

    If *include_birth* is True, a demographic/birth_date row is prepended
    for each patient so the tokenizer can derive age.
    If *include_admit* is True, admission/discharge rows are added.
    """
    rows = []
    # ── Patient 1: born 1980-06-15 ────────────────────────────────────
    if include_birth:
        rows.append({
            "id": 1,
            "time": datetime(1980, 6, 15),
            "class": "demographic",
            "text_value": "birth_date",
            "numeric_value": None,
        })
    if include_admit:
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0),
            "class": "admission",
            "text_value": None,
            "numeric_value": None,
        })
    for i in range(8):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i * 6),
            "class": "lab",
            "text_value": "hemoglobin" if i % 2 == 0 else "glucose",
            "numeric_value": 12.0 + i * 0.5 if i % 2 == 0 else 90.0 + i,
        })
    if include_admit:
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 3, 10, 0),
            "class": "discharge",
            "text_value": None,
            "numeric_value": None,
        })

    # ── Patient 2: born 1955-03-01 ────────────────────────────────────
    if include_birth:
        rows.append({
            "id": 2,
            "time": datetime(1955, 3, 1),
            "class": "demographic",
            "text_value": "birth_date",
            "numeric_value": None,
        })
    for i in range(6):
        rows.append({
            "id": 2,
            "time": datetime(2023, 3, 1, 10, 0) + timedelta(days=i * 10),
            "class": "lab",
            "text_value": "hemoglobin",
            "numeric_value": 11.0 + i * 0.3,
        })

    return pl.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: concept + discrete + factored (with birth dates)")
print("=" * 60)

df = _make_df(include_birth=True)
tok = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="factored",
    n_bins=5,
    level_threshold=10,
    milestone_ip="daily",
    milestone_op="week",
)
tok.train(df)
enc_ids, enc_vals = tok.encode(df)
decoded = tok.decode(enc_ids)
print(f"  Encoded {len(df)} rows -> {len(enc_ids)} tokens")
print(f"  vals is None: {enc_vals is None}")
print(f"  First 30 tokens: {decoded[:30]}")

# Verify SOS, EOS, and age tokens present
assert "<|sos|>" in decoded, "Missing <|sos|>"
assert "<|eos|>" in decoded, "Missing <|eos|>"
# Patient 1 born 1980, events in 2023 → age ≈ 42
ages_p1 = [t for t in decoded if t.startswith("<|age_")]
assert len(ages_p1) > 0, "No age tokens found"
print(f"  Age tokens: {ages_p1}")

# Check EOS count (should be 2 — one per patient)
eos_count = decoded.count("<|eos|>")
assert eos_count == 2, f"Expected 2 <|eos|> tokens, got {eos_count}"
print(f"  EOS count: {eos_count} ✓")

print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: concept + discrete + fused")
print("=" * 60)

tok2 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="fused",
    n_bins=5,
    level_threshold=10,
)
tok2.train(df)
enc_ids2, enc_vals2 = tok2.encode(df)
dbg2 = tok2.encode_debug(df)
print(f"  Encoded {len(df)} rows -> {len(enc_ids2)} tokens")
print("  Sample debug rows:")
for row in dbg2.head(5).iter_rows(named=True):
    print(f"    id={row['id']} cls={row['class']} tv={row['text_value']} "
          f"nv={row['numeric_value']} -> {row['tokens']}")
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: concept + continuous + factored")
print("=" * 60)

tok3 = DBTokenizer(
    text_mode_default="concept",
    num_type="continuous",
    num_seq="factored",
    n_bins=5,
    level_threshold=10,
)
tok3.train(df)
enc_ids3, enc_vals3 = tok3.encode(df)
print(f"  Encoded {len(df)} rows -> {len(enc_ids3)} tokens")
print(f"  vals length: {len(enc_vals3)}, sample vals: {enc_vals3[:10]}")
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: save / load round-trip")
print("=" * 60)

tok.save("/tmp/dbtoken_test")
tok_loaded = DBTokenizer.load("/tmp/dbtoken_test")
enc_ids_rt, _ = tok_loaded.encode(df)
assert enc_ids == enc_ids_rt, "Round-trip encoding mismatch!"
print("  Save/load round-trip: PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: milestone boundary emission (Kalman-filter)")
print("=" * 60)

tok5 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="factored",
    n_bins=5,
    milestone_ip="daily",
    milestone_op="week",
)
tok5.train(df)
dbg5 = tok5.encode_debug(df)

# Patient 2 has events spanning multiple weeks → should produce milestone tokens
p2_debug = dbg5.filter(pl.col("id") == 2)
all_toks = []
for row in p2_debug.iter_rows(named=True):
    all_toks.extend(row["tokens"])
ms_toks = [t for t in all_toks if t.startswith("<|ms_")]
print(f"  Patient 2 milestone tokens: {ms_toks}")
# Patient 2 events span ~50 days across several weeks → multiple week milestones
assert len(ms_toks) > 0, "Expected milestone tokens for patient 2"
print("  Milestone emission: PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: no birth dates → no age tokens, still works")
print("=" * 60)

df_no_birth = _make_df(include_birth=False)
tok6 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="factored",
    n_bins=5,
)
tok6.train(df_no_birth)
enc_ids6, _ = tok6.encode(df_no_birth)
decoded6 = tok6.decode(enc_ids6)
age_toks6 = [t for t in decoded6 if t.startswith("<|age_")]
assert len(age_toks6) == 0, f"Expected no age tokens, got {age_toks6}"
print(f"  No age tokens (correct): ✓")
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 7: milestone_op='none' suppresses OP milestones")
print("=" * 60)

tok7 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="factored",
    n_bins=5,
    milestone_ip="daily",
    milestone_op="none",
)
tok7.train(df)
dbg7 = tok7.encode_debug(df)
# All patients are outpatient (no admission/discharge) → OP milestones
# milestone_op='none' → should produce NO milestone tokens at all
all_toks7 = []
for row in dbg7.iter_rows(named=True):
    all_toks7.extend(row["tokens"])
ms_toks7 = [t for t in all_toks7 if t.startswith("<|ms_")]
assert len(ms_toks7) == 0, f"Expected no milestones with op='none', got {ms_toks7}"
print("  No milestones emitted: ✓")
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 8: pair-level text_mode_overrides")
print("=" * 60)

# Make data with a class that auto-detects as concept, but override
# one specific (class, text_value) pair to concept explicitly
df8 = _make_df(include_birth=True)
tok8 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="fused",
    n_bins=5,
    text_mode_overrides={
        ("lab", "hemoglobin"): "concept",  # tuple-key override
    },
)
tok8.train(df8)
enc_ids8, _ = tok8.encode(df8)
decoded8 = tok8.decode(enc_ids8)
print(f"  Encoded {len(df8)} rows -> {len(enc_ids8)} tokens")
# Should still work and hemoglobin concepts should appear
hb_toks = [t for t in decoded8 if "hemoglobin" in t]
assert len(hb_toks) > 0, "hemoglobin concept tokens missing"
print(f"  hemoglobin tokens: {hb_toks[:5]}...")

# Test save/load with pair overrides
tok8.save("/tmp/dbtoken_test_pair")
tok8_loaded = DBTokenizer.load("/tmp/dbtoken_test_pair")
enc_ids8_rt, _ = tok8_loaded.encode(df8)
assert enc_ids8 == enc_ids8_rt, "Pair-override round-trip mismatch!"
print("  Pair override save/load: ✓")
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 9: admission/discharge with IP milestones")
print("=" * 60)

df9 = _make_df(include_birth=True, include_admit=True)
tok9 = DBTokenizer(
    text_mode_default="concept",
    num_type="discrete",
    num_seq="factored",
    n_bins=5,
    admission_classes=["admission"],
    discharge_classes=["discharge"],
    milestone_ip="8hr",
    milestone_op="week",
    milestone_shift_start=7,
)
tok9.train(df9)
dbg9 = tok9.encode_debug(df9)

# Patient 1 is admitted → should use IP milestones
p1_debug = dbg9.filter(pl.col("id") == 1)
all_toks9 = []
for row in p1_debug.iter_rows(named=True):
    all_toks9.extend(row["tokens"])
ms_8hr = [t for t in all_toks9 if "<|ms_8hr_" in t]
print(f"  Patient 1 IP 8hr milestones: {ms_8hr}")
# Patient 1 has 8 lab events over 42 hours → should cross 8hr shifts
assert len(ms_8hr) > 0, "Expected 8hr IP milestones"
print("  PASS")

# ════════════════════════════════════════════════════════════════════════════
# BPE TESTS (skipped if no BPE backend available)
# ════════════════════════════════════════════════════════════════════════════

def _make_bpe_df():
    """Build a DataFrame with high-cardinality text_values that trigger BPE mode."""
    rows = []
    # Birth date for patient 1
    rows.append({
        "id": 1,
        "time": datetime(1980, 6, 15),
        "class": "demographic",
        "text_value": "birth_date",
        "numeric_value": None,
    })
    # 'notes' class: many unique free-text strings → auto-detected as BPE
    note_texts = [f"Patient presented with symptom variant {i}" for i in range(80)]
    for i, txt in enumerate(note_texts):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i),
            "class": "notes",
            "text_value": txt,
            "numeric_value": None,
        })
    # 'lab' class: few unique text_values → stays concept mode
    lab_texts = ["hemoglobin", "glucose", "creatinine"]
    for i, txt in enumerate(lab_texts * 5):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 5, 8, 0) + timedelta(hours=i),
            "class": "lab",
            "text_value": txt,
            "numeric_value": 10.0 + i * 0.5,
        })
    return pl.DataFrame(rows)


if HAS_BPE:
    print("\n" + "=" * 60)
    print("TEST 10: BPE auto-detect + factored")
    print("=" * 60)

    df_bpe = _make_bpe_df()
    tok10 = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        final_vocab_size=512,
    )
    tok10.train(df_bpe)
    # 'notes' has 80 unique texts → BPE; 'lab' has 3 → concept
    assert tok10.class_text_modes["notes"] == "bpe", \
        f"Expected notes=bpe, got {tok10.class_text_modes['notes']}"
    assert tok10.class_text_modes["lab"] == "concept", \
        f"Expected lab=concept, got {tok10.class_text_modes['lab']}"
    print(f"  Text modes: {tok10.class_text_modes}")

    enc_ids10, enc_vals10 = tok10.encode(df_bpe)
    decoded10 = tok10.decode(enc_ids10)
    print(f"  Encoded {len(df_bpe)} rows -> {len(enc_ids10)} tokens")
    assert enc_vals10 is None, "Expected vals=None for discrete mode"

    # Verify concept lab tokens still appear literally
    assert "hemoglobin" in decoded10, "hemoglobin concept token missing"
    assert "glucose" in decoded10, "glucose concept token missing"
    # Verify BPE produced multi-token sequences (notes are long strings)
    # The total token count should be much larger than row count
    assert len(enc_ids10) > len(df_bpe) * 2, \
        "BPE should expand long text into multiple tokens"

    # Debug encoding should also work
    dbg10 = tok10.encode_debug(df_bpe)
    assert "tokens" in dbg10.columns
    print("  PASS")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 11: BPE + fused (BPE classes fall back to factored)")
    print("=" * 60)

    tok11 = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="fused",
        n_bins=5,
        final_vocab_size=512,
    )
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok11.train(df_bpe)
        # Should warn about BPE classes falling back to factored
        fused_warnings = [x for x in w if "factored" in str(x.message)]
        assert len(fused_warnings) > 0, "Expected fused+BPE fallback warning"
        print(f"  Fused+BPE warning: ✓")

    enc_ids11, _ = tok11.encode(df_bpe)
    decoded11 = tok11.decode(enc_ids11)

    # Lab tokens should use fused concept+level (lab has ≤10 distinct numerics)
    fused_lab = [t for t in decoded11 if "::L" in t]
    assert len(fused_lab) > 0, "Expected fused lab tokens like 'hemoglobin::L3'"
    print(f"  Fused lab tokens: {fused_lab[:5]}...")

    # BPE 'notes' tokens should NOT have fused numerics (notes have no numeric)
    print(f"  Encoded {len(df_bpe)} rows -> {len(enc_ids11)} tokens")
    print("  PASS")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 12: BPE + pair override (force concept for one BPE pair)")
    print("=" * 60)

    tok12 = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        final_vocab_size=512,
        # Override: notes class is BPE by auto, but force one specific
        # text_value to concept mode
        text_mode_overrides={
            ("notes", "Patient presented with symptom variant 0"): "concept",
        },
    )
    tok12.train(df_bpe)
    assert tok12.class_text_modes["notes"] == "bpe"
    assert tok12.pair_text_modes[("notes", "Patient presented with symptom variant 0")] == "concept"

    enc_ids12, _ = tok12.encode(df_bpe)
    decoded12 = tok12.decode(enc_ids12)
    # The overridden text should appear as a single concept token
    assert "Patient presented with symptom variant 0" in decoded12, \
        "Pair-overridden text should appear as concept token"
    print("  Pair override concept token found: ✓")

    # Save/load round-trip with BPE
    tok12.save("/tmp/dbtoken_test_bpe")
    tok12_loaded = DBTokenizer.load("/tmp/dbtoken_test_bpe")
    enc_ids12_rt, _ = tok12_loaded.encode(df_bpe)
    assert enc_ids12 == enc_ids12_rt, "BPE save/load round-trip mismatch!"
    print("  BPE save/load round-trip: ✓")
    print("  PASS")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 13: BPE + continuous + factored")
    print("=" * 60)

    tok13 = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="continuous",
        num_seq="factored",
        n_bins=5,
        final_vocab_size=512,
    )
    tok13.train(df_bpe)
    enc_ids13, enc_vals13 = tok13.encode(df_bpe)
    assert enc_vals13 is not None, "Expected vals for continuous mode"
    assert len(enc_ids13) == len(enc_vals13), "ids and vals length mismatch"
    # Check that some vals are not NaN (numeric positions)
    real_vals = [v for v in enc_vals13 if not (isinstance(v, float) and np.isnan(v))]
    assert len(real_vals) > 0, "Expected some non-NaN vals for lab numerics"
    print(f"  Encoded {len(df_bpe)} rows -> {len(enc_ids13)} tokens, "
          f"{len(real_vals)} numeric vals")
    print("  PASS")

else:
    print("\n" + "=" * 60)
    print("SKIPPED: Tests 10-13 (BPE) — no BPE backend installed")
    print("  Install rustbpe or text_tokenizer to run BPE tests")
    print("=" * 60)

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
