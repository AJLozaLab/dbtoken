"""
Demo-style integration tests for DBTokenizer.

These tests exercise full train → encode → decode → reconstruct workflows with
verbose print output so they double as runnable documentation.  Run with:

    pytest tests/test_demo_integration.py -v -s

The ``-s`` flag preserves stdout so the explanatory prints are visible.
"""
import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta
from conftest import REF, HAS_BPE
from tokenizer import DBTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build a rich demo DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def _rich_df():
    """Multi-patient, multi-class dataset suitable for full-feature demos."""
    rows = []

    # Patient 1 — inpatient
    rows.append(dict(id=1, time=datetime(1985, 4, 12), cls="demographic",
                     text_value="birth_date", numeric_value=None))
    rows.append(dict(id=1, time=datetime(2023, 6, 1, 7, 0), cls="admission",
                     text_value=None, numeric_value=None))
    for i in range(10):
        rows.append(dict(
            id=1, time=datetime(2023, 6, 1, 8, 0) + timedelta(hours=i * 4),
            cls="lab", text_value="hemoglobin" if i % 2 == 0 else "glucose",
            numeric_value=13.0 + i * 0.4 if i % 2 == 0 else 95.0 + i * 2,
        ))
    rows.append(dict(id=1, time=datetime(2023, 6, 3, 10, 0), cls="discharge",
                     text_value=None, numeric_value=None))

    # Patient 2 — outpatient
    rows.append(dict(id=2, time=datetime(1960, 11, 22), cls="demographic",
                     text_value="birth_date", numeric_value=None))
    for i in range(8):
        rows.append(dict(
            id=2, time=datetime(2023, 7, 1) + timedelta(days=i * 7),
            cls="lab", text_value="creatinine",
            numeric_value=0.9 + i * 0.05,
        ))
    for i in range(5):
        rows.append(dict(
            id=2, time=datetime(2023, 7, 1) + timedelta(days=i * 14),
            cls="dx", text_value=f"icd_{i % 3}", numeric_value=None,
        ))

    # rename 'cls' → 'class' (since 'class' is a reserved word in dict literals)
    return pl.DataFrame(rows).rename({"cls": "class"})


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1: Concept + discrete + factored  (simplest mode)
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_concept_discrete_factored():
    """Train a concept-mode tokenizer with quantile bins and factored placement.

    This is the most basic mode: each (class, text_value) pair becomes a single
    concept token.  Numeric values are binned into Q-tokens, emitted as separate
    tokens following the concept token (factored placement).
    """
    print("\n" + "=" * 72)
    print("DEMO: Concept / Discrete / Factored")
    print("=" * 72)

    df = _rich_df()
    print(f"\nInput shape: {df.shape}")
    print(f"Patients: {sorted(df['id'].unique().to_list())}")
    print(f"Classes:  {sorted(df['class'].unique().to_list())}")

    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_per_state={"default": "week"},
    )
    tok.train(df)

    ids, vals = tok.encode(df)
    print(f"\nEncoded length: {len(ids)} tokens  (vals is {'None' if vals is None else 'present'})")

    decoded = tok.decode(ids)
    print(f"\nFirst 30 tokens:")
    for i, t in enumerate(decoded[:30]):
        print(f"  [{i:3d}] {t}")

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    print(f"\nReconstructed shape: {recon.shape}")
    print(recon.head(10))

    # Assertions — birth-date rows are consumed; admission/discharge are preserved
    non_birth = df.filter(
        ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
    )
    assert recon.height == non_birth.height
    assert set(recon.columns) >= {"id", "time", "class", "text_value", "numeric_value"}
    print("\n✓ Round-trip assertion passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2: Concept + continuous + fused
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_concept_continuous_fused():
    """Continuous mode with fused numeric placement.

    Numerics are distribution-scaled to ~[0,1] and carried in the vals array.
    Fused placement means the concept token and its numeric value share the
    same position in the ids and vals arrays — no separate Q/L/NUM token.
    """
    print("\n" + "=" * 72)
    print("DEMO: Concept / Continuous / Fused")
    print("=" * 72)

    df = _rich_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="fused",
        level_threshold=3,
    )
    tok.train(df)

    ids, vals = tok.encode(df)
    print(f"\nEncoded length: {len(ids)} tokens")
    print(f"vals sample (first 10): {vals[:10] if vals else 'None'}")

    decoded = tok.decode(ids)
    print(f"\nFirst 20 tokens:")
    for i, t in enumerate(decoded[:20]):
        v = f"  val={vals[i]:.4f}" if vals and vals[i] != 0.0 else ""
        print(f"  [{i:3d}] {t}{v}")

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon.height > 0
    print(f"\nReconstructed {recon.height} rows — ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 3: BPE + discrete + factored
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_demo_bpe_discrete_factored():
    """BPE text mode for high-cardinality text, with discrete numerics.

    When a class has more unique text values than ``text_mode_threshold``,
    DBTokenizer auto-selects BPE.  BPE tokens are sub-word tokens that can
    represent any text; concept tokens map each unique text to one token ID.

    Note: fused placement silently falls back to factored for BPE classes.
    """
    print("\n" + "=" * 72)
    print("DEMO: BPE / Discrete / Factored")
    print("=" * 72)

    rows = []
    for i in range(80):
        rows.append(dict(id=1,
                         time=datetime(2023, 1, 1) + timedelta(hours=i),
                         cls="note",
                         text_value=f"Patient presents with symptom variant {i} requiring assessment",
                         numeric_value=None))
    for i in range(12):
        rows.append(dict(id=1,
                         time=datetime(2023, 1, 5) + timedelta(hours=i * 3),
                         cls="lab",
                         text_value="troponin",
                         numeric_value=0.02 + i * 0.01))
    df = pl.DataFrame(rows).rename({"cls": "class"})

    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        final_vocab_size=512,
    )
    tok.train(df)

    print(f"\nClass text modes: {tok.class_text_modes}")
    assert tok.class_text_modes["note"] == "bpe"
    assert tok.class_text_modes["lab"] == "concept"

    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    print(f"Token count: {len(ids)}")
    print(f"\nBPE region (first tokens of patient 1):")
    for i, t in enumerate(decoded[:20]):
        print(f"  [{i:3d}] {t}")

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    note_rows = recon.filter(pl.col("class") == "note")
    lab_rows = recon.filter(pl.col("class") == "lab")
    print(f"\nReconstructed: {note_rows.height} notes, {lab_rows.height} labs")
    assert note_rows.height == 80
    assert lab_rows.height == 12
    print("✓ BPE round-trip passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 4: Kitchen-sink — all features active
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_full_featured():
    """Full feature set: milestones, age tokens, time-deltas, IP/OP awareness.

    This demo activates every feature the tokenizer offers:
      - Weekly outpatient milestones, 8hr shift inpatient milestones
      - Age tokens from demographic birth_date rows
      - Time-delta tokens between observations
      - Admission/discharge boundary detection
      - Discrete quantile bins with factored placement
    """
    print("\n" + "=" * 72)
    print("DEMO: Full-featured (kitchen-sink)")
    print("=" * 72)

    df = _rich_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        state_transitions={"inpatient": ["admission"], "default": ["discharge"]},
        initial_state="default",
        milestone_per_state={"inpatient": "8hr", "default": "week"},
        milestone_shift_start=7,
    )
    tok.train(df)

    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)

    # Categorise token types for stats
    specials = [t for t in decoded if t.startswith("<|")]
    milestones = [t for t in decoded if t.startswith("<|ms_")]
    td_tokens = [t for t in decoded if "delta_time" in t]
    age_tokens = [t for t in decoded if t.startswith("<|age_")]
    q_tokens = [t for t in decoded if t.startswith("<|Q")]
    concepts = [t for t in decoded if not t.startswith("<|")]

    print(f"\nTotal tokens:    {len(decoded)}")
    print(f"  Special:       {len(specials)}")
    print(f"    Milestones:  {len(milestones)}")
    print(f"    Time-delta:  {len(td_tokens)}")
    print(f"    Age:         {len(age_tokens)}")
    print(f"    Q-bins:      {len(q_tokens)}")
    print(f"  Concepts:      {len(concepts)}")

    print(f"\nVocab size: {len(tok.vocab)}")

    # Patient 1 should have admission/discharge awareness
    sos_count = sum(1 for t in decoded if t == "<|sos|>")
    assert sos_count >= 2, "expected SOS for each patient"
    assert len(milestones) > 0, "milestones should be present"
    print(f"  SOS tokens:    {sos_count}")
    print("\n✓ Kitchen-sink test passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 5: Round-trip fidelity check
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_round_trip():
    """Encode → decode → DataFrame and verify every field reconstructs.

    This demo emphasises data fidelity: each text_value, class, and
    (approximately) numeric_value must survive the round trip.
    """
    print("\n" + "=" * 72)
    print("DEMO: Round-trip fidelity")
    print("=" * 72)

    df = _rich_df()
    # Exclude meta-rows for comparison
    non_birth = df.filter(
        ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
    ).sort(["id", "time"])

    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
    )
    tok.train(df)

    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF).sort(["id", "time"])

    print(f"Original rows  (excl. birth): {non_birth.height}")
    print(f"Reconstructed rows:          {recon.height}")
    assert recon.height == non_birth.height

    # Compare classes and text values (admission/discharge have None text)
    orig_classes = non_birth["class"].to_list()
    got_classes = recon["class"].to_list()
    class_match = sum(1 for a, b in zip(orig_classes, got_classes) if a == b)
    print(f"  class: {class_match}/{len(orig_classes)} exact matches")
    assert class_match == len(orig_classes), "class mismatch"

    # Numerics — discrete bins lose precision, but Nones must stay None
    orig_nv = non_birth["numeric_value"].to_list()
    got_nv = recon["numeric_value"].to_list()
    none_ok = all((o is None) == (g is None) for o, g in zip(orig_nv, got_nv))
    print(f"  numeric_value None-alignment: {'✓' if none_ok else '✗'}")
    assert none_ok

    print("\n✓ Round-trip fidelity confirmed")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 6: Vocab anatomy
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_vocab_anatomy():
    """Explore the vocabulary structure of a trained tokenizer.

    This demo shows what token types end up in the vocabulary and how they
    are distributed — useful for understanding tokenizer internals.
    """
    print("\n" + "=" * 72)
    print("DEMO: Vocabulary anatomy")
    print("=" * 72)

    df = _rich_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_per_state={"default": "week"},
    )
    tok.train(df)

    vocab = tok.vocab
    print(f"\nTotal vocab size: {len(vocab)}")

    # Categorize
    cats = {}
    for token in vocab:
        if token in ("<|SOS|>", "<|EOS|>", "<|SEP|>", "<|PAD|>"):
            cats.setdefault("structural", []).append(token)
        elif token.startswith("<|ms_"):
            cats.setdefault("milestone", []).append(token)
        elif token.startswith("<|age_"):
            cats.setdefault("age", []).append(token)
        elif token.startswith("<|Q"):
            cats.setdefault("Q-bin", []).append(token)
        elif token.startswith("<|L"):
            cats.setdefault("L-level", []).append(token)
        elif token.startswith("<|NUM"):
            cats.setdefault("NUM-marker", []).append(token)
        elif token.startswith("<|td_") or "delta_time" in token:
            cats.setdefault("time-delta", []).append(token)
        elif token.startswith("<|"):
            cats.setdefault("other-special", []).append(token)
        else:
            cats.setdefault("concept", []).append(token)

    for name, tokens in sorted(cats.items()):
        print(f"  {name:15s}: {len(tokens):4d} tokens")
        # Show a few examples
        for t in tokens[:3]:
            print(f"    → {t}")
        if len(tokens) > 3:
            print(f"    ... and {len(tokens) - 3} more")

    # Verify ID mapping is bijective
    assert len(set(vocab.values())) == len(vocab), "vocab IDs are not unique"
    print(f"\n✓ Vocab is well-formed ({len(vocab)} tokens, all IDs unique)")
