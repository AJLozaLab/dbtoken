"""Tests for decode_to_dataframe: reversing tokenization back to structured data."""
import math
import polars as pl
import pytest
from datetime import datetime, timedelta
from conftest import (
    df_simple, df_level, df_no_numeric, df_bpe,
    REF, HAS_BPE,
    assert_schema, assert_row_count, assert_classes_match,
    assert_text_values_match, assert_numerics_finite,
    assert_level_exact, assert_time_increases, assert_patient_ids,
)
from tokenizer import DBTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Core reconstruction matrix
# ═══════════════════════════════════════════════════════════════════════════════

def test_discrete_factored_concept_bins():
    df = _s()
    tok = _tok_bins()
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is None
    decoded = tok.decode(ids)
    assert any(t.startswith("<|Q") for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=0)
    _check_all(recon, df, start=0)


def test_discrete_fused_concept_bins():
    df = _s()
    tok = _tok_bins(num_seq="fused")
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    assert any("::Q" in t for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)


def test_discrete_factored_concept_level():
    df = df_level()
    tok = _tok_level()
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    assert any(t.startswith("<|L") for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)
    assert_level_exact(recon, df, [1.0, 2.0, 3.0])


def test_discrete_fused_concept_level():
    df = df_level()
    tok = _tok_level(num_seq="fused")
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    assert any("::L" in t for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)
    assert_level_exact(recon, df, [1.0, 2.0, 3.0])


def test_continuous_factored_concept_scaling():
    df = _s()
    tok = _tok_continuous()
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None
    assert len(ids) == len(vals)
    decoded = tok.decode(ids)
    assert "<|NUM|>" in decoded
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)
    _check_approx_numerics(recon, df)


def test_continuous_fused_concept_scaling():
    df = _s()
    tok = _tok_continuous(num_seq="fused")
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None
    decoded = tok.decode(ids)
    assert "glucose" in decoded
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)


def test_concept_no_numeric():
    df = df_no_numeric()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored"
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    _check_core(recon, df)
    for nv in recon["numeric_value"].to_list():
        assert nv is None


# ═══════════════════════════════════════════════════════════════════════════════
# Time-delta reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_delta_factored_discrete():
    df = _s()
    tok = _tok_bins()
    tok.train(df)
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)
    p0 = recon.filter(pl.col("id") == 0).sort("time")["time"].to_list()
    assert all(t >= REF for t in p0)


def test_time_delta_fused_discrete():
    df = _s()
    tok = _tok_bins(num_seq="fused")
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    assert any("delta_time" in t and "_Q" in t for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


def test_time_delta_factored_continuous():
    df = _s()
    tok = _tok_continuous()
    tok.train(df)
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


def test_time_delta_fused_continuous():
    df = _s()
    tok = _tok_continuous(num_seq="fused")
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    td_markers = [t for t in decoded if "delta_time" in t and "::Q" not in t and "_Q" not in t]
    assert len(td_markers) > 0
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_multi_patient_ids():
    df = _s()
    tok = _tok_bins()
    tok.train(df)
    ids, vals = tok.encode(df)

    recon0 = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=0)
    assert_patient_ids(recon0, start=0)
    assert sorted(recon0["id"].unique().to_list()) == [0, 1]

    recon100 = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=100)
    assert_patient_ids(recon100, start=100)
    assert sorted(recon100["id"].unique().to_list()) == [100, 101]


def test_empty_stream():
    df = _s()
    tok = _tok_bins()
    tok.train(df)
    recon = tok.decode_to_dataframe([], None, reference_time=REF)
    assert recon.height == 0
    assert set(recon.columns) >= {"id", "time", "class", "text_value", "numeric_value"}


def test_auto_concept_mode():
    df = df_no_numeric()
    tok = DBTokenizer(
        text_mode_default="auto", text_mode_threshold=64,
        num_type="discrete", num_seq="factored"
    )
    tok.train(df)
    assert tok.class_text_modes.get("dx") == "concept"
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_text_values_match(recon, df)


def test_milestones_dropped():
    df = _s()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3, milestone_per_state={"default": "week"},
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    assert any(t.startswith("<|ms_") for t in decoded)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_row_count(recon, df)


# ═══════════════════════════════════════════════════════════════════════════════
# BPE decode tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_factored_discrete():
    df = df_bpe()
    tok = _tok_bpe()
    tok.train(df)
    assert tok.class_text_modes["note"] == "bpe"
    assert tok.class_text_modes["lab"] == "concept"
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    lab_orig = df.filter(pl.col("class") == "lab").sort("time")["text_value"].to_list()
    lab_recon = recon.filter(pl.col("class") == "lab").sort("time")["text_value"].to_list()
    assert lab_orig == lab_recon
    note_recon = recon.filter(pl.col("class") == "note")["text_value"].to_list()
    assert all(tv is not None and len(tv) > 0 for tv in note_recon)
    assert_time_increases(recon)
    assert_numerics_finite(recon, df)


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_factored_continuous():
    df = df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto", text_mode_threshold=64,
        num_type="continuous", num_seq="factored",
        level_threshold=3, final_vocab_size=512
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_numerics_finite(recon, df)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _s():
    return df_simple()


def _tok_bins(num_seq="factored"):
    return DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq=num_seq,
        n_bins=5, level_threshold=3
    )


def _tok_level(num_seq="factored"):
    return DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq=num_seq,
        n_bins=5, level_threshold=10
    )


def _tok_continuous(num_seq="factored"):
    return DBTokenizer(
        text_mode_default="concept", num_type="continuous", num_seq=num_seq,
        level_threshold=3
    )


def _tok_bpe():
    return DBTokenizer(
        text_mode_default="auto", text_mode_threshold=64,
        num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3, final_vocab_size=512
    )


def _check_all(recon, df, start=0):
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_numerics_finite(recon, df)
    assert_time_increases(recon)
    assert_patient_ids(recon, start=start)


def _check_core(recon, df):
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_time_increases(recon)


def _check_approx_numerics(recon, df):
    orig_nv = df.sort(["id", "time"])["numeric_value"].to_list()
    got_nv = recon.sort(["id", "time"])["numeric_value"].to_list()
    for o, g in zip(orig_nv, got_nv):
        if o is None:
            continue
        assert g is not None
        assert abs(g - o) / max(abs(o), 1.0) < 5.0, (
            f"continuous round-trip too far off: orig={o}, got={g}"
        )
