"""Tests for save / load round-trip serialization."""
import math
import os
import tempfile
import pytest
from conftest import df_simple, df_bpe, REF, HAS_BPE
from dbtoken import DBTokenizer


def _assert_vals_equal(vals1, vals2):
    """Compare vals lists allowing NaN == NaN."""
    assert len(vals1) == len(vals2)
    for v1, v2 in zip(vals1, vals2):
        if math.isnan(v1):
            assert math.isnan(v2)
        else:
            assert v1 == pytest.approx(v2)


# ═══════════════════════════════════════════════════════════════════════════════
# Concept serialization
# ═══════════════════════════════════════════════════════════════════════════════

def test_concept_discrete_factored_roundtrip(tmp_path):
    df = df_simple()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3,
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    save_path = str(tmp_path / "tok_concept")
    tok.save(save_path)
    assert (tmp_path / "tok_concept.json").exists()

    tok2 = DBTokenizer.load(save_path)
    ids2, vals2 = tok2.encode(df)
    assert ids1 == ids2
    assert vals1 == vals2
    assert tok2.vocab == tok.vocab


def test_concept_continuous_roundtrip(tmp_path):
    df = df_simple()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="continuous", num_seq="factored",
        level_threshold=3,
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    save_path = str(tmp_path / "tok_cont")
    tok.save(save_path)
    tok2 = DBTokenizer.load(save_path)
    ids2, vals2 = tok2.encode(df)
    assert ids1 == ids2
    _assert_vals_equal(vals1, vals2)


def test_pair_override_roundtrip(tmp_path):
    df = df_simple()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3,
        text_mode_overrides={("lab", "glucose"): "concept"},
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    save_path = str(tmp_path / "tok_pair")
    tok.save(save_path)
    tok2 = DBTokenizer.load(save_path)
    ids2, vals2 = tok2.encode(df)
    assert ids1 == ids2
    assert ("lab", "glucose") in tok2.pair_text_modes or \
           ("lab", "glucose") in tok2.text_mode_overrides


def test_milestone_config_roundtrip(tmp_path):
    df = df_simple()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3,
        milestone_per_state={"default": "week"},
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    save_path = str(tmp_path / "tok_ms")
    tok.save(save_path)
    tok2 = DBTokenizer.load(save_path)
    assert tok2.milestone_per_state == {"default": "week"}
    ids2, vals2 = tok2.encode(df)
    assert ids1 == ids2


def test_decode_after_load(tmp_path):
    df = df_simple()
    tok = DBTokenizer(
        text_mode_default="concept", num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3,
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    recon1 = tok.decode_to_dataframe(ids, vals, reference_time=REF)

    tok.save(str(tmp_path / "tok_dec"))
    tok2 = DBTokenizer.load(str(tmp_path / "tok_dec"))
    recon2 = tok2.decode_to_dataframe(ids, vals, reference_time=REF)

    assert recon1.columns == recon2.columns
    assert recon1.height == recon2.height
    assert recon1["class"].to_list() == recon2["class"].to_list()
    assert recon1["text_value"].to_list() == recon2["text_value"].to_list()


# ═══════════════════════════════════════════════════════════════════════════════
# BPE serialization
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_roundtrip(tmp_path):
    df = df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto", text_mode_threshold=64,
        num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3, final_vocab_size=512,
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    save_path = str(tmp_path / "tok_bpe")
    tok.save(save_path)
    assert (tmp_path / "tok_bpe.json").exists()
    assert (tmp_path / "tok_bpe.bpe").exists()

    tok2 = DBTokenizer.load(save_path)
    ids2, vals2 = tok2.encode(df)
    assert ids1 == ids2
    assert tok2.has_bpe_classes is True


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_decode_after_load(tmp_path):
    df = df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto", text_mode_threshold=64,
        num_type="discrete", num_seq="factored",
        n_bins=5, level_threshold=3, final_vocab_size=512,
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded1 = tok.decode(ids)

    tok.save(str(tmp_path / "tok_bpe_dec"))
    tok2 = DBTokenizer.load(str(tmp_path / "tok_bpe_dec"))
    decoded2 = tok2.decode(ids)
    assert decoded1 == decoded2
