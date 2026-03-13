"""Tests for tokenization mode combinations: BPE vs concept × discrete vs continuous × factored vs fused."""
import polars as pl
import numpy as np
import warnings
import pytest
from conftest import make_df, make_bpe_df, HAS_BPE, REF
from tokenizer import DBTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Concept-mode tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptDiscrete:

    def test_factored(self):
        """concept + discrete + factored: basic encoding produces token IDs, no vals."""
        df = make_df(include_birth=True)
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
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert len(ids) > len(df), "encoding should expand row count"
        assert vals is None, "discrete mode should produce vals=None"
        assert "<|sos|>" in decoded
        assert "<|eos|>" in decoded
        # Should have class tokens
        assert any(t == "<|lab|>" for t in decoded)

    def test_fused(self):
        """concept + discrete + fused: produces fused tokens like tv::L0, tv::Q3."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            level_threshold=10,
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        dbg = tok.encode_debug(df)

        assert vals is None
        # Check that fused tokens appear in debug output
        all_toks = []
        for row in dbg.iter_rows(named=True):
            all_toks.extend(row["tokens"])
        fused = [t for t in all_toks if "::" in t]
        assert len(fused) > 0, "expected fused concept+numeric tokens"


class TestConceptContinuous:

    def test_factored(self):
        """concept + continuous + factored: produces ids+vals, NUM tokens present."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            n_bins=5,
            level_threshold=10,
        )
        tok.train(df)
        ids, vals = tok.encode(df)

        assert vals is not None, "continuous mode must return vals"
        assert len(ids) == len(vals), "ids and vals length mismatch"
        decoded = tok.decode(ids)
        num_toks = [t for t in decoded if t == "<|NUM|>"]
        assert len(num_toks) > 0, "expected <|NUM|> tokens"
        # Some vals should be actual floats (not NaN)
        real_vals = [v for v in vals if not np.isnan(v)]
        assert len(real_vals) > 0, "expected some non-NaN vals"

    def test_fused(self):
        """concept + continuous + fused: concept tokens carry scaled vals, no ::L or ::Q."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="fused",
            n_bins=5,
            level_threshold=3,
        )
        tok.train(df)
        ids, vals = tok.encode(df)

        assert vals is not None
        decoded = tok.decode(ids)
        # Fused continuous: concept token is plain, value stored in vals
        assert "hemoglobin" in decoded or "glucose" in decoded


# ═══════════════════════════════════════════════════════════════════════════════
# BPE-mode tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
class TestBPEDiscrete:

    def test_factored(self):
        """BPE + discrete + factored: high-cardinality text auto-detects as BPE."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=512,
        )
        tok.train(df)

        assert tok.class_text_modes["notes"] == "bpe"
        assert tok.class_text_modes["lab"] == "concept"

        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is None
        assert "hemoglobin" in decoded, "concept lab tokens should appear literally"
        assert len(ids) > len(df) * 2, "BPE should expand long text into multiple tokens"

    def test_fused_falls_back_to_factored(self):
        """BPE + discrete + fused: BPE classes silently fall back to factored."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            final_vocab_size=512,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tok.train(df)
            fused_warnings = [x for x in w if "factored" in str(x.message)]
            assert len(fused_warnings) > 0, "Expected fused+BPE fallback warning"

        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        # Lab tokens (concept mode) should use fused
        fused_lab = [t for t in decoded if "::L" in t]
        assert len(fused_lab) > 0, "Expected fused lab tokens like 'hemoglobin::L3'"


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
class TestBPEContinuous:

    def test_factored(self):
        """BPE + continuous + factored: vals array includes numeric values."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="continuous",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=512,
        )
        tok.train(df)
        ids, vals = tok.encode(df)

        assert vals is not None
        assert len(ids) == len(vals)
        real_vals = [v for v in vals if not np.isnan(v)]
        assert len(real_vals) > 0, "expected some non-NaN vals for lab numerics"

    def test_fused_falls_back_to_factored(self):
        """BPE + continuous + fused: BPE classes fall back to factored, concept stays fused."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="continuous",
            num_seq="fused",
            n_bins=5,
            final_vocab_size=512,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tok.train(df)
            fused_warnings = [x for x in w if "factored" in str(x.message)]
            assert len(fused_warnings) > 0, "Expected fused+BPE fallback warning"

        ids, vals = tok.encode(df)
        assert vals is not None
        decoded = tok.decode(ids)

        # Concept lab tokens should appear plain (fused continuous = plain token + val)
        assert any("creatinine" in t or "hemoglobin" in t or "glucose" in t
                    for t in decoded)
