"""Tests for auto BPE/concept resolution, auto level, pair-level overrides, fused→factored BPE fallback."""
import polars as pl
import warnings
import pytest
from datetime import datetime, timedelta
from conftest import make_df, make_bpe_df, HAS_BPE
from tokenizer import DBTokenizer


class TestAutoTextModeResolution:

    def test_auto_below_threshold_is_concept(self):
        """With text_mode='auto' and few unique values, class resolves to concept."""
        rows = []
        for pid in [1, 2]:
            for i in range(10):
                rows.append({
                    "id": pid,
                    "time": datetime(2020, 1, 1) + timedelta(hours=i),
                    "class": "dx",
                    "text_value": f"code_{i % 5}",  # 5 unique → concept
                    "numeric_value": None,
                })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.class_text_modes["dx"] == "concept"

    @pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
    def test_auto_above_threshold_is_bpe(self):
        """With text_mode='auto' and many unique values, class resolves to bpe."""
        rows = []
        for i in range(100):
            rows.append({
                "id": 1,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "note",
                "text_value": f"note text variant {i}",  # 100 unique → BPE
                "numeric_value": None,
            })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            final_vocab_size=512,
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.class_text_modes["note"] == "bpe"

    @pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
    def test_auto_at_threshold_boundary(self):
        """With exactly threshold unique values, should resolve to concept (not >)."""
        rows = []
        for i in range(64):
            rows.append({
                "id": 1,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "note",
                "text_value": f"label_{i}",  # exactly 64 unique
                "numeric_value": None,
            })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            final_vocab_size=512,
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        # threshold=64, n_unique=64 → NOT > 64 → concept
        assert tok.class_text_modes["note"] == "concept"

    @pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
    def test_auto_one_above_threshold(self):
        """With threshold+1 unique values, should resolve to BPE."""
        rows = []
        for i in range(65):
            rows.append({
                "id": 1,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "note",
                "text_value": f"label_{i}",  # 65 unique > 64 threshold
                "numeric_value": None,
            })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            final_vocab_size=512,
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.class_text_modes["note"] == "bpe"


class TestAutoLevel:

    def test_level_threshold_triggers_level_tokens(self):
        """n_distinct <= level_threshold → level tokens, not bins."""
        rows = []
        for pid in [1, 2]:
            for i in range(20):
                rows.append({
                    "id": pid,
                    "time": datetime(2020, 1, 1) + timedelta(hours=i),
                    "class": "lab",
                    "text_value": "platelet",
                    "numeric_value": float(i % 3),  # 3 distinct values
                })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=10,  # 3 < 10 → level
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.numeric_params[("lab", "platelet")]["type"] == "level"

    def test_above_level_threshold_uses_bins(self):
        """n_distinct > level_threshold → bins."""
        rows = []
        for pid in [1, 2]:
            for i in range(20):
                rows.append({
                    "id": pid,
                    "time": datetime(2020, 1, 1) + timedelta(hours=i),
                    "class": "lab",
                    "text_value": "platelet",
                    "numeric_value": float(i),  # 20 distinct values
                })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=10,  # 20 > 10 → bins
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.numeric_params[("lab", "platelet")]["type"] == "bins"


class TestPairOverrides:

    def test_pair_level_override(self):
        """Tuple-key (class, text_value) override controls individual pair's text mode."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            text_mode_overrides={
                ("lab", "hemoglobin"): "concept",
            },
        )
        tok.train(df)

        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        hb_toks = [t for t in decoded if "hemoglobin" in t]
        assert len(hb_toks) > 0, "hemoglobin concept tokens missing"

    @pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
    def test_pair_override_concept_in_bpe_class(self):
        """Force a specific text_value to concept in a class that auto-detects as BPE."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=512,
            text_mode_overrides={
                ("notes", "Patient presented with symptom variant 0"): "concept",
            },
        )
        tok.train(df)

        assert tok.class_text_modes["notes"] == "bpe"
        assert tok.pair_text_modes[("notes", "Patient presented with symptom variant 0")] == "concept"

        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        assert "Patient presented with symptom variant 0" in decoded

    def test_class_level_override(self):
        """String-key override controls entire class text mode."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            text_mode_overrides={"lab": "concept"},
            milestone_op="none",
            milestone_ip="none",
        )
        tok.train(df)
        assert tok.class_text_modes["lab"] == "concept"


class TestFusedBPEFallback:

    @pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
    def test_fused_bpe_warning(self):
        """Requesting fused with BPE classes should emit a warning."""
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

    def test_fused_concept_only_no_warning(self):
        """Fused with only concept classes should NOT emit a fallback warning."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            milestone_op="none",
            milestone_ip="none",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tok.train(df)
            fused_warnings = [x for x in w if "factored" in str(x.message)]
            assert len(fused_warnings) == 0, "No fused+BPE warning expected with concept-only"
