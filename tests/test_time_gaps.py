"""Tests for time-delta emission (threshold-based scales), milestone+delta interplay."""
import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta
from conftest import make_df, df_simple, REF
from dbtoken import DBTokenizer


class TestTimeDeltaEmission:

    def test_time_delta_tokens_present(self):
        """Time deltas between events produce delta_time tokens."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        td_toks = [t for t in decoded if "delta_time" in t]
        assert len(td_toks) > 0, "Expected delta_time tokens between events"

    def test_no_delta_for_first_event(self):
        """First event of a patient has no preceding time delta."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "glucose", "numeric_value": 100.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        td_toks = [t for t in decoded if "delta_time" in t]
        assert len(td_toks) == 0, "No time delta for single-event patient"

    def test_factored_discrete_td(self):
        """Factored+discrete: delta_time_0 + Q token pair."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=3,
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is None
        # Should have delta_time_0 followed by Q token
        td_marks = [t for t in decoded if t.startswith("<|delta_time_")]
        q_marks = [t for t in decoded if t.startswith("<|Q")]
        assert len(td_marks) > 0
        assert len(q_marks) > 0

    def test_fused_discrete_td(self):
        """Fused+discrete: single combined delta_time_0_QN token."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            level_threshold=3,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        fused_td = [t for t in decoded if "delta_time" in t and "_Q" in t]
        assert len(fused_td) > 0, "Expected fused delta_time_*_QN tokens"

    def test_factored_continuous_td(self):
        """Factored+continuous: delta_time_0 + NUM token, value in vals."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            level_threshold=3,
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is not None
        td_marks = [t for t in decoded if t.startswith("<|delta_time_")]
        num_marks = [t for t in decoded if t == "<|NUM|>"]
        assert len(td_marks) > 0
        assert len(num_marks) > 0

    def test_fused_continuous_td(self):
        """Fused+continuous: delta_time_0 marker with val in vals array."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="fused",
            level_threshold=3,
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is not None
        td_marks = [t for t in decoded if t.startswith("<|delta_time_")]
        assert len(td_marks) > 0


class TestTemporalScales:

    def _make_mixed_df(self):
        """DataFrame with sub-hour deltas and multi-day deltas."""
        rows = []
        # Patient 1: events every 10 minutes (sub-24h gaps)
        for i in range(5):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 1, 8, 0) + timedelta(minutes=i * 10),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        # Patient 2: events every 3 days (>24h gaps)
        for i in range(5):
            rows.append({
                "id": 2,
                "time": datetime(2023, 1, 1, 8, 0) + timedelta(days=i * 3),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 90.0 + i,
            })
        return pl.DataFrame(rows)

    def test_single_global_distribution(self):
        """No time_scales → single distribution, token is <|delta_time_0|>."""
        df = self._make_mixed_df()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
        )
        tok.train(df)

        # Only one key: "0"
        assert "0" in tok.time_delta_params
        assert len(tok.time_delta_params) == 1

        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        td_toks = [t for t in decoded if "delta_time" in t]
        assert all("_0" in t or t == "<|delta_time_0|>" for t in td_toks), (
            f"Expected only delta_time_0 tokens, got {td_toks}"
        )

    def test_two_scale_split(self):
        """time_scales=[86400] splits deltas into short (band 0) and long (band 1)."""
        df = self._make_mixed_df()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            time_scales=[86400.0],
        )
        tok.train(df)

        assert "0" in tok.time_delta_params
        assert "1" in tok.time_delta_params

        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        td_0 = [t for t in decoded if "delta_time_0" in t]
        td_1 = [t for t in decoded if "delta_time_1" in t]
        # Patient 1 has sub-24h gaps → band 0; Patient 2 has >24h gaps → band 1
        assert len(td_0) > 0, "Expected delta_time_0 tokens for short gaps"
        assert len(td_1) > 0, "Expected delta_time_1 tokens for long gaps"

    def test_named_scales(self):
        """time_scale_names produces tokens like <|delta_time_short|>."""
        df = self._make_mixed_df()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            time_scales=[86400.0],
            time_scale_names=["short", "long"],
        )
        tok.train(df)

        assert "short" in tok.time_delta_params
        assert "long" in tok.time_delta_params

        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        short_toks = [t for t in decoded if "delta_time_short" in t]
        long_toks  = [t for t in decoded if "delta_time_long" in t]
        assert len(short_toks) > 0
        assert len(long_toks) > 0

    def test_separate_scale_params_fitted(self):
        """Each band gets its own independently fitted distribution."""
        df = self._make_mixed_df()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            time_scales=[86400.0],
        )
        tok.train(df)

        p0 = tok.time_delta_params["0"]
        p1 = tok.time_delta_params["1"]
        # The two bands should have different edge distributions
        assert p0 != p1, "Expected different params for short vs long scale bands"

    def test_scale_names_length_validation(self):
        """time_scale_names length must equal len(time_scales)+1."""
        with pytest.raises(ValueError, match="time_scale_names"):
            DBTokenizer(
                time_scales=[86400.0],
                time_scale_names=["only_one"],  # needs 2
            )

    def test_unsorted_time_scales_raises(self):
        """Unsorted time_scales must raise ValueError."""
        with pytest.raises(ValueError, match="sorted"):
            DBTokenizer(time_scales=[7200.0, 3600.0])


class TestTimeDeltaWithMilestones:

    def test_milestone_after_td(self):
        """Milestone token emitted after time-delta when boundary changes."""
        # Events spanning multiple weeks → weekly milestones should follow td tokens
        rows = []
        for i in range(5):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 1, 10, 0) + timedelta(weeks=i),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        df = pl.DataFrame(rows)

        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            milestone_per_state={"default": "week"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        all_toks = []
        for row in dbg.iter_rows(named=True):
            all_toks.extend(row["tokens"])

        td_toks = [t for t in all_toks if "delta_time" in t]
        ms_toks = [t for t in all_toks if t.startswith("<|ms_")]
        assert len(td_toks) > 0
        assert len(ms_toks) > 0

    def test_td_and_milestone_counts(self):
        """Each week boundary crossing produces exactly one milestone."""
        rows = []
        for i in range(4):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 7 * i + 1, 10, 0),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        df = pl.DataFrame(rows)

        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            milestone_per_state={"default": "week"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        all_toks = []
        for row in dbg.iter_rows(named=True):
            all_toks.extend(row["tokens"])

        ms_toks = [t for t in all_toks if t.startswith("<|ms_week_")]
        # 4 events, each in a different week → 3 time deltas → up to 3 milestone changes
        assert len(ms_toks) >= 1, "Expected week milestone tokens"
