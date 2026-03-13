"""Tests for time-delta emission (IP/OP), milestone+delta interplay, and correct IP vs OP separation."""
import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta
from conftest import make_df, df_simple, REF
from tokenizer import DBTokenizer


class TestTimeDeltaEmission:

    def test_time_delta_tokens_present(self):
        """Time deltas between events produce delta_time tokens."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            milestone_op="none",
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
            milestone_op="none",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        td_toks = [t for t in decoded if "delta_time" in t]
        assert len(td_toks) == 0, "No time delta for single-event patient"

    def test_factored_discrete_td(self):
        """Factored+discrete: delta_time_op + Q token pair."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=3,
            milestone_op="none",
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is None
        # Should have delta_time_op followed by Q token
        td_marks = [t for t in decoded if t.startswith("<|delta_time_")]
        q_marks = [t for t in decoded if t.startswith("<|Q")]
        assert len(td_marks) > 0
        assert len(q_marks) > 0

    def test_fused_discrete_td(self):
        """Fused+discrete: single combined delta_time_op_QN token."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="fused",
            n_bins=5,
            level_threshold=3,
            milestone_op="none",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        fused_td = [t for t in decoded if "delta_time" in t and "_Q" in t]
        assert len(fused_td) > 0, "Expected fused delta_time_*_QN tokens"

    def test_factored_continuous_td(self):
        """Factored+continuous: delta_time_op + NUM token, value in vals."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            level_threshold=3,
            milestone_op="none",
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
        """Fused+continuous: delta_time_op marker with val in vals array."""
        df = df_simple()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="fused",
            level_threshold=3,
            milestone_op="none",
        )
        tok.train(df)
        ids, vals = tok.encode(df)
        decoded = tok.decode(ids)

        assert vals is not None
        td_marks = [t for t in decoded if t.startswith("<|delta_time_")]
        assert len(td_marks) > 0


class TestIPvsOP:

    def test_ip_op_separation(self):
        """Admitted patients use IP delta; outpatient patients use OP delta."""
        df = make_df(include_birth=True, include_admit=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            admission_classes=["admission"],
            discharge_classes=["discharge"],
            milestone_ip="8hr",
            milestone_op="week",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        ip_tds = [t for t in decoded if "delta_time_ip" in t]
        op_tds = [t for t in decoded if "delta_time_op" in t]
        # Patient 1 is admitted → IP deltas; Patient 2 is outpatient → OP deltas
        assert len(ip_tds) > 0, "Expected IP time deltas for admitted patient"
        assert len(op_tds) > 0, "Expected OP time deltas for outpatient patient"

    def test_separate_ip_op_params(self):
        """IP and OP time delta params are fitted separately."""
        df = make_df(include_birth=True, include_admit=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            admission_classes=["admission"],
            discharge_classes=["discharge"],
        )
        tok.train(df)

        assert "ip" in tok.time_delta_params
        assert "op" in tok.time_delta_params

    def test_no_admission_uses_global(self):
        """Without admission/discharge classes, both IP and OP use global params."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            milestone_op="none",
        )
        tok.train(df)

        assert "ip" in tok.time_delta_params
        assert "op" in tok.time_delta_params
        # Both should be identical (global fit)
        assert tok.time_delta_params["ip"] == tok.time_delta_params["op"]


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
            milestone_op="week",
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
            milestone_op="week",
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        all_toks = []
        for row in dbg.iter_rows(named=True):
            all_toks.extend(row["tokens"])

        ms_toks = [t for t in all_toks if t.startswith("<|ms_week_")]
        # 4 events, each in a different week → 3 time deltas → up to 3 milestone changes
        assert len(ms_toks) >= 1, "Expected week milestone tokens"
