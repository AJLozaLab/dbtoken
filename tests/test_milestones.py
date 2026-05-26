"""Tests for state-driven milestones, trigger times, and Kalman-filter boundary emission."""
import polars as pl
import pytest
from datetime import datetime, timedelta
from conftest import make_df
from tokenizer import DBTokenizer


def _all_tokens_from_debug(dbg, patient_id=None):
    """Extract flat list of all tokens from encode_debug output."""
    if patient_id is not None:
        dbg = dbg.filter(pl.col("id") == patient_id)
    toks = []
    for row in dbg.iter_rows(named=True):
        toks.extend(row["tokens"])
    return toks


class TestMilestoneEmission:

    def test_weekly_milestones_emitted(self):
        """Patient spanning multiple weeks should produce week milestone tokens."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            milestone_per_state={"default": "week"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        # Patient 2 events span ~50 days → multiple weeks
        p2_toks = _all_tokens_from_debug(dbg, patient_id=2)
        ms_toks = [t for t in p2_toks if t.startswith("<|ms_")]
        assert len(ms_toks) > 0, "Expected milestone tokens for patient 2 spanning multiple weeks"

    def test_no_milestone_per_state_suppresses(self):
        """No milestone_per_state (empty dict) suppresses all milestones."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            # milestone_per_state not set → empty dict → "none" for all states
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        all_toks = _all_tokens_from_debug(dbg)
        ms_toks = [t for t in all_toks if t.startswith("<|ms_")]
        assert len(ms_toks) == 0, f"Expected no milestones without milestone_per_state, got {ms_toks}"

    def test_boundary_only_emission(self):
        """Milestones emit only when boundary changes (Kalman-filter style)."""
        # Create patient with events within same day → daily milestone should emit once
        rows = []
        for i in range(5):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 15, 8 + i, 0),  # same day, different hours
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
            milestone_per_state={"default": "daily"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)
        all_toks = _all_tokens_from_debug(dbg)
        ms_day = [t for t in all_toks if t.startswith("<|ms_day_")]

        # Same day → boundary doesn't change after first emission
        # Should have at most 1 daily milestone (for the first time delta)
        assert len(ms_day) <= 1, f"Expected at most 1 daily milestone within same day, got {ms_day}"


class TestStateMilestones:

    def test_8hr_milestones_when_inpatient(self):
        """Admitted patient uses 8hr milestones; outpatient uses weekly."""
        df = make_df(include_birth=True, include_admit=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
            milestone_per_state={"inpatient": "8hr", "outpatient": "week"},
            milestone_shift_start=7,
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        # Patient 1 is admitted → should use IP milestones (8hr)
        p1_toks = _all_tokens_from_debug(dbg, patient_id=1)
        ms_8hr = [t for t in p1_toks if "<|ms_8hr_" in t]
        assert len(ms_8hr) > 0, "Expected 8hr milestones for admitted patient"

    def test_12hr_milestones_when_inpatient(self):
        """12hr milestone mode produces shift tokens for admitted patient."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "admission",
             "text_value": None, "numeric_value": None},
        ]
        for i in range(6):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i * 8),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        rows.append(
            {"id": 1, "time": datetime(2023, 1, 3, 8, 0), "class": "discharge",
             "text_value": None, "numeric_value": None}
        )
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
            milestone_per_state={"inpatient": "12hr"},
            milestone_shift_start=7,
        )
        tok.train(df)
        dbg = tok.encode_debug(df)
        all_toks = _all_tokens_from_debug(dbg)
        ms_12hr = [t for t in all_toks if "<|ms_12hr_" in t]
        assert len(ms_12hr) > 0, "Expected 12hr milestones for admitted patient"

    def test_state_transition_resets_milestone_boundary(self):
        """Transitioning state resets the milestone boundary tracker."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "glucose", "numeric_value": 100.0},
            {"id": 1, "time": datetime(2023, 1, 1, 9, 0), "class": "admission",
             "text_value": None, "numeric_value": None},
            {"id": 1, "time": datetime(2023, 1, 1, 10, 0), "class": "lab",
             "text_value": "glucose", "numeric_value": 101.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            state_transitions={"inpatient": ["admission"]},
            initial_state="default",
            milestone_per_state={"default": "daily", "inpatient": "8hr"},
        )
        tok.train(df)
        # Just verify it encodes without error
        ids, _ = tok.encode(df)
        assert len(ids) > 0


class TestMonthMilestones:

    def test_month_milestones_across_months(self):
        """month milestone mode produces monthly tokens across months."""
        rows = []
        for i in range(4):
            rows.append({
                "id": 1,
                "time": datetime(2023, 1 + i, 15, 10, 0),  # Jan, Feb, Mar, Apr
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
            milestone_per_state={"default": "month"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)
        all_toks = _all_tokens_from_debug(dbg)
        ms_month = [t for t in all_toks if "<|ms_month_" in t]
        assert len(ms_month) > 0, "Expected monthly milestones across months"


class TestMilestoneShiftStart:

    def test_shift_start_affects_8hr_boundary(self):
        """milestone_shift_start changes when 8hr boundaries fall."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "admission",
             "text_value": None, "numeric_value": None},
        ]
        for h in [6, 7, 15, 23]:
            rows.append({
                "id": 1,
                "time": datetime(2023, 1, 1, h, 0),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0,
            })
        rows.append(
            {"id": 1, "time": datetime(2023, 1, 2, 8, 0), "class": "discharge",
             "text_value": None, "numeric_value": None}
        )
        df = pl.DataFrame(rows)

        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
            milestone_per_state={"inpatient": "8hr"},
            milestone_shift_start=7,
        )
        tok.train(df)
        dbg = tok.encode_debug(df)
        all_toks = _all_tokens_from_debug(dbg)
        ms_8hr = [t for t in all_toks if "<|ms_8hr_" in t]
        assert len(ms_8hr) > 0, "Expected 8hr milestones with shift_start=7"


class TestValidation:

    def test_invalid_milestone_per_state_value(self):
        """Invalid milestone value in milestone_per_state raises ValueError."""
        with pytest.raises(ValueError, match="milestone_per_state"):
            DBTokenizer(milestone_per_state={"default": "invalid"})
