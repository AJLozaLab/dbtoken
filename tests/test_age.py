"""Tests for age token generation, suppression, and Kalman-filter change emission."""
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


class TestAgeTokens:

    def test_age_tokens_present_with_birth_dates(self):
        """Birth-date rows generate age tokens."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=10,
            milestone_per_state={"default": "week"},
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        age_toks = [t for t in decoded if t.startswith("<|age_")]
        assert len(age_toks) > 0, "No age tokens found"

        # Patient 1 born 1980, events in 2023 → age ≈ 42
        assert any("age_42" in t for t in age_toks), (
            f"Expected age_42 for patient 1, got {age_toks}"
        )

    def test_no_age_without_birth_dates(self):
        """Without birth-date rows, no age tokens emitted."""
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

        age_toks = [t for t in decoded if t.startswith("<|age_")]
        assert len(age_toks) == 0, f"Expected no age tokens, got {age_toks}"

    def test_age_kalman_emission_no_duplicate(self):
        """Age token emitted only when age changes, not repeated within same year."""
        # Patient with events all within same year → same age → only 1 age token
        rows = [
            {"id": 1, "time": datetime(1990, 6, 1), "class": "demographic",
             "text_value": "birth_date", "numeric_value": None},
        ]
        for i in range(10):
            rows.append({
                "id": 1,
                "time": datetime(2023, 3, 1) + timedelta(days=i),  # same age all events
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
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        age_toks = [t for t in decoded if t.startswith("<|age_")]
        # Only the initial age at SOS — no subsequent duplicates
        assert len(age_toks) == 1, (
            f"Expected exactly 1 age token (Kalman: no change), got {age_toks}"
        )

    def test_age_boundary_crossing(self):
        """Age token re-emitted when patient's age changes mid-sequence."""
        # Patient born 2000-06-15, events from 2032-06-01 to 2033-06-30
        # Age changes from 31 to 32 to 33 over this span
        rows = [
            {"id": 1, "time": datetime(2000, 6, 15), "class": "demographic",
             "text_value": "birth_date", "numeric_value": None},
        ]
        # Events before birthday: age 31
        for i in range(3):
            rows.append({
                "id": 1,
                "time": datetime(2032, 1, 1) + timedelta(days=i * 60),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        # Events after birthday: age 32
        for i in range(3):
            rows.append({
                "id": 1,
                "time": datetime(2032, 7, 1) + timedelta(days=i * 60),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        # Events in next year after birthday: age 33
        for i in range(2):
            rows.append({
                "id": 1,
                "time": datetime(2033, 7, 1) + timedelta(days=i * 30),
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
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        age_toks = [t for t in decoded if t.startswith("<|age_")]
        # Should have age_31, age_32, and age_33
        age_values = sorted(set(age_toks))
        assert len(age_values) >= 2, (
            f"Expected at least 2 distinct age tokens for boundary crossing, got {age_values}"
        )

    def test_sos_before_age(self):
        """SOS token appears before the initial age token."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        sos_idx = decoded.index("<|sos|>")
        first_age = next(i for i, t in enumerate(decoded) if t.startswith("<|age_"))
        assert sos_idx < first_age, "SOS should come before first age token"


class TestMultiPatientAge:

    def test_each_patient_gets_own_age(self):
        """Each patient gets age computed from their own birth date."""
        df = make_df(include_birth=True)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
        )
        tok.train(df)
        dbg = tok.encode_debug(df)

        p1_toks = _all_tokens_from_debug(dbg, patient_id=1)
        p2_toks = _all_tokens_from_debug(dbg, patient_id=2)

        p1_ages = [t for t in p1_toks if t.startswith("<|age_")]
        p2_ages = [t for t in p2_toks if t.startswith("<|age_")]

        assert len(p1_ages) > 0, "Patient 1 should have age tokens"
        assert len(p2_ages) > 0, "Patient 2 should have age tokens"

        # Patient 1: born 1980, events 2023 → ~42
        # Patient 2: born 1955, events 2023 → ~67
        assert p1_ages[0] != p2_ages[0], (
            f"Different birth dates → different ages: P1={p1_ages[0]}, P2={p2_ages[0]}"
        )
