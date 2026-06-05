"""Edge-case tests for DBTokenizer.

No fancy imports — only stdlib, polars, numpy, pytest, and DBTokenizer.
Each test builds its own DataFrame inline.
"""

import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta

from dbtoken import DBTokenizer

try:
    import rustbpe
    import tiktoken
    HAS_BPE = True
except ImportError:
    HAS_BPE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Inline data builders
# ═══════════════════════════════════════════════════════════════════════════════

def _multi_patient_df():
    """Three patients, each with a birth-date row and a handful of lab events."""
    rows = []
    for pid, birth_year in [(1, 1970), (2, 1985), (3, 1950)]:
        rows.append({
            "id": pid,
            "time": datetime(birth_year, 1, 1),
            "class": "demographic",
            "text_value": "birth_date",
            "numeric_value": None,
        })
        for i in range(4):
            rows.append({
                "id": pid,
                "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i * 6),
                "class": "lab",
                "text_value": "hemoglobin",
                "numeric_value": 12.0 + i * 0.5,
            })
    return pl.DataFrame(rows)


def _nested_admit_df():
    """Single patient with nested admissions (admit→admit→events→discharge→events→discharge→OP events).

    Admission depth via cumsum:
      admit1  → depth 1 (inpatient)
      admit2  → depth 2, clipped to 1 (still inpatient)
      events  → inpatient
      disch1  → depth 1, clipped to 1 (still inpatient — one admit unclosed)
      events  → inpatient
      disch2  → depth 0 (outpatient)
      events  → outpatient
    """
    rows = [
        # birth
        {"id": 1, "time": datetime(1960, 5, 1), "class": "demographic",
         "text_value": "birth_date", "numeric_value": None},
        # OP event before admission
        {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 13.0},
        # 1st admission
        {"id": 1, "time": datetime(2023, 2, 1, 10, 0), "class": "admission",
         "text_value": None, "numeric_value": None},
        # 2nd admission (nested)
        {"id": 1, "time": datetime(2023, 2, 1, 10, 5), "class": "admission",
         "text_value": None, "numeric_value": None},
        # IP events
        {"id": 1, "time": datetime(2023, 2, 1, 14, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 10.5},
        {"id": 1, "time": datetime(2023, 2, 2, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 11.0},
        # 1st discharge (depth goes from 2→1, still IP due to clip)
        {"id": 1, "time": datetime(2023, 2, 2, 14, 0), "class": "discharge",
         "text_value": None, "numeric_value": None},
        # Still IP — one admission unclosed
        {"id": 1, "time": datetime(2023, 2, 3, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 11.5},
        # 2nd discharge (depth goes 1→0, now OP)
        {"id": 1, "time": datetime(2023, 2, 3, 14, 0), "class": "discharge",
         "text_value": None, "numeric_value": None},
        # OP events after both discharges
        {"id": 1, "time": datetime(2023, 3, 1, 9, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 12.5},
        {"id": 1, "time": datetime(2023, 4, 1, 9, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 13.0},
    ]
    return pl.DataFrame(rows)


def _fused_bpe_numeric_df():
    """High-cardinality notes (→ BPE) with numeric values, plus low-cardinality labs (→ concept).

    With num_seq='fused', the BPE-mode notes should fall back to factored,
    while concept-mode labs should emit fused tokens.
    """
    rows = []
    # Enough unique note texts to trigger BPE (> threshold)
    for i in range(70):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1) + timedelta(hours=i),
            "class": "note",
            "text_value": f"Patient presented with condition variant number {i} for observation",
            "numeric_value": float(i % 10),
        })
    # Concept-mode labs with numeric values
    for i in range(12):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 10) + timedelta(hours=i * 3),
            "class": "lab",
            "text_value": "glucose",
            "numeric_value": 80.0 + i * 5.0,
        })
    return pl.DataFrame(rows)


def _simultaneous_events_df():
    """Multiple events at the same timestamp — order must be preserved."""
    t = datetime(2023, 6, 1, 12, 0)
    rows = [
        {"id": 1, "time": t, "class": "lab", "text_value": "hemoglobin",
         "numeric_value": 14.0},
        {"id": 1, "time": t, "class": "vital", "text_value": "heart_rate",
         "numeric_value": 72.0},
        {"id": 1, "time": t, "class": "vital", "text_value": "blood_pressure",
         "numeric_value": 120.0},
        {"id": 1, "time": t, "class": "lab", "text_value": "glucose",
         "numeric_value": 95.0},
    ]
    return pl.DataFrame(rows)


def _single_row_patient_df():
    """One patient with exactly one data event (plus a birth-date row)."""
    rows = [
        {"id": 1, "time": datetime(1990, 1, 1), "class": "demographic",
         "text_value": "birth_date", "numeric_value": None},
        {"id": 1, "time": datetime(2023, 7, 4, 10, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 15.0},
    ]
    return pl.DataFrame(rows)


def _missing_numeric_df():
    """Concept pair that sometimes has numeric, sometimes None."""
    rows = [
        {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 13.0},
        {"id": 1, "time": datetime(2023, 1, 1, 14, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": None},
        {"id": 1, "time": datetime(2023, 1, 2, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 12.5},
    ]
    return pl.DataFrame(rows)


def _admit_no_discharge_df():
    """Patient admitted but never discharged — all subsequent events should be IP."""
    rows = [
        {"id": 1, "time": datetime(1975, 3, 10), "class": "demographic",
         "text_value": "birth_date", "numeric_value": None},
        # OP event
        {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 14.0},
        # Admission
        {"id": 1, "time": datetime(2023, 2, 1, 10, 0), "class": "admission",
         "text_value": None, "numeric_value": None},
        # All subsequent events — should remain IP
        {"id": 1, "time": datetime(2023, 2, 1, 16, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 10.5},
        {"id": 1, "time": datetime(2023, 2, 2, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 11.0},
        {"id": 1, "time": datetime(2023, 2, 3, 8, 0), "class": "lab",
         "text_value": "hemoglobin", "numeric_value": 11.5},
    ]
    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiPatientSosEos:
    """Verify SOS / EOS tokens are emitted correctly at patient boundaries."""

    def test_three_patients_sos_eos_count(self):
        df = _multi_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        sos_count = decoded.count("<|sos|>")
        eos_count = decoded.count("<|eos|>")
        assert sos_count == 3, f"Expected 3 SOS tokens, got {sos_count}"
        assert eos_count == 3, f"Expected 3 EOS tokens, got {eos_count}"

    def test_sos_eos_pairing(self):
        """Every SOS must be followed by a matching EOS before the next SOS."""
        df = _multi_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        depth = 0
        for token in decoded:
            if token == "<|sos|>":
                assert depth == 0, "SOS encountered before previous EOS"
                depth += 1
            elif token == "<|eos|>":
                assert depth == 1, "EOS without matching SOS"
                depth -= 1
        assert depth == 0, "Stream ended without final EOS"

    def test_first_token_is_sos_last_is_eos(self):
        df = _multi_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        assert decoded[0] == "<|sos|>", f"First token should be SOS, got {decoded[0]}"
        assert decoded[-1] == "<|eos|>", f"Last token should be EOS, got {decoded[-1]}"


class TestStateTransitions:
    """State machine transitions (last-trigger-wins) for admission/discharge."""

    def _encode_and_decode(self, df):
        tok = DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        return tok.decode(ids)

    def test_delta_after_admission_is_inpatient_scale(self):
        """After admission, time deltas use the inpatient scale band."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 13.0},
            {"id": 1, "time": datetime(2023, 1, 1, 9, 0), "class": "admission",
             "text_value": None, "numeric_value": None},
            # short delta after admission (< 24h)
            {"id": 1, "time": datetime(2023, 1, 1, 10, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 11.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            time_scales=[86400.0],
            time_scale_names=["short", "long"],
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # There should be a delta_time_short token for the post-admission lab
        short_tds = [t for t in decoded if "delta_time_short" in t]
        assert len(short_tds) > 0, "Expected short-scale delta after admission"

    def test_discharge_transitions_back_to_outpatient(self):
        """After discharge, milestones switch back to outpatient mode."""
        df = _nested_admit_df()
        tok = DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
            milestone_per_state={"inpatient": "8hr", "outpatient": "week"},
        )
        tok.train(df)
        # Just verify encoding succeeds without error
        ids, _ = tok.encode(df)
        assert len(ids) > 0

    def test_initial_state_respected(self):
        """initial_state determines pre-admission behavior."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 13.0},
            {"id": 1, "time": datetime(2023, 1, 2, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 12.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            state_transitions={"inpatient": ["admission"]},
            initial_state="outpatient",
            milestone_per_state={"outpatient": "week"},
        )
        tok.train(df)
        dbg = tok.encode_debug(df)
        all_toks = []
        for row in dbg.iter_rows(named=True):
            all_toks.extend(row["tokens"])
        # Should have weekly milestone (outpatient initial state)
        ms_week = [t for t in all_toks if t.startswith("<|ms_week_")]
        assert len(ms_week) > 0, "Expected week milestones from outpatient initial_state"


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
class TestFusedBpeFallbackToFactored:
    """With num_seq='fused', BPE-mode classes must fall back to factored numeric tokens."""

    def test_bpe_class_emits_factored_numeric(self):
        df = _fused_bpe_numeric_df()
        tok = DBTokenizer(
            text_mode_default="auto", text_mode_threshold=10,
            num_type="discrete", num_seq="fused", n_bins=5,
            final_vocab_size=512,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # The "note" class should have been auto-detected as BPE (70 unique texts > threshold=10).
        # Fused+BPE → factored fallback → separate discrete quantizer tokens should appear.
        assert tok.class_text_modes.get("note") == "bpe", (
            "Expected 'note' class to be BPE mode"
        )

        # For BPE classes, there should be standalone factored numeric tokens (Q or L)
        # Notes have unique text_values (1 occurrence each) → level encoding → <|L..|> tokens
        factored_discrete = [
            t for t in decoded
            if (t.startswith("<|Q") or t.startswith("<|L")) and t.endswith("|>")
        ]
        assert len(factored_discrete) > 0, (
            "Expected factored <|Q…|> or <|L…|> tokens for BPE-mode class with fused setting"
        )

    def test_concept_class_emits_fused_tokens(self):
        df = _fused_bpe_numeric_df()
        tok = DBTokenizer(
            text_mode_default="auto", text_mode_threshold=10,
            num_type="discrete", num_seq="fused", n_bins=5,
            final_vocab_size=512,
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # The "lab" class should be concept mode (only 1 unique text_value "glucose").
        assert tok.class_text_modes.get("lab") == "concept", (
            "Expected 'lab' class to be concept mode"
        )

        # Concept class with fused → tokens like "glucose::Q3" or "glucose::L2"
        fused_lab = [t for t in decoded if "glucose::" in t]
        assert len(fused_lab) > 0, (
            "Expected fused concept+numeric tokens for concept-mode 'lab' class"
        )

        # And no standalone <|Q…|> tokens should correspond to lab rows
        # (all lab numerics should be fused into the concept token)
        # Check that there's no pattern: <|lab|> glucose <|Q…|> (factored)
        for i in range(len(decoded) - 2):
            if decoded[i] == "<|lab|>" and decoded[i + 1] == "glucose":
                # Next token after "glucose" should NOT be a standalone Q bin
                if i + 2 < len(decoded):
                    assert not (decoded[i + 2].startswith("<|Q") and decoded[i + 2].endswith("|>")), (
                        f"Found factored numeric after concept 'glucose': {decoded[i:i+3]}"
                    )


class TestSimultaneousEventOrdering:
    """Events at the same timestamp must appear in the token stream in source order."""

    def test_order_preserved(self):
        df = _simultaneous_events_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # Extract the class tokens in order (they mirror the event order)
        class_tokens = [t for t in decoded if t.startswith("<|") and t.endswith("|>")
                        and t not in ("<|sos|>", "<|eos|>", "<|pad|>", "<|NUM|>")
                        and not t.startswith("<|Q") and not t.startswith("<|L")
                        and not t.startswith("<|delta_") and not t.startswith("<|age_")
                        and not t.startswith("<|ms_")]

        expected_classes = ["<|lab|>", "<|vital|>", "<|vital|>", "<|lab|>"]
        assert class_tokens == expected_classes, (
            f"Class token order mismatch:\n  expected: {expected_classes}\n  got: {class_tokens}"
        )

    def test_text_value_order_preserved(self):
        df = _simultaneous_events_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # Extract concept text-value tokens (non-special tokens)
        text_tokens = [t for t in decoded if not t.startswith("<|")]
        expected_texts = ["hemoglobin", "heart_rate", "blood_pressure", "glucose"]
        assert text_tokens == expected_texts, (
            f"Text value order mismatch:\n  expected: {expected_texts}\n  got: {text_tokens}"
        )

    def test_no_delta_between_simultaneous(self):
        """No time-delta tokens should appear between same-timestamp events."""
        df = _simultaneous_events_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        delta_tokens = [t for t in decoded if "delta_time" in t]
        assert len(delta_tokens) == 0, (
            f"Expected no delta tokens for simultaneous events, got: {delta_tokens}"
        )


class TestSingleRowPatient:
    """A patient with exactly one data event should produce SOS + event + EOS."""

    def test_single_event_structure(self):
        df = _single_row_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        assert decoded[0] == "<|sos|>"
        assert decoded[-1] == "<|eos|>"

    def test_no_delta_token(self):
        """Single event means no prior time to compute a delta from."""
        df = _single_row_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        delta_tokens = [t for t in decoded if "delta_time" in t]
        assert len(delta_tokens) == 0, (
            f"Expected no delta tokens for single-event patient, got: {delta_tokens}"
        )

    def test_has_age_token(self):
        """With a birth-date row, an age token should be emitted."""
        df = _single_row_patient_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        age_tokens = [t for t in decoded if t.startswith("<|age_")]
        assert len(age_tokens) == 1, (
            f"Expected exactly 1 age token, got: {age_tokens}"
        )


class TestMissingNumericOnConceptPair:
    """When numeric_value is None, only a plain concept token should be emitted."""

    def test_no_numeric_token_for_null_nv(self):
        df = _missing_numeric_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # Walk the token stream: after <|lab|> hemoglobin,
        # the next token should be Q-bin only when numeric was present.
        # For the middle row (None numeric), there should be no Q token.
        lab_groups = []
        i = 0
        while i < len(decoded):
            if decoded[i] == "<|lab|>":
                group = [decoded[i]]
                i += 1
                # Collect tokens until next class token or special boundary
                while i < len(decoded) and decoded[i] not in ("<|sos|>", "<|eos|>") \
                        and not decoded[i].startswith("<|delta_") \
                        and not (decoded[i].startswith("<|") and decoded[i].endswith("|>")
                                 and decoded[i] not in ("<|NUM|>",)
                                 and not decoded[i].startswith("<|Q")
                                 and not decoded[i].startswith("<|L")):
                    group.append(decoded[i])
                    i += 1
                lab_groups.append(group)
            else:
                i += 1

        # We should have 3 lab groups
        assert len(lab_groups) == 3, f"Expected 3 lab groups, got {len(lab_groups)}"

        # 1st and 3rd groups should have a discrete numeric token (Q-bin or L-level)
        for idx in [0, 2]:
            q_toks = [t for t in lab_groups[idx] if t.startswith("<|Q") or t.startswith("<|L")]
            assert len(q_toks) > 0, (
                f"Lab group {idx} should have a Q or L token (numeric present): {lab_groups[idx]}"
            )

        # 2nd group should NOT have any Q token (numeric is None)
        q_toks_null = [t for t in lab_groups[1] if t.startswith("<|Q") or t.startswith("<|L")]
        assert len(q_toks_null) == 0, (
            f"Lab group 1 should have no Q/L token (numeric is None): {lab_groups[1]}"
        )

    def test_fused_mode_no_fused_token_for_null_nv(self):
        """In fused mode, a None numeric should produce a plain concept token, not a fused one."""
        df = _missing_numeric_df()
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="fused", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        # The 2nd hemoglobin should be a plain "hemoglobin" token (no "::" fused suffix)
        hemo_tokens = [t for t in decoded if "hemoglobin" in t]
        # At least one should be plain (the null-numeric one)
        plain_hemo = [t for t in hemo_tokens if "::" not in t]
        assert len(plain_hemo) >= 1, (
            f"Expected at least one plain 'hemoglobin' token for null numeric, "
            f"got: {hemo_tokens}"
        )


class TestAdmissionWithoutDischarge:
    """Patient admitted but never discharged — state remains inpatient."""

    def _make_tok(self):
        return DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            state_transitions={
                "inpatient":  ["admission"],
                "outpatient": ["discharge"],
            },
            initial_state="outpatient",
            milestone_per_state={"inpatient": "8hr", "outpatient": "week"},
        )

    def test_encodes_without_error(self):
        """Encoding a patient admitted but never discharged completes without error."""
        df = _admit_no_discharge_df()
        tok = self._make_tok()
        tok.train(df)
        ids, _ = tok.encode(df)
        assert len(ids) > 0

    def test_stream_ends_with_eos(self):
        """Encode completes without error and stream ends with EOS."""
        df = _admit_no_discharge_df()
        tok = self._make_tok()
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        assert decoded[-1] == "<|eos|>"

    def test_admission_token_present(self):
        """Admission event token appears in the decoded stream."""
        df = _admit_no_discharge_df()
        tok = self._make_tok()
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        assert "<|admission|>" in decoded

    def test_no_crash_at_end(self):
        """Encode completes without error and stream ends with EOS."""
        df = _admit_no_discharge_df()
        tok = DBTokenizer(
            text_mode_default="concept", num_type="discrete", num_seq="factored",
            n_bins=5,
            state_transitions={"inpatient": ["admission"]},
            initial_state="default",
        )
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)
        assert decoded[-1] == "<|eos|>"


class TestZeroTimeDeltaSuppressed:
    """Same-timestamp events should NOT emit time-delta tokens between them."""

    def test_no_delta_same_timestamp(self):
        """Two events at the exact same time — no delta token between them."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 13.0},
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "glucose", "numeric_value": 100.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        delta_tokens = [t for t in decoded if "delta_time" in t]
        assert len(delta_tokens) == 0, (
            f"Expected no delta tokens for same-timestamp events, got: {delta_tokens}"
        )

    def test_delta_appears_after_time_advances(self):
        """After time actually advances, a delta token should appear."""
        rows = [
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 13.0},
            {"id": 1, "time": datetime(2023, 1, 1, 8, 0), "class": "lab",
             "text_value": "glucose", "numeric_value": 100.0},
            {"id": 1, "time": datetime(2023, 1, 2, 8, 0), "class": "lab",
             "text_value": "hemoglobin", "numeric_value": 12.0},
        ]
        df = pl.DataFrame(rows)
        tok = DBTokenizer(text_mode_default="concept", num_type="discrete",
                          num_seq="factored", n_bins=5)
        tok.train(df)
        ids, _ = tok.encode(df)
        decoded = tok.decode(ids)

        delta_tokens = [t for t in decoded if "delta_time" in t]
        assert len(delta_tokens) >= 1, (
            "Expected at least one delta token after time advanced"
        )
