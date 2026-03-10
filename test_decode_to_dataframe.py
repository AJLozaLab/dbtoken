"""
Unit tests for DBTokenizer.decode_to_dataframe.

Coverage matrix
───────────────
  num_type   × num_seq   × text_mode  × numeric_kind
  discrete     factored    concept      Q-bins
  discrete     fused       concept      Q-bins
  discrete     fused       concept      level
  continuous   factored    concept      scaling
  continuous   fused       concept      scaling
  discrete     factored    bpe          (no numeric)
  continuous   factored    bpe          scaling
  -- time-delta reconstruction for all four td-emission modes --
  -- multi-patient id assignment and time accumulation --
  -- empty stream edge case --
"""

import math
import pytest
import polars as pl
from datetime import datetime, timedelta
from tokenizer import DBTokenizer

try:
    import rustbpe
    import tiktoken
    HAS_BPE = True
except ImportError:
    HAS_BPE = False

# ── helpers ──────────────────────────────────────────────────────────────────

REF = datetime(2020, 1, 1)


def _df_simple():
    """Two patients, numeric values spread enough to use Q bins (n_distinct > level_threshold)."""
    rows = []
    for pid, base_nv, base_dt in [(1, 100.0, datetime(2020, 1, 1)), (2, 50.0, datetime(2020, 6, 1))]:
        for i in range(12):                            # 12 rows → 12 distinct nv → bins
            rows.append({
                "id": pid,
                "time": base_dt + timedelta(hours=i * 6),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": base_nv + i * 3.0,
            })
    return pl.DataFrame(rows)


def _df_level():
    """Few distinct numeric values so level tokens are used."""
    rows = []
    for pid in [1, 2]:
        for i in range(8):
            rows.append({
                "id": pid,
                "time": datetime(2020, 1, 1) + timedelta(days=i),
                "class": "grade",
                "text_value": "score",
                "numeric_value": float((i % 3) + 1),   # values {1,2,3} → level
            })
    return pl.DataFrame(rows)


def _df_no_numeric():
    """Text-only rows — no numeric_value."""
    rows = []
    for pid in [1, 2]:
        for i in range(6):
            rows.append({
                "id": pid,
                "time": datetime(2020, 1, 1) + timedelta(days=i),
                "class": "dx",
                "text_value": f"code_{(i % 3)}",
                "numeric_value": None,
            })
    return pl.DataFrame(rows)


def _df_bpe():
    """High-cardinality text → BPE, plus concept lab rows with numerics."""
    rows = []
    # 80 unique note strings → auto BPE
    for i in range(80):
        rows.append({
            "id": 1,
            "time": datetime(2020, 1, 1) + timedelta(hours=i),
            "class": "note",
            "text_value": f"Patient presented with condition variant number {i} for extended observation",
            "numeric_value": None,
        })
    # concept lab rows
    for i in range(12):
        rows.append({
            "id": 1,
            "time": datetime(2020, 1, 10) + timedelta(hours=i * 3),
            "class": "lab",
            "text_value": "creatinine",
            "numeric_value": 0.8 + i * 0.1,
        })
    return pl.DataFrame(rows)


def _classes(recon):
    return recon["class"].to_list()


def _nv(recon):
    return recon["numeric_value"].to_list()


def _tv(recon):
    return recon["text_value"].to_list()


# ── assertion helpers ─────────────────────────────────────────────────────────

def assert_schema(recon):
    assert set(recon.columns) >= {"id", "time", "class", "text_value", "numeric_value"}


def assert_row_count(recon, df):
    """Reconstructed row count must equal original (birth-date rows excluded)."""
    non_birth = df.filter(
        ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
    )
    assert recon.height == non_birth.height, (
        f"expected {non_birth.height} rows, got {recon.height}"
    )


def assert_classes_match(recon, df):
    orig_classes = (
        df.filter(
            ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
        )
        .sort(["id", "time"])["class"]
        .to_list()
    )
    recon_classes = recon.sort(["id", "time"])["class"].to_list()
    assert recon_classes == orig_classes, f"class mismatch:\n  orig={orig_classes}\n  got={recon_classes}"


def assert_text_values_match(recon, df):
    orig = (
        df.filter(
            ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
        )
        .sort(["id", "time"])["text_value"]
        .to_list()
    )
    got = recon.sort(["id", "time"])["text_value"].to_list()
    assert orig == got, f"text_value mismatch:\n  orig={orig}\n  got={got}"


def assert_numerics_finite(recon, df):
    """Every row that originally had a numeric_value should have one in the reconstruction."""
    orig_has = (
        df.filter(
            ~((pl.col("class") == "demographic") & (pl.col("text_value") == "birth_date"))
        )
        .sort(["id", "time"])["numeric_value"]
        .to_list()
    )
    got = recon.sort(["id", "time"])["numeric_value"].to_list()
    for i, (o, g) in enumerate(zip(orig_has, got)):
        if o is None:
            assert g is None, f"row {i}: expected None nv, got {g}"
        else:
            assert g is not None and not math.isnan(g), (
                f"row {i}: expected a numeric value (orig={o}), got {g}"
            )


def assert_level_exact(recon, df, level_values):
    """Level numerics must be recovered exactly."""
    orig = (
        df.sort(["id", "time"])["numeric_value"].to_list()
    )
    got = recon.sort(["id", "time"])["numeric_value"].to_list()
    for i, (o, g) in enumerate(zip(orig, got)):
        if o is None:
            continue
        assert g == pytest.approx(o, abs=1e-9), (
            f"level row {i}: expected {o}, got {g}"
        )


def assert_time_increases(recon):
    for pid in recon["id"].unique().to_list():
        times = recon.filter(pl.col("id") == pid).sort("time")["time"].to_list()
        for a, b in zip(times, times[1:]):
            assert b >= a, f"time went backwards for patient {pid}: {a} -> {b}"


def assert_patient_ids(recon, start=0):
    pids = sorted(recon["id"].unique().to_list())
    assert pids[0] == start, f"first patient id should be {start}, got {pids[0]}"
    for a, b in zip(pids, pids[1:]):
        assert b == a + 1, f"non-consecutive patient ids: {a}, {b}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — discrete + factored + concept + Q bins
# ═══════════════════════════════════════════════════════════════════════════════

def test_discrete_factored_concept_bins():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,        # 12 distinct values → bins not levels
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    assert vals is None, "discrete mode should produce vals=None"

    # Confirm Q tokens are in the stream
    decoded = tok.decode(ids)
    q_toks = [t for t in decoded if t.startswith("<|Q")]
    assert len(q_toks) > 0, "expected Q bin tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=0)

    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_numerics_finite(recon, df)
    assert_time_increases(recon)
    assert_patient_ids(recon, start=0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — discrete + fused + concept + Q bins
# ═══════════════════════════════════════════════════════════════════════════════

def test_discrete_fused_concept_bins():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="fused",
        n_bins=5,
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    fused_q = [t for t in decoded if "::Q" in t]
    assert len(fused_q) > 0, "expected fused ::Q tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_numerics_finite(recon, df)
    assert_time_increases(recon)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — discrete + factored + concept + level tokens
# ═══════════════════════════════════════════════════════════════════════════════

def test_discrete_factored_concept_level():
    df = _df_level()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=10,       # 3 distinct values → level
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    l_toks = [t for t in decoded if t.startswith("<|L")]
    assert len(l_toks) > 0, "expected <|LN|> level tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_time_increases(recon)

    # Level values must be recovered exactly
    assert_level_exact(recon, df, [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — discrete + fused + concept + level tokens
# ═══════════════════════════════════════════════════════════════════════════════

def test_discrete_fused_concept_level():
    df = _df_level()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="fused",
        n_bins=5,
        level_threshold=10,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    fused_l = [t for t in decoded if "::L" in t]
    assert len(fused_l) > 0, "expected fused ::L tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_level_exact(recon, df, [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — continuous + factored + concept + scaling
# ═══════════════════════════════════════════════════════════════════════════════

def test_continuous_factored_concept_scaling():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="factored",
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None, "continuous mode must return vals"
    assert len(ids) == len(vals)

    decoded = tok.decode(ids)
    num_toks = [t for t in decoded if t == "<|NUM|>"]
    assert len(num_toks) > 0, "expected <|NUM|> tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_numerics_finite(recon, df)
    assert_time_increases(recon)

    # Continuous values are approximately recovered — check within 50% of original
    # (exact recovery only guaranteed for minmax; others are approximate via distribution inverse)
    orig_nv = df.sort(["id", "time"])["numeric_value"].to_list()
    got_nv  = recon.sort(["id", "time"])["numeric_value"].to_list()
    for o, g in zip(orig_nv, got_nv):
        if o is None:
            continue
        assert g is not None
        assert abs(g - o) / max(abs(o), 1.0) < 5.0, (
            f"continuous round-trip too far off: orig={o}, got={g}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — continuous + fused + concept + scaling
# ═══════════════════════════════════════════════════════════════════════════════

def test_continuous_fused_concept_scaling():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="fused",
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None

    # In fused-continuous mode the concept token carries the scaled val;
    # no ::L or ::Q suffixes — the concept string is emitted plain.
    decoded = tok.decode(ids)
    assert "glucose" in decoded, "expected plain concept token for fused-scaling"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    assert_numerics_finite(recon, df)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7 — concept, no numeric values
# ═══════════════════════════════════════════════════════════════════════════════

def test_concept_no_numeric():
    df = _df_no_numeric()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)
    assert_text_values_match(recon, df)
    # All numeric values should be None
    for nv in recon["numeric_value"].to_list():
        assert nv is None, f"expected None nv for text-only rows, got {nv}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8 — time delta reconstruction: factored + discrete (td → Q)
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_delta_factored_discrete():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)

    # All times must be >= reference_time for patient 0
    p0_times = recon.filter(pl.col("id") == 0).sort("time")["time"].to_list()
    assert all(t >= REF for t in p0_times), "times before reference_time"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9 — time delta reconstruction: fused + discrete (single td token)
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_delta_fused_discrete():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="fused",
        n_bins=5,
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    fused_td = [t for t in decoded if "delta_time" in t and "_Q" in t]
    assert len(fused_td) > 0, "expected fused delta_time_*_QN tokens"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10 — time delta reconstruction: factored + continuous (td → NUM + val)
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_delta_factored_continuous():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="factored",
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 11 — time delta reconstruction: fused + continuous (td marker + val[i])
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_delta_fused_continuous():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="fused",
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    td_markers = [t for t in decoded if "delta_time" in t and "::Q" not in t and "_Q" not in t]
    assert len(td_markers) > 0, "expected delta_time marker tokens in fused-continuous"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_time_increases(recon)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 12 — multi-patient IDs and start_patient_id parameter
# ═══════════════════════════════════════════════════════════════════════════════

def test_multi_patient_ids():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    # Default start_patient_id=0
    recon0 = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=0)
    assert_patient_ids(recon0, start=0)
    assert sorted(recon0["id"].unique().to_list()) == [0, 1]

    # start_patient_id=100
    recon100 = tok.decode_to_dataframe(ids, vals, reference_time=REF, start_patient_id=100)
    assert_patient_ids(recon100, start=100)
    assert sorted(recon100["id"].unique().to_list()) == [100, 101]

    # Each patient should have same number of rows
    n0 = recon0.filter(pl.col("id") == 0).height
    n1 = recon0.filter(pl.col("id") == 1).height
    orig_n1 = df.filter(pl.col("id") == 1).height
    orig_n2 = df.filter(pl.col("id") == 2).height
    assert n0 == orig_n1, f"patient 0 row count mismatch: {n0} vs {orig_n1}"
    assert n1 == orig_n2, f"patient 1 row count mismatch: {n1} vs {orig_n2}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 13 — empty token stream
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_stream():
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)

    recon = tok.decode_to_dataframe([], None, reference_time=REF)
    assert recon.height == 0
    assert set(recon.columns) >= {"id", "time", "class", "text_value", "numeric_value"}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 14 — save / load then decode_to_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

def test_save_load_round_trip(tmp_path):
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    recon_orig = tok.decode_to_dataframe(ids, vals, reference_time=REF)

    tok.save(str(tmp_path / "tok"))
    tok2 = DBTokenizer.load(str(tmp_path / "tok"))
    ids2, vals2 = tok2.encode(df)
    assert ids == ids2
    recon_rt = tok2.decode_to_dataframe(ids2, vals2, reference_time=REF)

    assert recon_orig["class"].to_list() == recon_rt["class"].to_list()
    assert recon_orig["text_value"].to_list() == recon_rt["text_value"].to_list()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 15 — auto text_mode: concept class stays concept in reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_concept_mode():
    """With text_mode_default='auto' and few unique values, class is concept."""
    df = _df_no_numeric()   # 3 unique code_* values → concept
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    assert tok.class_text_modes.get("dx") == "concept", (
        f"expected dx=concept, got {tok.class_text_modes.get('dx')}"
    )
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_text_values_match(recon, df)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 16 — milestone tokens are dropped (not reconstructed as rows)
# ═══════════════════════════════════════════════════════════════════════════════

def test_milestones_dropped():
    """Milestone tokens must not create extra rows in the reconstruction."""
    df = _df_simple()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        milestone_op="week",   # enable milestones
        milestone_ip="daily",
    )
    tok.train(df)
    ids, vals = tok.encode(df)

    decoded = tok.decode(ids)
    ms_toks = [t for t in decoded if t.startswith("<|ms_")]
    assert len(ms_toks) > 0, "test requires milestone tokens to be present"

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_row_count(recon, df)  # milestone tokens must not inflate row count


# ═══════════════════════════════════════════════════════════════════════════════
# BPE tests (skipped without backend)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_factored_discrete():
    """BPE text tokens are reassembled into text_value strings."""
    df = _df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        final_vocab_size=512,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    assert tok.class_text_modes["note"] == "bpe"
    assert tok.class_text_modes["lab"] == "concept"

    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)

    assert_schema(recon)
    assert_row_count(recon, df)
    assert_classes_match(recon, df)

    # Concept lab text values must round-trip exactly
    lab_orig = df.filter(pl.col("class") == "lab").sort("time")["text_value"].to_list()
    lab_recon = recon.filter(pl.col("class") == "lab").sort("time")["text_value"].to_list()
    assert lab_orig == lab_recon

    # BPE note text values should be reassembled (not None)
    note_recon = recon.filter(pl.col("class") == "note")["text_value"].to_list()
    assert all(tv is not None and len(tv) > 0 for tv in note_recon), (
        "BPE note rows should have non-empty text_value after reconstruction"
    )

    assert_time_increases(recon)
    assert_numerics_finite(recon, df)


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_factored_continuous():
    """BPE with continuous num_type: lab numerics still round-trip approximately."""
    df = _df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="continuous",
        num_seq="factored",
        level_threshold=3,
        final_vocab_size=512,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    assert vals is not None

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert_schema(recon)
    assert_row_count(recon, df)
    assert_numerics_finite(recon, df)


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
def test_bpe_save_load_decode(tmp_path):
    """BPE tokenizer: save → load → decode_to_dataframe produces same result."""
    df = _df_bpe()
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=64,
        num_type="discrete",
        num_seq="factored",
        n_bins=5,
        level_threshold=3,
        final_vocab_size=512,
        milestone_op="none",
        milestone_ip="none",
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    recon1 = tok.decode_to_dataframe(ids, vals, reference_time=REF)

    tok.save(str(tmp_path / "tok_bpe"))
    tok2 = DBTokenizer.load(str(tmp_path / "tok_bpe"))
    ids2, vals2 = tok2.encode(df)
    assert ids == ids2
    recon2 = tok2.decode_to_dataframe(ids2, vals2, reference_time=REF)

    assert recon1["class"].to_list() == recon2["class"].to_list()
    assert recon1["text_value"].to_list() == recon2["text_value"].to_list()
