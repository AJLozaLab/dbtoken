"""Shared fixtures and data builders for DBTokenizer tests."""
import sys
from pathlib import Path

import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta

# Ensure the project root and tests dir are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import DBTokenizer

try:
    import rustbpe
    import tiktoken
    HAS_BPE = True
except ImportError:
    HAS_BPE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data builders (returned directly, not as fixtures so tests can choose params)
# ═══════════════════════════════════════════════════════════════════════════════

REF = datetime(2020, 1, 1)


def make_df(include_birth=True, include_admit=False):
    """Build a tiny two-patient DataFrame.

    Patient 1: born 1980-06-15, 8 lab events (hemoglobin/glucose alternating).
    Patient 2: born 1955-03-01, 6 hemoglobin labs spanning ~50 days.
    """
    rows = []
    # ── Patient 1 ─────────────────────────────────────────────────────
    if include_birth:
        rows.append({
            "id": 1,
            "time": datetime(1980, 6, 15),
            "class": "demographic",
            "text_value": "birth_date",
            "numeric_value": None,
        })
    if include_admit:
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0),
            "class": "admission",
            "text_value": None,
            "numeric_value": None,
        })
    for i in range(8):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i * 6),
            "class": "lab",
            "text_value": "hemoglobin" if i % 2 == 0 else "glucose",
            "numeric_value": 12.0 + i * 0.5 if i % 2 == 0 else 90.0 + i,
        })
    if include_admit:
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 3, 10, 0),
            "class": "discharge",
            "text_value": None,
            "numeric_value": None,
        })

    # ── Patient 2 ─────────────────────────────────────────────────────
    if include_birth:
        rows.append({
            "id": 2,
            "time": datetime(1955, 3, 1),
            "class": "demographic",
            "text_value": "birth_date",
            "numeric_value": None,
        })
    for i in range(6):
        rows.append({
            "id": 2,
            "time": datetime(2023, 3, 1, 10, 0) + timedelta(days=i * 10),
            "class": "lab",
            "text_value": "hemoglobin",
            "numeric_value": 11.0 + i * 0.3,
        })

    return pl.DataFrame(rows)


def df_simple():
    """Two patients, numeric values spread enough to use Q bins (n_distinct > level_threshold)."""
    rows = []
    for pid, base_nv, base_dt in [(1, 100.0, datetime(2020, 1, 1)),
                                   (2, 50.0, datetime(2020, 6, 1))]:
        for i in range(12):
            rows.append({
                "id": pid,
                "time": base_dt + timedelta(hours=i * 6),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": base_nv + i * 3.0,
            })
    return pl.DataFrame(rows)


def df_level():
    """Few distinct numeric values so level tokens are used."""
    rows = []
    for pid in [1, 2]:
        for i in range(8):
            rows.append({
                "id": pid,
                "time": datetime(2020, 1, 1) + timedelta(days=i),
                "class": "grade",
                "text_value": "score",
                "numeric_value": float((i % 3) + 1),
            })
    return pl.DataFrame(rows)


def df_no_numeric():
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


def df_bpe():
    """High-cardinality text → BPE, plus concept lab rows with numerics."""
    rows = []
    for i in range(80):
        rows.append({
            "id": 1,
            "time": datetime(2020, 1, 1) + timedelta(hours=i),
            "class": "note",
            "text_value": f"Patient presented with condition variant number {i} for extended observation",
            "numeric_value": None,
        })
    for i in range(12):
        rows.append({
            "id": 1,
            "time": datetime(2020, 1, 10) + timedelta(hours=i * 3),
            "class": "lab",
            "text_value": "creatinine",
            "numeric_value": 0.8 + i * 0.1,
        })
    return pl.DataFrame(rows)


def make_bpe_df():
    """DataFrame with high-cardinality notes (→BPE) + concept labs (from smoke tests)."""
    rows = []
    rows.append({
        "id": 1,
        "time": datetime(1980, 6, 15),
        "class": "demographic",
        "text_value": "birth_date",
        "numeric_value": None,
    })
    note_texts = [f"Patient presented with symptom variant {i}" for i in range(80)]
    for i, txt in enumerate(note_texts):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1, 8, 0) + timedelta(hours=i),
            "class": "notes",
            "text_value": txt,
            "numeric_value": None,
        })
    lab_texts = ["hemoglobin", "glucose", "creatinine"]
    for i, txt in enumerate(lab_texts * 5):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 5, 8, 0) + timedelta(hours=i),
            "class": "lab",
            "text_value": txt,
            "numeric_value": 10.0 + i * 0.5,
        })
    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Assertion helpers (decode tests)
# ═══════════════════════════════════════════════════════════════════════════════

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
    import math
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
    orig = df.sort(["id", "time"])["numeric_value"].to_list()
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
