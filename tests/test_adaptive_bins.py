"""Adaptive unique quantile edges for discrete numeric / time-delta bins."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dbtoken.tokenizer import DBTokenizer, _adaptive_quantile_edges


def _occupancy(vals, edges):
    ids = np.digitize(vals, edges, right=False)
    return np.bincount(ids, minlength=len(edges) + 1)


def test_edges_strictly_increasing():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(900), rng.uniform(1, 100, 100)])
    edges = _adaptive_quantile_edges(x, n_bins=20)
    assert len(edges) == 19
    assert np.all(np.diff(edges) > 0)


def test_point_mass_reallocates_collapsed_cuts():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(1800), rng.uniform(1, 100, 200)])
    raw = np.quantile(x, np.linspace(0.01, 0.99, 19))
    assert len(np.unique(np.round(raw, 12))) < 19

    edges = _adaptive_quantile_edges(x, n_bins=20)
    occ = _occupancy(x, edges)
    assert len(edges) == 19
    assert int((occ > 0).sum()) == 20
    assert int((occ == 0).sum()) == 0
    # The zero atom is one bin, not ~18 collapsed copies.
    assert occ[0] == 1800


def test_well_spread_keeps_equal_mass():
    rng = np.random.default_rng(0)
    x = rng.normal(13.5, 1.2, 2000)
    edges = _adaptive_quantile_edges(x, n_bins=20)
    occ = _occupancy(x, edges)
    interior = occ[1:-1]
    assert len(edges) == 19
    assert int((occ > 0).sum()) == 20
    # Interior bins stay near-equal (clip tails are smaller by construction).
    assert interior.max() - interior.min() <= 5


def test_cannot_exceed_unique_value_gaps():
    x = np.concatenate([
        np.full(700, 5.0),
        np.full(200, 10.0),
        np.full(50, 20.0),
        np.full(20, 7.5),
        np.full(15, 15.0),
        np.full(10, 25.0),
        np.full(5, 30.0),
    ])
    edges = _adaptive_quantile_edges(x, n_bins=15)
    u = np.unique(x[(x >= np.percentile(x, 1)) & (x <= np.percentile(x, 99))])
    assert len(edges) == len(u) - 1
    assert np.all(np.diff(edges) > 0)


def test_identical_values_yield_no_edges():
    edges = _adaptive_quantile_edges(np.ones(50), n_bins=10)
    assert edges == []


def test_tokenizer_clustered_group_uses_all_bins():
    rng = np.random.default_rng(1)
    rows = []
    zeros = 80
    tail = rng.uniform(1, 50, 40)
    vals = np.concatenate([np.zeros(zeros), tail])
    for i, v in enumerate(vals):
        rows.append({
            "id": 1,
            "time": datetime(2023, 1, 1) + timedelta(hours=i),
            "class": "lab",
            "text_value": "spike",
            "numeric_value": float(v),
        })
    df = pl.DataFrame(rows)
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        n_bins=20,
        level_threshold=10,
    )
    tok.train(df)
    params = tok.numeric_params[("lab", "spike")]
    edges = params["edges"]
    assert params["type"] == "bins"
    assert len(edges) == 19
    assert len(set(edges)) == 19
    assert np.all(np.diff(edges) > 0)

    ids, _ = tok.encode(df)
    decoded = tok.decode(ids)
    q_ids = sorted({int(t[3:-2]) for t in decoded if t.startswith("<|Q")})
    assert q_ids[0] == 0
    assert len(q_ids) == 20


def test_time_delta_bins_are_strictly_increasing():
    rows = []
    t0 = datetime(2023, 1, 1, 8, 0)
    # Many 1-hour gaps (clustered) plus a spread of longer gaps.
    hours = [1] * 40 + list(range(2, 30))
    t = t0
    rows.append({
        "id": 1, "time": t, "class": "lab",
        "text_value": "hr", "numeric_value": 70.0,
    })
    for i, h in enumerate(hours):
        t = t + timedelta(hours=h)
        rows.append({
            "id": 1, "time": t, "class": "lab",
            "text_value": "hr", "numeric_value": 70.0 + (i % 12),
        })
    df = pl.DataFrame(rows)
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        n_bins=15,
        level_threshold=3,
    )
    tok.train(df)
    td = next(iter(tok.time_delta_params.values()))
    edges = td["edges"]
    assert td["type"] == "bins"
    assert len(edges) >= 2
    assert np.all(np.diff(edges) > 0)
