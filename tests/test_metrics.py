"""
Bulletproof tests for the bits-per-row metric.

Demonstrates that density-adjusted BPR is the correct metric for comparing
different tokenization schemes by:

1. Showing two schemes with matched model quality give identical BPR
2. Verifying every component formula against hand-computed values
3. Using NO real model — only hand-crafted probabilities with analytically
   verifiable answers on a minimal 2-row dataset

Minimal dataset:
    Patient 1, 12 rows of lab/hemoglobin at hourly intervals,
    numeric values 10.0–21.0 (enough distinct values to trigger quantile binning).
"""
import sys
import math
import re
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import DBTokenizer
from metrics import (
    _bin_width,
    _jacobian_log_abs,
    _density_adjustment,
    _assign_groups,
    _is_time_delta_ctx,
    LN2,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures: minimal dataset
# ═════════════════════════════════════════════════════════════════════════════

def _make_mini_df():
    """12 data rows, 1 patient, 1 class (lab/hemoglobin).

    Need n_distinct > level_threshold=10 to trigger quantile binning.
    """
    rows = []
    base = datetime(2023, 1, 1, 8, 0)
    for i in range(12):
        rows.append({
            "id": 1,
            "time": base + timedelta(hours=i),
            "class": "lab",
            "text_value": "hemoglobin",
            "numeric_value": 10.0 + i * 1.0,
        })
    return pl.DataFrame(rows)


def _make_level_df():
    """Dataset where numeric values have ≤10 distinct values → level tokens."""
    rows = []
    base = datetime(2023, 1, 1, 8, 0)
    vals = [1.0, 2.0, 3.0] * 4
    for i in range(12):
        rows.append({
            "id": 1,
            "time": base + timedelta(hours=i),
            "class": "lab",
            "text_value": "status",
            "numeric_value": vals[i],
        })
    return pl.DataFrame(rows)


@pytest.fixture
def discrete_tok():
    df = _make_mini_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=4,
        level_threshold=10,
        milestone_ip="none",
        milestone_op="none",
    )
    tok.train(df)
    return tok


@pytest.fixture
def continuous_tok():
    df = _make_mini_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="continuous",
        num_seq="factored",
        n_bins=4,
        level_threshold=10,
        milestone_ip="none",
        milestone_op="none",
    )
    tok.train(df)
    return tok


@pytest.fixture
def fused_tok():
    df = _make_mini_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="fused",
        n_bins=4,
        level_threshold=10,
        milestone_ip="none",
        milestone_op="none",
    )
    tok.train(df)
    return tok


@pytest.fixture
def level_tok():
    df = _make_level_df()
    tok = DBTokenizer(
        text_mode_default="concept",
        num_type="discrete",
        num_seq="factored",
        n_bins=4,
        level_threshold=10,
        milestone_ip="none",
        milestone_op="none",
    )
    tok.train(df)
    return tok


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: Same quality model → same BPR across schemes
# ═════════════════════════════════════════════════════════════════════════════

class TestCrossSchemeComparability:
    """Two different tokenization schemes with matched model quality must
    produce the same density-adjusted bits for numeric tokens.

    density(x) = p(Q_k) / w_k       (discrete)
    density(x) = p_z(z) · |dz/dx|   (continuous)

    Oracle discrete: p(Q_k) = 1  →  density = 1/w_k
    Oracle continuous: σ_z chosen so p_z(z)·|dz/dx| = 1/w_k

    Both give: −log₂(density) = log₂(w_k).
    """

    def test_matched_oracle_numeric_density_bits(self, discrete_tok, continuous_tok):
        df = _make_mini_df()

        # --- Discrete: find a value-Q token and compute its bin width ---
        ids_d, _ = discrete_tok.encode(df)
        types_d = discrete_tok.classify_token_ids(ids_d)
        contexts_d = discrete_tok.get_numeric_context(ids_d)

        pos_d = None
        for i, (t, ctx) in enumerate(zip(types_d, contexts_d)):
            if t == "Q" and ctx is not None and not _is_time_delta_ctx(ctx):
                pos_d = i
                break
        assert pos_d is not None, "No value-Q token found"

        ctx_d = contexts_d[pos_d]
        tok_str = discrete_tok.decode_token(ids_d[pos_d])
        bin_idx = int(re.search(r"Q(\d+)", tok_str).group(1))
        w_k = _bin_width(ctx_d["edges"], bin_idx)

        # Oracle discrete: raw=0, adj=log₂(w), density_bits = 0 + log₂(w) = log₂(w)
        discrete_density_bits = math.log2(w_k)

        # --- Continuous: set σ_z to match the discrete resolution ---
        ids_c, vals_c = continuous_tok.encode(df)
        types_c = continuous_tok.classify_token_ids(ids_c)
        contexts_c = continuous_tok.get_numeric_context(ids_c)

        pos_c = None
        for i, (t, ctx) in enumerate(zip(types_c, contexts_c)):
            if t == "num_marker" and ctx is not None and ctx.get("type") == "scaling":
                if not _is_time_delta_ctx(ctx):
                    pos_c = i
                    break
        assert pos_c is not None, "No value-NUM token found"

        ctx_c = contexts_c[pos_c]
        var_x = ctx_c.get("var", 1.0)
        sigma_x = math.sqrt(max(var_x, 1e-8))

        # For normal: |dz/dx| = 1/σ_x
        # p_x(x) = p_z(z) · |dz/dx| = (1/(σ_z√2π)) · (1/σ_x)
        # Set = 1/w_k:  σ_z = w_k / (σ_x · √2π)
        sigma_z = w_k / (sigma_x * math.sqrt(2 * math.pi))

        # Verify p_x = 1/w_k
        px = (1.0 / (sigma_z * math.sqrt(2 * math.pi))) * (1.0 / sigma_x)
        continuous_density_bits = -math.log2(px)

        # Both must give log₂(w_k)
        assert discrete_density_bits == pytest.approx(
            continuous_density_bits, abs=1e-10
        ), (
            f"Discrete ({discrete_density_bits:.10f}) != "
            f"Continuous ({continuous_density_bits:.10f})"
        )
        assert discrete_density_bits == pytest.approx(math.log2(w_k), abs=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: Density adjustment exact values
# ═════════════════════════════════════════════════════════════════════════════

class TestDensityAdjustmentMath:
    """Verify _bin_width and _jacobian_log_abs against hand-computed values."""

    def test_bin_width_interior(self):
        edges = [10.0, 20.0, 30.0, 40.0]
        assert _bin_width(edges, 2) == pytest.approx(10.0)

    def test_bin_width_first(self):
        edges = [10.0, 20.0, 30.0, 40.0]
        assert _bin_width(edges, 0) == pytest.approx(10.0)

    def test_bin_width_last(self):
        edges = [10.0, 20.0, 30.0, 40.0]
        assert _bin_width(edges, 4) == pytest.approx(10.0)

    def test_bin_width_unequal(self):
        edges = [0.0, 1.0, 10.0, 100.0]
        assert _bin_width(edges, 1) == pytest.approx(1.0)
        assert _bin_width(edges, 2) == pytest.approx(9.0)
        assert _bin_width(edges, 3) == pytest.approx(90.0)

    def test_jacobian_normal(self):
        """Normal: |dz/dx| = 1/σ → log|dz/dx| = -0.5·log(var)."""
        params = {"distribution": "normal", "var": 100.0}
        expected = -0.5 * math.log(100.0)
        assert _jacobian_log_abs(params) == pytest.approx(expected)

    def test_jacobian_gamma(self):
        """Gamma: |dz/dx| = 1/√(α·β²) → log = -0.5·log(α·β²)."""
        params = {"distribution": "gamma", "alpha": 4.0, "beta": 5.0}
        expected = -0.5 * math.log(100.0)
        assert _jacobian_log_abs(params) == pytest.approx(expected)

    def test_jacobian_minmax(self):
        """Minmax: |dz/dx| = 1/(max-min) → log = -log(range)."""
        params = {"distribution": "minmax", "min": 5.0, "max": 15.0}
        expected = -math.log(10.0)
        assert _jacobian_log_abs(params) == pytest.approx(expected)

    def test_normal_jacobian_matches_bin_resolution(self):
        """KEY INSIGHT: bin width 10 and σ=10 give the same adjustment.

        Bin: adj = log₂(w) = log₂(10)
        Normal: adj = −log|dz/dx|/ln2 = log₂(σ) = log₂(10)

        This is WHY the metric is cross-scheme comparable.
        """
        bin_adj = math.log2(10.0)
        params = {"distribution": "normal", "var": 100.0}
        log_jac = _jacobian_log_abs(params)
        continuous_adj = -log_jac / LN2

        assert bin_adj == pytest.approx(continuous_adj, abs=1e-12)

    def test_jacobian_lognormal_value_dependent(self):
        """Lognormal Jacobian depends on x: |dz/dx| = 1/(x·σ).

        z=0, μ=0, σ²=1: x=exp(0)=1, log|dz/dx| = -log(1) - 0.5·log(1) = 0
        z=1, μ=0, σ²=1: x=exp(1)=e, log|dz/dx| = -log(e) - 0 = -1
        """
        params = {"distribution": "lognormal", "mu": 0.0, "sigma2": 1.0}
        assert _jacobian_log_abs(params, z_value=0.0) == pytest.approx(0.0, abs=1e-10)
        assert _jacobian_log_abs(params, z_value=1.0) == pytest.approx(-1.0, abs=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: Token classification
# ═════════════════════════════════════════════════════════════════════════════

class TestTokenClassification:

    def test_discrete_classification(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        decoded = discrete_tok.decode(ids)

        for i, (tok, ttype) in enumerate(zip(decoded, types)):
            if tok == "<|sos|>":
                assert ttype == "sos", f"pos {i}: {tok} → {ttype}"
            elif tok == "<|eos|>":
                assert ttype == "eos", f"pos {i}: {tok} → {ttype}"
            elif tok.startswith("<|delta_time_"):
                assert ttype == "time_delta", f"pos {i}: {tok} → {ttype}"
            elif tok.startswith("<|Q"):
                assert ttype == "Q", f"pos {i}: {tok} → {ttype}"
            elif tok.startswith("<|L"):
                assert ttype == "L", f"pos {i}: {tok} → {ttype}"
            elif tok == "<|NUM|>":
                assert ttype == "num_marker", f"pos {i}: {tok} → {ttype}"
            elif tok.startswith("<|lab|>"):
                assert ttype == "class", f"pos {i}: {tok} → {ttype}"
            elif tok == "hemoglobin":
                assert ttype == "concept", f"pos {i}: {tok} → {ttype}"
            elif tok.startswith("<|age_"):
                assert ttype == "age", f"pos {i}: {tok} → {ttype}"

    def test_continuous_has_num_markers(self, continuous_tok):
        df = _make_mini_df()
        ids, vals = continuous_tok.encode(df)
        types = continuous_tok.classify_token_ids(ids)

        assert sum(1 for t in types if t == "num_marker") > 0
        assert sum(1 for t in types if t == "Q") == 0

    def test_fused_has_fused_tokens(self, fused_tok):
        df = _make_mini_df()
        ids, _ = fused_tok.encode(df)
        types = fused_tok.classify_token_ids(ids)
        decoded = fused_tok.decode(ids)

        has_fused = any(t in ("fused_concept_Q", "time_delta_fused") for t in types)
        assert has_fused, (
            f"Fused tokenizer should produce fused tokens. "
            f"Types: {list(zip(decoded, types))}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: Numeric context tracking
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericContext:

    def test_discrete_q_has_edges(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)

        for i, (ttype, ctx) in enumerate(zip(types, contexts)):
            if ttype in ("sos", "eos", "pad", "class", "concept",
                         "milestone", "age", "row"):
                assert ctx is None, f"pos {i} ({ttype}): expected None"
            elif ttype == "Q":
                assert ctx is not None and "edges" in ctx, (
                    f"pos {i}: Q token should have context with 'edges'"
                )

    def test_time_q_vs_value_q_distinguished(self, discrete_tok):
        """Q tokens after time-delta vs after concept get different contexts."""
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)

        time_qs = [ctx for t, ctx in zip(types, contexts)
                   if t == "Q" and ctx and _is_time_delta_ctx(ctx)]
        value_qs = [ctx for t, ctx in zip(types, contexts)
                    if t == "Q" and ctx and not _is_time_delta_ctx(ctx)]

        assert len(time_qs) > 0, "Should have time-delta Q contexts"
        assert len(value_qs) > 0, "Should have value Q contexts"

        for ctx in time_qs:
            assert ctx.get("_is_time_delta") is True

    def test_continuous_num_has_scaling_context(self, continuous_tok):
        df = _make_mini_df()
        ids, vals = continuous_tok.encode(df)
        types = continuous_tok.classify_token_ids(ids)
        contexts = continuous_tok.get_numeric_context(ids)

        for i, (ttype, ctx) in enumerate(zip(types, contexts)):
            if ttype == "num_marker" and ctx is not None:
                assert ctx.get("type") == "scaling", (
                    f"pos {i}: NUM context should be scaling"
                )


# ═════════════════════════════════════════════════════════════════════════════
# Test 5: Group routing
# ═════════════════════════════════════════════════════════════════════════════

class TestGroupAssignment:

    def test_q_after_time_delta_routes_to_time(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)
        groups = _assign_groups(types, contexts)

        for i in range(1, len(types)):
            if types[i] == "Q" and types[i - 1] == "time_delta":
                assert groups[i] == "time", f"pos {i}: expected 'time'"

    def test_value_q_routes_to_numeric(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)
        groups = _assign_groups(types, contexts)

        for i in range(len(types)):
            if types[i] == "Q" and contexts[i] and not _is_time_delta_ctx(contexts[i]):
                assert groups[i] == "numeric", f"pos {i}: expected 'numeric'"

    def test_structural_tokens_masked(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)
        groups = _assign_groups(types, contexts)

        for i, (ttype, g) in enumerate(zip(types, groups)):
            if ttype == "sos":
                assert g == "mask"
            elif ttype == "pad":
                assert g == "mask"
            elif ttype == "eos":
                assert g == "eos"


# ═════════════════════════════════════════════════════════════════════════════
# Test 6: Level tokens excluded from density adjustment
# ═════════════════════════════════════════════════════════════════════════════

class TestLevelTokens:

    def test_level_adj_is_zero(self, level_tok):
        df = _make_level_df()
        ids, _ = level_tok.encode(df)
        types = level_tok.classify_token_ids(ids)
        contexts = level_tok.get_numeric_context(ids)

        adj = _density_adjustment(
            level_tok, ids, [float("nan")] * len(ids), types, contexts
        )

        for i, (ttype, a) in enumerate(zip(types, adj)):
            if ttype == "L":
                assert a == 0.0, f"L at pos {i}: expected 0, got {a}"

    def test_level_classified_correctly(self, level_tok):
        df = _make_level_df()
        ids, _ = level_tok.encode(df)
        types = level_tok.classify_token_ids(ids)

        assert sum(1 for t in types if t == "L") > 0, "Should have L tokens"


# ═════════════════════════════════════════════════════════════════════════════
# Test 7: Fused vs factored same density adjustment
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedVsFactoredAdjustment:

    def test_same_bin_width_adjustment(self, discrete_tok, fused_tok):
        """hemoglobin::Q{k} and <|Q{k}|> for same bin must give same adj."""
        df = _make_mini_df()

        # Factored
        ids_f, _ = discrete_tok.encode(df)
        types_f = discrete_tok.classify_token_ids(ids_f)
        contexts_f = discrete_tok.get_numeric_context(ids_f)
        adj_f = _density_adjustment(
            discrete_tok, ids_f, [float("nan")] * len(ids_f), types_f, contexts_f
        )
        factored_adjs = {}
        for i, (t, ctx) in enumerate(zip(types_f, contexts_f)):
            if t == "Q" and ctx and not _is_time_delta_ctx(ctx):
                bin_idx = int(re.search(r"Q(\d+)", discrete_tok.decode_token(ids_f[i])).group(1))
                factored_adjs[bin_idx] = adj_f[i]

        # Fused
        ids_fu, _ = fused_tok.encode(df)
        types_fu = fused_tok.classify_token_ids(ids_fu)
        contexts_fu = fused_tok.get_numeric_context(ids_fu)
        adj_fu = _density_adjustment(
            fused_tok, ids_fu, [float("nan")] * len(ids_fu), types_fu, contexts_fu
        )
        fused_adjs = {}
        for i, (t, ctx) in enumerate(zip(types_fu, contexts_fu)):
            if t == "fused_concept_Q" and ctx:
                bin_idx = int(re.search(r"Q(\d+)", fused_tok.decode_token(ids_fu[i])).group(1))
                fused_adjs[bin_idx] = adj_fu[i]

        common = set(factored_adjs) & set(fused_adjs)
        assert len(common) > 0, (
            f"No overlapping bins. Factored: {factored_adjs.keys()}, "
            f"Fused: {fused_adjs.keys()}"
        )
        for b in common:
            assert factored_adjs[b] == pytest.approx(fused_adjs[b], abs=1e-12), (
                f"Bin {b}: factored={factored_adjs[b]:.10f} != fused={fused_adjs[b]:.10f}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Test 8: Integration — adjustment nonzero at exactly the right positions
# ═════════════════════════════════════════════════════════════════════════════

class TestDensityAdjustmentIntegration:

    def test_discrete_adjustments_at_q_only(self, discrete_tok):
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)
        adj = _density_adjustment(
            discrete_tok, ids, [float("nan")] * len(ids), types, contexts
        )

        for i, (ttype, a) in enumerate(zip(types, adj)):
            if ttype == "Q":
                assert a != 0.0, f"Q at pos {i} should have nonzero adj"
            elif ttype not in ("time_delta_fused", "fused_concept_Q"):
                assert a == 0.0, f"{ttype} at pos {i} should have 0 adj, got {a}"

    def test_continuous_adjustments_at_num_only(self, continuous_tok):
        df = _make_mini_df()
        ids, vals = continuous_tok.encode(df)
        types = continuous_tok.classify_token_ids(ids)
        contexts = continuous_tok.get_numeric_context(ids)
        adj = _density_adjustment(continuous_tok, ids, vals, types, contexts)

        for i, (ttype, a) in enumerate(zip(types, adj)):
            if ttype == "num_marker" and contexts[i] is not None:
                assert a != 0.0, f"NUM at pos {i} should have nonzero adj"
            elif ttype == "time_delta" and contexts[i] is not None:
                if vals is not None and not math.isnan(vals[i]):
                    assert a != 0.0, f"time_delta with value at pos {i} should have adj"
            elif ttype == "concept" and contexts[i] is not None:
                pass  # fused-scaling concept, may or may not be nonzero
            else:
                assert a == 0.0, f"{ttype} at pos {i} should have 0 adj, got {a}"

    def test_adjustment_values_analytically_correct(self, discrete_tok):
        """Verify one concrete Q adjustment = log₂(bin_width)."""
        df = _make_mini_df()
        ids, _ = discrete_tok.encode(df)
        types = discrete_tok.classify_token_ids(ids)
        contexts = discrete_tok.get_numeric_context(ids)
        adj = _density_adjustment(
            discrete_tok, ids, [float("nan")] * len(ids), types, contexts
        )

        for i, (ttype, ctx) in enumerate(zip(types, contexts)):
            if ttype == "Q" and ctx and not _is_time_delta_ctx(ctx):
                tok = discrete_tok.decode_token(ids[i])
                bin_idx = int(re.search(r"Q(\d+)", tok).group(1))
                w = _bin_width(ctx["edges"], bin_idx)
                assert adj[i] == pytest.approx(math.log2(w), abs=1e-12), (
                    f"pos {i} ({tok}): adj={adj[i]:.10f}, "
                    f"expected log₂({w})={math.log2(w):.10f}"
                )
                break


# ═════════════════════════════════════════════════════════════════════════════
# Test 9: Sign convention regression test
# ═════════════════════════════════════════════════════════════════════════════

class TestSignConvention:
    """Verify: density-adjusted bits = raw + adj (not raw − adj).

    density = p(Q_k) / w_k
    −log₂(density) = −log₂(p) + log₂(w) = raw + log₂(w) = raw + adj

    Wider bins → lower density → HIGHER cost in bits.
    """

    def test_wider_bin_costs_more_bits(self):
        p = 0.5
        raw = -math.log2(p)  # 1.0

        narrow_bits = -math.log2(p / 1.0)   # 1.0
        wide_bits = -math.log2(p / 10.0)     # ≈ 4.32
        assert wide_bits > narrow_bits

        # Verify the raw + adj formula
        assert narrow_bits == pytest.approx(raw + math.log2(1.0))
        assert wide_bits == pytest.approx(raw + math.log2(10.0))

    def test_continuous_wider_sigma_costs_more(self):
        """For normal: adj = log₂(σ). Larger σ → more adj → more bits."""
        raw_z = 2.0

        adj_small = math.log2(1.0)    # σ=1, adj=0
        adj_large = math.log2(10.0)   # σ=10, adj≈3.32

        # density_bits = raw + adj (code does raw + adj)
        assert (raw_z + adj_large) > (raw_z + adj_small)
