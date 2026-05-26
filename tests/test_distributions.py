"""Tests for distribution selection, auto-detection, overrides, and negative-value handling."""
import polars as pl
import numpy as np
import warnings
import pytest
from datetime import datetime, timedelta
from conftest import make_df, REF
from tokenizer import DBTokenizer


def _df_with_negatives():
    """DataFrame where numeric values include negatives (forces normal distribution)."""
    rows = []
    for pid in [1, 2]:
        for i in range(15):
            rows.append({
                "id": pid,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "lab",
                "text_value": "temperature_delta",
                "numeric_value": -5.0 + i * 1.0,  # range: -5 to 9
            })
    return pl.DataFrame(rows)


def _df_positive_only():
    """DataFrame with all-positive numeric values (compatible with lognormal/gamma)."""
    rows = []
    for pid in [1, 2]:
        for i in range(15):
            rows.append({
                "id": pid,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "lab",
                "text_value": "crp",
                "numeric_value": 1.0 + i * 2.0,  # all positive
            })
    return pl.DataFrame(rows)


class TestDistributionAutoDetect:

    def test_continuous_fits_distribution(self):
        """Continuous mode fits distribution params for each (class, text_value) group."""
        df = make_df(include_birth=False)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            level_threshold=3,  # force scaling (make_df has >3 distinct values)
        )
        tok.train(df)

        # Should have numeric params for (lab, hemoglobin) and (lab, glucose)
        assert ("lab", "hemoglobin") in tok.numeric_params
        assert ("lab", "glucose") in tok.numeric_params
        hb_params = tok.numeric_params[("lab", "hemoglobin")]
        assert hb_params["type"] == "scaling"
        assert "distribution" in hb_params

    def test_auto_distribution_recommendation(self):
        """Distribution auto-detect stores recommendations for continuous mode."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored"
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        assert params["type"] == "scaling"
        # Should have picked a distribution
        assert params["distribution"] in ("normal", "lognormal", "gamma", "minmax")


class TestDistributionOverrides:

    def test_static_distribution_dict(self):
        """distributions dict forces specific distribution per class."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"lognormal": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        assert params["distribution"] == "lognormal"

    def test_gamma_distribution_override(self):
        """distributions dict can force gamma."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"gamma": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        assert params["distribution"] == "gamma"

    def test_minmax_distribution_override(self):
        """distributions dict can force minmax."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"minmax": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        assert params["distribution"] == "minmax"


class TestNegativeValueHandling:

    def test_negatives_force_normal(self):
        """Negative values force distribution to 'normal' even if lognormal requested."""
        df = _df_with_negatives()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"lognormal": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "temperature_delta")]
        assert params["distribution"] == "normal", (
            f"Negative values should force normal, got {params['distribution']}"
        )

    def test_negatives_force_normal_over_gamma(self):
        """Negative values force distribution to 'normal' when gamma is requested."""
        df = _df_with_negatives()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"gamma": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "temperature_delta")]
        assert params["distribution"] == "normal"


class TestScaling:

    def test_scale_unscale_roundtrip(self):
        """scale() followed by unscale() should approximately recover original value."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="continuous",
            num_seq="factored",
            distributions={"normal": ["lab"]}
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        original = 10.0
        scaled = tok.scale(params, original)
        recovered = tok.unscale(params, scaled)
        assert abs(recovered - original) < 1e-6, (
            f"scale/unscale roundtrip: {original} -> {scaled} -> {recovered}"
        )

    def test_discrete_mode_skips_distribution(self):
        """Discrete mode uses bin edges, not distribution scaling."""
        df = _df_positive_only()
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            level_threshold=3
        )
        tok.train(df)

        params = tok.numeric_params[("lab", "crp")]
        assert params["type"] == "bins"
        assert "edges" in params
        assert "distribution" not in params
