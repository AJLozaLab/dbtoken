"""
DBTokenizer — tokenizer for long-format database/EHR tabular data.

Tokenizes tables with schema: id, time, class, text_value, numeric_value

Supports:
  - Per-class (or per-(class, text_value)) text tokenization: BPE or concept
  - Numeric tokenization: discrete (quantile bins), continuous (distribution
    scaling), or categorical levels (≤ threshold distinct values → L tokens)
  - Fused or factored numeric sequence placement (fused auto-enforced only
    for concept-mode groups; BPE groups silently fall back to factored)
  - Inpatient / outpatient-aware time-delta tokens
  - Configurable milestone frequencies (week, month, daily, 12hr, 8hr, none)
    with Kalman-filter style emission (delta + most-recent milestone only)
  - Age milestones derived from demographic birth_date rows in data
  - JSON-based save / load

Two-stage workflow:
  1. train(df)   — learn vocab, numeric transforms, time-delta fits
  2. encode(df)  — produce (ids, vals) token sequences
"""

import polars as pl
import numpy as np
import math
import json
import warnings
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# BPE: rustbpe (training) + tiktoken (inference)
# ---------------------------------------------------------------------------
try:
    import rustbpe
    import tiktoken
    HAS_BPE_BACKEND = True
except ImportError:
    HAS_BPE_BACKEND = False

try:
    from dbtoken.distribution_utils import analyze_distributions
except ImportError:
    analyze_distributions = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_MILESTONE_TYPES = ("week", "month", "daily", "12hr", "8hr", "none")
_MAX_AGE = 120
_MAX_WEEK = 53
_MAX_MONTH = 12
_MAX_SHIFT_12 = 2   # 0 or 1 (two 12-hr shifts per day)
_MAX_SHIFT_8 = 3    # 0, 1, or 2 (three 8-hr shifts per day)
_MAX_DAY_OF_WEEK = 7


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SCALING_FORMULAS = {
    "normal": {
        "scale": lambda p, x: (x - p["mean"]) / math.sqrt(max(p["var"], 1e-8)),
        "unscale": lambda p, x: x * math.sqrt(max(p["var"], 1e-8)) + p["mean"],
    },
    "lognormal": {
        "scale": lambda p, x: (
            (math.log(max(x, 1e-8)) - p["mu"])
            / math.sqrt(max(p["sigma2"], 1e-8))
        ),
        "unscale": lambda p, x: math.exp(
            x * math.sqrt(max(p["sigma2"], 1e-8)) + p["mu"]
        ),
    },
    "gamma": {
        "scale": lambda p, x: (
            (x - p["alpha"] * p["beta"])
            / math.sqrt(max(p["alpha"] * p["beta"] ** 2, 1e-8))
        ),
        "unscale": lambda p, x: (
            x * math.sqrt(max(p["alpha"] * p["beta"] ** 2, 1e-8))
            + p["alpha"] * p["beta"]
        ),
    },
    "minmax": {
        "scale": lambda p, x: (x - p["min"]) / max(p["max"] - p["min"], 1e-8),
        "unscale": lambda p, x: x * max(p["max"] - p["min"], 1e-8) + p["min"],
    },
}


_FIT_PARAM_FUNCS = {
    "normal": lambda v: {"mean": float(np.mean(v)), "var": float(np.var(v))},
    "lognormal": lambda v: {
        "mu": float(np.mean(np.log(v[v > 0]))) if (v > 0).any() else 0.0,
        "sigma2": float(np.var(np.log(v[v > 0]))) if (v > 0).any() else 1.0,
    },
    "gamma": lambda v: {
        "alpha": float(np.mean(v) ** 2 / max(np.var(v), 1e-8)),
        "beta": float(np.var(v) / max(np.mean(v), 1e-8)),
    },
    "minmax": lambda v: {"min": float(np.min(v)), "max": float(np.max(v))},
}


def _fit_params(distribution: str, vals: np.ndarray) -> dict:
    """Compute moment-matched distribution parameters from a numpy array."""
    return _FIT_PARAM_FUNCS[distribution](vals)


def _milestone_marker(kind: str, ts: datetime, shift_start: int) -> Optional[str]:
    """Return the milestone-position token string for *ts* under *kind*.

    Returns None for kind='none'.  For the others:
      - week   → <|ms_week_{iso_week}|>  (1..53)
      - month  → <|ms_month_{month}|>    (1..12)
      - daily  → <|ms_day_{dow}|>        (1..7, Mon=1)
      - 12hr   → <|ms_12hr_{shift}|>     (0 or 1)
      - 8hr    → <|ms_8hr_{shift}|>      (0, 1, or 2)
    """
    if kind == "none":
        return None
    if kind == "week":
        w = ts.isocalendar()[1]
        return f"<|ms_week_{w}|>"
    if kind == "month":
        return f"<|ms_month_{ts.month}|>"
    if kind == "daily":
        return f"<|ms_day_{ts.isoweekday()}|>"
    if kind == "12hr":
        shift = (ts.hour - shift_start) % 24 // 12
        return f"<|ms_12hr_{shift}|>"
    if kind == "8hr":
        shift = (ts.hour - shift_start) % 24 // 8
        return f"<|ms_8hr_{shift}|>"
    return None


def _milestone_boundary(kind: str, ts: datetime, shift_start: int):
    """Return a comparable boundary key for *ts* under milestone *kind*.

    Two timestamps share a milestone epoch iff they return the same key.
    """
    if kind == "none":
        return None
    if kind == "week":
        return (ts.isocalendar()[0], ts.isocalendar()[1])
    if kind == "month":
        return (ts.year, ts.month)
    if kind == "daily":
        return ts.date()
    if kind == "12hr":
        h = (ts.hour - shift_start) % 24
        return (ts.date(), h // 12)
    if kind == "8hr":
        h = (ts.hour - shift_start) % 24
        return (ts.date(), h // 8)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DBTokenizer
# ──────────────────────────────────────────────────────────────────────────────

class DBTokenizer:
    """
    Tokenizer for long-format database tables with schema:
        id, time, class, text_value, numeric_value

    Two-stage workflow:
        1. train(df)   — learn vocabulary, numeric transforms, time-delta fits
        2. encode(df)  — produce (ids, vals) token sequences
    """

    # ─── Constructor ──────────────────────────────────────────────────────

    def __init__(
        self,
        # --- text tokenization ---
        text_mode_default: str = "auto",          # 'auto', 'bpe', 'concept'
        text_mode_overrides: Optional[Dict] = None,
        #   text_mode_overrides accepts:
        #     str  key → per-class:          {'lab': 'concept'}
        #     tuple key → per-(class,tv):    {('lab','hemoglobin'): 'concept'}
        text_mode_threshold: int = 64,
        # --- numeric tokenization ---
        num_type: str = "discrete",               # 'discrete' or 'continuous'
        num_seq: str = "factored",                # 'fused' or 'factored'
        n_bins: int = 10,
        level_threshold: int = 10,
        bin_clip_min: float = 1.0,
        bin_clip_max: float = 99.0,
        continuous_clip_min: float = 1.0,
        continuous_clip_max: float = 99.0,
        distributions: Optional[Dict[str, List[str]]] = None,
        # --- BPE settings ---
        final_vocab_size: int = 4096,
        split_pattern: str = (
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+"""
            r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
            r"""|\s+(?!\S)|\s+"""
        ),
        # --- temporal scales (time-delta distributions) ---
        time_scales: Optional[List[float]] = None,   # sorted thresholds, e.g. [86400.0]
        time_scale_names: Optional[List[str]] = None, # optional labels per band
        # --- state machine (milestones) ---
        state_transitions: Optional[Dict[str, List[str]]] = None,
        initial_state: str = "default",
        milestone_per_state: Optional[Dict[str, str]] = None,
        milestone_shift_start: int = 7,           # hour-of-day for shift start
        # --- demographics ---
        birth_date_class: str = "demographic",
        birth_date_text_value: str = "birth_date",
    ):
        # Validate
        if text_mode_default not in ("auto", "bpe", "concept"):
            raise ValueError(
                f"text_mode_default must be 'auto'|'bpe'|'concept', "
                f"got '{text_mode_default}'"
            )
        if num_type not in ("discrete", "continuous"):
            raise ValueError(f"num_type must be 'discrete'|'continuous', got '{num_type}'")
        if num_seq not in ("fused", "factored"):
            raise ValueError(f"num_seq must be 'fused'|'factored', got '{num_seq}'")
        # Validate time_scales / time_scale_names
        if time_scales is not None:
            if sorted(time_scales) != list(time_scales):
                raise ValueError("time_scales must be a sorted (ascending) list of floats")
            if any(t <= 0 for t in time_scales):
                raise ValueError("time_scales values must be positive")
        if time_scale_names is not None:
            n_bands = (len(time_scales) + 1) if time_scales else 1
            if len(time_scale_names) != n_bands:
                raise ValueError(
                    f"time_scale_names must have length len(time_scales)+1 "
                    f"(= {n_bands}), got {len(time_scale_names)}"
                )
        # Validate milestone_per_state values
        if milestone_per_state is not None:
            for state, ms in milestone_per_state.items():
                if ms not in _MILESTONE_TYPES:
                    raise ValueError(
                        f"milestone_per_state[{state!r}] must be one of "
                        f"{_MILESTONE_TYPES}, got {ms!r}"
                    )

        # Store config
        self.text_mode_default = text_mode_default
        self.text_mode_overrides: Dict = text_mode_overrides or {}
        self.text_mode_threshold = text_mode_threshold
        self.num_type = num_type
        self.num_seq = num_seq
        self.n_bins = n_bins
        self.level_threshold = level_threshold
        self.bin_clip_min = bin_clip_min
        self.bin_clip_max = bin_clip_max
        self.continuous_clip_min = continuous_clip_min
        self.continuous_clip_max = continuous_clip_max
        self.distributions = distributions
        self.final_vocab_size = final_vocab_size
        self.split_pattern = split_pattern
        self.time_scales: List[float] = list(time_scales) if time_scales else []
        self.time_scale_names: Optional[List[str]] = time_scale_names
        self.state_transitions: Dict[str, List[str]] = state_transitions or {}
        self.initial_state: str = initial_state
        self.milestone_per_state: Dict[str, str] = milestone_per_state or {}
        self.milestone_shift_start = milestone_shift_start
        self.birth_date_class = birth_date_class
        self.birth_date_text_value = birth_date_text_value

        # Learned state (populated by train)
        self.class_text_modes: Dict[str, str] = {}
        self.pair_text_modes: Dict[Tuple[str, str], str] = {}
        self.numeric_params: Dict[Tuple[str, Optional[str]], dict] = {}
        self.time_delta_params: Dict[str, dict] = {}
        self.vocab: Dict[str, int] = {}
        self.ivocab: Dict[int, str] = {}
        self.bpe: Any = None
        self.bpe_cache: List[List[int]] = []
        self.has_bpe_classes: bool = False
        self._dist_recommendations: Dict[str, str] = {}
        self._trained: bool = False
        self._birth_dates: Dict = {}   # id → datetime, extracted from data

    # ─── Schema validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_schema(df: pl.DataFrame):
        required = {"id", "time", "class", "text_value", "numeric_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(self, df: pl.DataFrame):
        """
        Learn vocabulary, numeric transforms, and time-delta fits from *df*.

        Birth dates are extracted automatically from rows where
        class == ``birth_date_class`` and text_value == ``birth_date_text_value``
        (the ``time`` column of those rows holds the birth date).
        """
        self._validate_schema(df)

        print("Extracting birth dates from data...")
        self._extract_birth_dates(df)

        # Filter out demographic-birth-date rows for downstream training
        df_clinical = self._filter_clinical(df)

        print("Resolving per-class text modes...")
        self._resolve_text_modes(df_clinical)

        print("Training numeric transforms...")
        self._train_numeric(df_clinical)

        print("Training time-delta transforms...")
        self._train_time_deltas(df_clinical)

        print("Building vocabulary...")
        self._train_vocab(df_clinical)

        self._trained = True
        print("Training complete.")

    # ·····  birth-date extraction  ····································

    def _extract_birth_dates(self, df: pl.DataFrame):
        """Extract birth dates from demographic rows in the data."""
        bd_rows = df.filter(
            (pl.col("class") == self.birth_date_class)
            & (pl.col("text_value") == self.birth_date_text_value)
        )
        if bd_rows.height > 0:
            for row in bd_rows.iter_rows(named=True):
                self._birth_dates[row["id"]] = row["time"]
            print(f"  Found birth dates for {len(self._birth_dates)} patients")
        else:
            print("  No birth-date rows found")

    def _filter_clinical(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove demographic birth-date rows so they don't pollute training."""
        return df.filter(
            ~(
                (pl.col("class") == self.birth_date_class)
                & (pl.col("text_value") == self.birth_date_text_value)
            )
        )

    # ·····  text-mode resolution  ·····································

    def _resolve_text_modes(self, df: pl.DataFrame):
        """Determine BPE vs concept text tokenization per class
        and per (class, text_value) pair."""

        # --- Separate class-level and pair-level overrides ---
        class_overrides = {}
        pair_overrides = {}
        for k, v in self.text_mode_overrides.items():
            if isinstance(k, tuple):
                pair_overrides[k] = v
            else:
                class_overrides[k] = v

        # --- Per-class resolution ---
        class_stats = (
            df.group_by("class")
            .agg(pl.col("text_value").n_unique().alias("n_unique_text"))
        )
        for row in class_stats.iter_rows(named=True):
            cls = row["class"]
            if cls in class_overrides:
                mode = class_overrides[cls]
            elif self.text_mode_default in ("bpe", "concept"):
                mode = self.text_mode_default
            else:  # auto
                mode = (
                    "bpe"
                    if row["n_unique_text"] > self.text_mode_threshold
                    else "concept"
                )
            self.class_text_modes[cls] = mode

        # --- Per-(class, text_value) overrides ---
        self.pair_text_modes = dict(pair_overrides)

        self.has_bpe_classes = any(m == "bpe" for m in self.class_text_modes.values())
        # Also check if any pair overrides introduce BPE
        if not self.has_bpe_classes:
            self.has_bpe_classes = any(
                m == "bpe" for m in self.pair_text_modes.values()
            )

        n_bpe = sum(1 for m in self.class_text_modes.values() if m == "bpe")
        n_concept = sum(1 for m in self.class_text_modes.values() if m == "concept")
        n_pair = len(self.pair_text_modes)
        print(f"  {n_concept} concept classes, {n_bpe} BPE classes, "
              f"{n_pair} pair-level overrides")

        if self.num_seq == "fused" and self.has_bpe_classes:
            bpe_cls = [c for c, m in self.class_text_modes.items() if m == "bpe"]
            warnings.warn(
                f"num_seq='fused' but {len(bpe_cls)} classes use BPE text mode "
                f"(e.g. {bpe_cls[:5]}). These will silently use factored numeric "
                f"placement unless overridden at pair level."
            )

    def _get_text_mode(self, cls: str, tv: Optional[str]) -> str:
        """Resolve the effective text mode for a (class, text_value) pair."""
        if tv is not None and (cls, tv) in self.pair_text_modes:
            return self.pair_text_modes[(cls, tv)]
        return self.class_text_modes.get(cls, "concept")

    # ·····  numeric training  ·········································

    def _train_numeric(self, df: pl.DataFrame):
        """Learn numeric transforms per (class, text_value) group."""
        num_df = df.filter(
            pl.col("numeric_value").is_not_null()
            & (~pl.col("numeric_value").is_nan())
        )
        if num_df.height == 0:
            print("  No numeric values found.")
            return

        # ── Auto-detect best distributions (continuous mode) ──────────
        if self.num_type == "continuous" and analyze_distributions is not None:
            auto_df = num_df.with_columns(
                (
                    pl.col("class") + "||" + pl.col("text_value").fill_null("")
                ).alias("_group_key")
            )
            try:
                analysis = analyze_distributions(
                    auto_df,
                    value_col="numeric_value",
                    group_col="_group_key",
                    clip_min=self.continuous_clip_min,
                    clip_max=self.continuous_clip_max,
                )
                self._dist_recommendations = {
                    row["_group_key"]: row["recommendation"]
                    for row in analysis.iter_rows(named=True)
                }
            except Exception as e:
                warnings.warn(
                    f"Distribution auto-detect failed: {e}. "
                    f"Falling back to normal."
                )

        # ── Per-group numeric parameter fitting ───────────────────────
        group_stats = (
            num_df.group_by(["class", "text_value"])
            .agg([
                pl.col("numeric_value").n_unique().alias("n_distinct"),
                pl.col("numeric_value").count().alias("count"),
            ])
        )

        n_level = n_bins = n_scaling = 0

        for row in group_stats.iter_rows(named=True):
            key = (row["class"], row["text_value"])
            n_distinct = row["n_distinct"]

            tv_filter = (
                (pl.col("text_value") == row["text_value"])
                if row["text_value"] is not None
                else pl.col("text_value").is_null()
            )

            if n_distinct <= self.level_threshold:
                sorted_vals = (
                    num_df
                    .filter((pl.col("class") == row["class"]) & tv_filter)
                    .select("numeric_value")
                    .unique()
                    .sort("numeric_value")
                    .to_series()
                    .to_list()
                )
                self.numeric_params[key] = {
                    "type": "level",
                    "values": sorted_vals,
                    "value_to_idx": {v: i for i, v in enumerate(sorted_vals)},
                }
                n_level += 1

            elif self.num_type == "discrete":
                group_vals = (
                    num_df
                    .filter((pl.col("class") == row["class"]) & tv_filter)
                    .select("numeric_value")
                    .to_series()
                )
                p_edges = list(
                    np.linspace(
                        self.bin_clip_min / 100,
                        self.bin_clip_max / 100,
                        self.n_bins - 1,
                    )
                )
                edges = [float(group_vals.quantile(p)) for p in p_edges]
                lo = float(group_vals.quantile(self.bin_clip_min / 100))
                hi = float(group_vals.quantile(self.bin_clip_max / 100))
                self.numeric_params[key] = {
                    "type": "bins", "edges": edges,
                    "min_threshold": lo, "max_threshold": hi,
                }
                n_bins += 1

            else:  # continuous scaling
                group_vals = (
                    num_df
                    .filter((pl.col("class") == row["class"]) & tv_filter)
                    .select("numeric_value")
                    .to_series()
                    .to_numpy()
                    .astype(float)
                )
                params = self._fit_distribution(
                    group_vals, key,
                    self.continuous_clip_min, self.continuous_clip_max,
                )
                self.numeric_params[key] = {"type": "scaling", **params}
                n_scaling += 1

        print(
            f"  {len(self.numeric_params)} numeric groups: "
            f"{n_level} level, {n_bins} bins, {n_scaling} scaling"
        )

    def _fit_distribution(
        self, vals: np.ndarray, key: Tuple[str, Optional[str]],
        clip_min: float, clip_max: float,
    ) -> dict:
        """Fit the best distribution to a numpy array of values."""
        lo = float(np.percentile(vals, clip_min))
        hi = float(np.percentile(vals, clip_max))
        vals = np.clip(vals, lo, hi)
        has_negatives = bool((vals < 0).any())

        distribution = None
        if self.distributions is not None:
            for dist_name, class_list in self.distributions.items():
                if key[0] in class_list:
                    distribution = dist_name
                    break
        if distribution is None:
            group_key = f"{key[0]}||{key[1] or ''}"
            distribution = self._dist_recommendations.get(group_key, "normal")
        if has_negatives and distribution in ("lognormal", "gamma"):
            distribution = "normal"

        params = _fit_params(distribution, vals)
        for k, v in params.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                params[k] = 1.0
                warnings.warn(
                    f"Group {key}: parameter '{k}' was NaN, defaulting to 1.0"
                )

        return {
            "distribution": distribution,
            "min_threshold": lo,
            "max_threshold": hi,
            **params,
        }

    # ·····  time-delta training  ······································

    def _train_time_deltas(self, df: pl.DataFrame):
        """Fit time-delta distributions/bins per temporal scale band."""
        with_deltas = (
            df
            .sort(["id", "time"])
            .with_columns(
                ((pl.col("time") - pl.col("time").shift(1)).over("id"))
                .dt.total_seconds()
                .alias("time_delta_s")
            )
            .filter(
                pl.col("time_delta_s").is_not_null()
                & (pl.col("time_delta_s") > 0)
            )
        )
        if with_deltas.height == 0:
            warnings.warn(
                "No positive time deltas found; skipping time-delta training."
            )
            return

        all_deltas = with_deltas.select("time_delta_s").to_series()
        scale_keys = self._scale_keys()
        thresholds = self.time_scales  # sorted list of floats, possibly empty

        if not thresholds:
            # Single global distribution
            self.time_delta_params[scale_keys[0]] = self._fit_time_delta_series(all_deltas)
        else:
            global_p = None  # computed lazily if a band is too sparse
            lo = 0.0
            for i, sk in enumerate(scale_keys):
                hi = thresholds[i] if i < len(thresholds) else float("inf")
                band = (
                    with_deltas
                    .filter(
                        (pl.col("time_delta_s") > lo)
                        & (pl.col("time_delta_s") <= hi if hi != float("inf")
                           else pl.lit(True))
                    )
                    .select("time_delta_s")
                    .to_series()
                )
                if band.len() > 1:
                    self.time_delta_params[sk] = self._fit_time_delta_series(band)
                else:
                    # Fallback to global params for sparse bands
                    if global_p is None:
                        global_p = self._fit_time_delta_series(all_deltas)
                    self.time_delta_params[sk] = global_p
                lo = hi

        for sk in scale_keys:
            p = self.time_delta_params.get(sk, {})
            print(f"  scale '{sk}' time-delta: {p.get('type', 'N/A')} "
                  f"({p.get('distribution', '-')})")


    def _fit_time_delta_series(self, series: pl.Series) -> dict:
        vals = series.to_numpy().astype(float)
        if self.num_type == "continuous":
            clip_lo, clip_hi = self.continuous_clip_min, self.continuous_clip_max
        else:
            clip_lo, clip_hi = self.bin_clip_min, self.bin_clip_max

        lo = float(np.percentile(vals, clip_lo))
        hi = float(np.percentile(vals, clip_hi))
        vals_clipped = np.clip(vals, lo, hi)

        if self.num_type == "continuous":
            params = _fit_params("lognormal", vals_clipped)
            return {
                "type": "scaling",
                "distribution": "lognormal",
                "min_threshold": lo,
                "max_threshold": hi,
                **params,
            }
        else:
            p_edges = list(
                np.linspace(clip_lo / 100, clip_hi / 100, self.n_bins - 1)
            )
            edges = [float(np.percentile(vals, p * 100)) for p in p_edges]
            return {
                "type": "bins", "edges": edges,
                "min_threshold": lo, "max_threshold": hi,
            }

    # ·····  delta scale classification  ······························

    def _classify_delta_scale(self, dt: float) -> str:
        """Return the scale key for a given time delta *dt* (seconds).

        Bands are defined by ``self.time_scales`` (sorted thresholds).
        Band 0: dt <= time_scales[0]
        Band i: time_scales[i-1] < dt <= time_scales[i]
        Band N: dt > time_scales[-1]
        """
        keys = self._scale_keys()
        for i, threshold in enumerate(self.time_scales):
            if dt <= threshold:
                return keys[i]
        return keys[-1]

    # ·····  scale keys helper  ········································

    def _scale_keys(self) -> List[str]:
        """Return the ordered list of time-scale band key strings.

        If ``time_scale_names`` is set, those names are returned directly.
        Otherwise index strings '0', '1', ... are generated, one per band
        (number of bands = len(time_scales) + 1, minimum 1).
        """
        n_bands = len(self.time_scales) + 1 if self.time_scales else 1
        if self.time_scale_names is not None:
            return list(self.time_scale_names)
        return [str(i) for i in range(n_bands)]

    # ·····  vocabulary / BPE training  ································

    def _train_vocab(self, df: pl.DataFrame):
        """Build unified token vocabulary."""

        # ── Standard special tokens ───────────────────────────────────
        special: List[str] = ["<|sos|>", "<|eos|>", "<|pad|>"]
        for sk in self._scale_keys():
            special.append(f"<|delta_time_{sk}|>")
        special.append("<|NUM|>")

        for i in range(self.n_bins + 1):
            special.append(f"<|Q{i}|>")
        for i in range(self.level_threshold + 1):
            special.append(f"<|L{i}|>")

        # ── Milestone tokens (parameterized) ─────────────────────────
        for i in range(1, _MAX_WEEK + 1):
            special.append(f"<|ms_week_{i}|>")
        for i in range(1, _MAX_MONTH + 1):
            special.append(f"<|ms_month_{i}|>")
        for i in range(1, _MAX_DAY_OF_WEEK + 1):
            special.append(f"<|ms_day_{i}|>")
        for i in range(_MAX_SHIFT_12):
            special.append(f"<|ms_12hr_{i}|>")
        for i in range(_MAX_SHIFT_8):
            special.append(f"<|ms_8hr_{i}|>")

        # ── Age tokens ───────────────────────────────────────────────
        for i in range(_MAX_AGE + 1):
            special.append(f"<|age_{i}|>")

        special.append("<|ROW|>")

        # ── Class tokens ─────────────────────────────────────────────
        unique_classes = df.select("class").unique().to_series().to_list()
        for cls in unique_classes:
            special.append(f"<|{cls}|>")

        # ── Fused time-delta tokens (fused + discrete) ───────────────
        if self.num_seq == "fused" and self.num_type == "discrete":
            for sk in self._scale_keys():
                for i in range(self.n_bins + 1):
                    special.append(f"<|delta_time_{sk}_Q{i}|>")

        # ── Concept text tokens ──────────────────────────────────────
        concept_tokens: List[str] = []
        concept_text_vals = set()
        for row in df.select(["class", "text_value"]).unique().iter_rows(named=True):
            cls = row["class"]
            tv = row["text_value"]
            if tv is None:
                continue
            mode = self._get_text_mode(cls, tv)
            if mode == "concept":
                concept_text_vals.add(tv)
        concept_tokens.extend(concept_text_vals)

        # Fused concept+numeric tokens
        if self.num_seq == "fused":
            for key, params in self.numeric_params.items():
                cls, tv = key
                if tv is None:
                    continue
                mode = self._get_text_mode(cls, tv)
                if mode != "concept":
                    continue
                if params["type"] == "level":
                    for i in range(len(params["values"])):
                        concept_tokens.append(f"{tv}::L{i}")
                elif params["type"] == "bins":
                    for i in range(self.n_bins + 1):
                        concept_tokens.append(f"{tv}::Q{i}")

        # Deduplicate preserving order
        all_special = list(dict.fromkeys(special + concept_tokens))

        # ── Build vocab ───────────────────────────────────────────────
        if self.has_bpe_classes:
            if not HAS_BPE_BACKEND:
                raise ImportError(
                    "BPE tokenizer not available.  Install rustbpe and "
                    "tiktoken to enable BPE text mode."
                )
            # Gather BPE training text — only from BPE-mode rows
            bpe_df = df.filter(
                pl.col("text_value").is_not_null()
            )
            # Only keep rows whose (class, text_value) resolved to BPE
            bpe_text_list = []
            for r in bpe_df.iter_rows(named=True):
                if self._get_text_mode(r["class"], r["text_value"]) == "bpe":
                    bpe_text_list.append(r["text_value"])

            # --- Train BPE merges with rustbpe ---
            n_special = len(all_special)
            bpe_vocab_size = max(self.final_vocab_size - n_special, 256)
            trainer = rustbpe.Tokenizer()
            if bpe_text_list:
                trainer.train_from_iterator(
                    iter(bpe_text_list), bpe_vocab_size,
                    pattern=self.split_pattern,
                )

            # --- Build tiktoken Encoding for inference ---
            mergeable_ranks = {
                bytes(k): v
                for k, v in trainer.get_mergeable_ranks()
            }
            pattern = trainer.get_pattern()
            bpe_base = len(mergeable_ranks)
            special_tokens_map = {
                name: bpe_base + i for i, name in enumerate(all_special)
            }
            self.bpe = tiktoken.Encoding(
                name="dbtokenizer",
                pat_str=pattern,
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens_map,
            )

            # Populate vocab / ivocab from special tokens
            self.vocab = {}
            self.ivocab = {}
            for tok_str, tok_id in special_tokens_map.items():
                self.vocab[tok_str] = tok_id
                self.ivocab[tok_id] = tok_str

            n_sp = len(self.vocab)
            print(
                f"  BPE vocab size: {self.bpe.n_vocab} "
                f"({n_sp} special/concept, {bpe_base} BPE merges)"
            )
        else:
            self.vocab = {tok: i for i, tok in enumerate(all_special)}
            self.ivocab = {i: tok for i, tok in enumerate(all_special)}
            self.final_vocab_size = len(self.vocab)
            print(f"  Concept vocab size: {self.final_vocab_size}")

    # ──────────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────────

    def encode(
        self,
        df: pl.DataFrame,
    ) -> Tuple[List[int], Optional[List[float]]]:
        """
        Encode a DataFrame into token sequences.

        Returns
        -------
        ids : list[int]
            Token ID sequence.
        vals : list[float] | None
            Parallel float values when num_type='continuous' (NaN for
            non-numeric positions); ``None`` for discrete mode.
        """
        if not self._trained:
            raise RuntimeError("Call train() before encode().")
        self._validate_schema(df)

        df_e = self._precompute(df)
        if self.has_bpe_classes:
            self._build_bpe_cache(df_e)

        ids: List[int] = []
        vals: Optional[List[float]] = [] if self.num_type == "continuous" else None

        last_id = None
        last_age: Optional[int] = None
        last_time: Optional[datetime] = None
        current_state: str = self.initial_state
        last_ms = None    # last milestone boundary key

        for row in df_e.iter_rows(named=True):
            cls   = row["class"]
            tv    = row["text_value"]
            nv    = row["numeric_value"]
            ts    = row["time"]                   # datetime

            # ── Skip demographic birth-date rows ──────────────────────
            if cls == self.birth_date_class and tv == self.birth_date_text_value:
                # EOS for previous patient if we had one
                if last_id is not None and row["id"] != last_id:
                    ids.append(self.vocab["<|eos|>"])
                    if vals is not None:
                        vals.append(float("nan"))

                last_id = row["id"]
                last_time = None   # explicitly no delta from birth row
                last_age = None
                last_ms = None
                current_state = self.initial_state

                # Emit SOS for new patient
                ids.append(self.vocab["<|sos|>"])
                if vals is not None:
                    vals.append(float("nan"))
                # Age will be emitted at the first real event via Kalman check
                continue

            # ── New patient (no birth-date row) → <|eos|> + <|sos|> ───
            if row["id"] != last_id:
                # EOS for previous patient
                if last_id is not None:
                    ids.append(self.vocab["<|eos|>"])
                    if vals is not None:
                        vals.append(float("nan"))

                ids.append(self.vocab["<|sos|>"])
                if vals is not None:
                    vals.append(float("nan"))
                last_id = row["id"]
                last_time = None
                last_age = None
                last_ms = None
                current_state = self.initial_state
                # Emit initial age if available
                age = self._compute_age(row["id"], ts)
                if age is not None and 0 <= age <= _MAX_AGE:
                    ids.append(self.vocab[f"<|age_{age}|>"])
                    if vals is not None:
                        vals.append(float("nan"))
                    last_age = age

            # ── Time delta + milestone (Kalman-filter style) ──────────
            if ts is not None and last_time is not None:
                dt = (ts - last_time).total_seconds()
                if dt > 0:
                    # Emit time delta
                    scale_key = self._classify_delta_scale(dt)
                    self._emit_time_delta(ids, vals, dt, scale_key)

                    # Determine milestone kind from current state
                    ms_kind = self.milestone_per_state.get(current_state, "none")
                    if ms_kind != "none":
                        cur_boundary = _milestone_boundary(
                            ms_kind, ts, self.milestone_shift_start
                        )
                        # Emit milestone only if boundary changed
                        if cur_boundary != last_ms:
                            ms_tok = _milestone_marker(
                                ms_kind, ts, self.milestone_shift_start
                            )
                            if ms_tok is not None and ms_tok in self.vocab:
                                ids.append(self.vocab[ms_tok])
                                if vals is not None:
                                    vals.append(float("nan"))
                        last_ms = cur_boundary

            # ── Update state machine ──────────────────────────────────
            for state, trigger_classes in self.state_transitions.items():
                if cls in trigger_classes:
                    current_state = state
                    last_ms = None  # reset milestone boundary on state change

            # ── Age milestone (Kalman-filter style) ───────────────────
            age = self._compute_age(row["id"], ts)
            if age is not None and 0 <= age <= _MAX_AGE and age != last_age:
                ids.append(self.vocab[f"<|age_{age}|>"])
                if vals is not None:
                    vals.append(float("nan"))
                last_age = age

            if ts is not None:
                last_time = ts

            # ── Class token ───────────────────────────────────────────
            ids.append(self.vocab[f"<|{cls}|>"])
            if vals is not None:
                vals.append(float("nan"))

            # ── Resolve effective text mode and num_seq ───────────────
            text_mode = self._get_text_mode(cls, tv)
            eff_seq = (
                "fused"
                if self.num_seq == "fused" and text_mode == "concept"
                else "factored"
            )

            num_key  = (cls, tv)
            num_info = self.numeric_params.get(num_key)
            has_num  = (
                nv is not None
                and not (isinstance(nv, float) and math.isnan(nv))
            )

            # ── Text tokens ───────────────────────────────────────────
            if tv is not None:
                if text_mode == "concept":
                    if eff_seq == "fused" and has_num and num_info is not None:
                        self._emit_fused_concept(ids, vals, tv, nv, num_info)
                    else:
                        ids.append(self._tok_id(tv))
                        if vals is not None:
                            vals.append(float("nan"))
                else:
                    # BPE
                    cache_idx = row["_bpe_cache_idx"]
                    if cache_idx >= 0:
                        bpe_toks = self.bpe_cache[cache_idx]
                        ids.extend(bpe_toks)
                        if vals is not None:
                            vals.extend([float("nan")] * len(bpe_toks))

            # ── Factored numeric token ────────────────────────────────
            if eff_seq == "factored" and has_num and num_info is not None:
                self._emit_factored_numeric(ids, vals, nv, num_info)

        # ── Append final <|eos|> ──────────────────────────────────────
        if ids and last_id is not None:
            ids.append(self.vocab["<|eos|>"])
            if vals is not None:
                vals.append(float("nan"))

        return ids, vals

    # ·····  emit helpers  ·············································

    def _emit_time_delta(
        self, ids: list, vals: Optional[list], dt: float, scale_key: str,
    ):
        """Emit time-delta token(s) for the given scale band *scale_key*."""
        td_name = f"<|delta_time_{scale_key}|>"
        td_p = self.time_delta_params.get(scale_key)
        if td_p is None:
            return

        if self.num_seq == "fused" and self.num_type == "discrete":
            bi = int(np.digitize(dt, td_p["edges"], right=False))
            ids.append(self.vocab[f"<|delta_time_{scale_key}_Q{bi}|>"])
            if vals is not None:
                vals.append(float("nan"))

        elif self.num_type == "discrete":
            ids.append(self.vocab[td_name])
            bi = int(np.digitize(dt, td_p["edges"], right=False))
            ids.append(self.vocab[f"<|Q{bi}|>"])
            if vals is not None:
                vals.extend([float("nan"), float("nan")])

        elif self.num_seq == "fused":
            ids.append(self.vocab[td_name])
            if vals is not None:
                vals.append(self._scale_td(dt, td_p))

        else:
            ids.append(self.vocab[td_name])
            ids.append(self.vocab["<|NUM|>"])
            if vals is not None:
                vals.append(float("nan"))
                vals.append(self._scale_td(dt, td_p))

    def _emit_fused_concept(
        self, ids: list, vals: Optional[list],
        tv: str, nv: float, ni: dict,
    ):
        """Emit a fused concept+numeric token."""
        if ni["type"] == "level":
            idx = self._level_idx(ni, nv)
            ids.append(self._tok_id(f"{tv}::L{idx}"))
            if vals is not None:
                vals.append(float("nan"))
        elif ni["type"] == "bins":
            bi = int(np.digitize(nv, ni["edges"], right=False))
            ids.append(self._tok_id(f"{tv}::Q{bi}"))
            if vals is not None:
                vals.append(float("nan"))
        elif ni["type"] == "scaling":
            ids.append(self._tok_id(tv))
            if vals is not None:
                vals.append(self.scale(ni, nv))

    def _emit_factored_numeric(
        self, ids: list, vals: Optional[list], nv: float, ni: dict,
    ):
        """Emit a separate numeric token (factored path)."""
        if ni["type"] == "level":
            idx = self._level_idx(ni, nv)
            ids.append(self.vocab[f"<|L{idx}|>"])
            if vals is not None:
                vals.append(float("nan"))
        elif ni["type"] == "bins":
            bi = int(np.digitize(nv, ni["edges"], right=False))
            ids.append(self.vocab[f"<|Q{bi}|>"])
            if vals is not None:
                vals.append(float("nan"))
        elif ni["type"] == "scaling":
            ids.append(self.vocab["<|NUM|>"])
            if vals is not None:
                vals.append(self.scale(ni, nv))

    # ·····  token / level look-ups  ···································

    def _tok_id(self, token: str) -> int:
        """Resolve *token* to its integer id (concept vocab or BPE)."""
        if token in self.vocab:
            return self.vocab[token]
        if self.bpe is not None:
            try:
                return self.bpe.encode_single_token(token)
            except KeyError:
                pass
        raise KeyError(f"Token '{token}' not found in vocabulary")

    @staticmethod
    def _level_idx(ni: dict, val: float) -> int:
        v2i = ni["value_to_idx"]
        if val in v2i:
            return v2i[val]
        sv = ni["values"]
        idx = int(np.searchsorted(sv, val))
        return max(0, min(idx, len(sv) - 1))

    def _compute_age(self, patient_id, ts) -> Optional[int]:
        """Compute integer age at *ts* for *patient_id*."""
        bd = self._birth_dates.get(patient_id)
        if bd is None or ts is None:
            return None
        delta_days = (ts - bd).total_seconds() / 86400
        return int(delta_days // 365)

    # ──────────────────────────────────────────────────────────────────────
    # Precompute & BPE cache
    # ──────────────────────────────────────────────────────────────────────

    def _precompute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived columns for encoding."""
        df = df.sort(["id", "time"])

        # ── Admission state ──────────────────────────────────────────
        # ── BPE cache index ──────────────────────────────────────────
        # Build a mask of rows that use BPE text mode (at pair level)
        text_modes_list = []
        for r in df.select(["class", "text_value"]).iter_rows():
            c, t = r
            text_modes_list.append(self._get_text_mode(c, t))

        is_bpe = pl.Series("_is_bpe", [1 if m == "bpe" else 0 for m in text_modes_list])
        bpe_cum = is_bpe.cum_sum()
        df = df.with_columns(
            pl.when(is_bpe == 1)
            .then(bpe_cum - 1)
            .otherwise(-1)
            .alias("_bpe_cache_idx")
        )

        return df

    def _build_bpe_cache(self, df: pl.DataFrame):
        """Batch-encode all BPE-mode text_value strings in one pass."""
        bpe_rows = df.filter(pl.col("_bpe_cache_idx") >= 0)

        if bpe_rows.height == 0:
            self.bpe_cache = []
            return

        texts = bpe_rows.select("text_value").to_series().to_list()
        texts = [t if t is not None else "" for t in texts]
        # tiktoken's encode_ordinary_batch is multithreaded in Rust
        self.bpe_cache = self.bpe.encode_ordinary_batch(texts)

    # ──────────────────────────────────────────────────────────────────────
    # Scaling helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def scale(params: dict, x: float) -> float:
        lo = params.get("min_threshold")
        hi = params.get("max_threshold")
        if lo is not None and hi is not None:
            x = max(lo, min(x, hi))
        f = _SCALING_FORMULAS.get(params.get("distribution"))
        return f["scale"](params, x) if f else x

    @staticmethod
    def unscale(params: dict, x_scaled: float) -> float:
        f = _SCALING_FORMULAS.get(params.get("distribution"))
        return f["unscale"](params, x_scaled) if f else x_scaled

    def _scale_td(self, dt: float, params: dict) -> float:
        return self.scale(params, dt)

    # ──────────────────────────────────────────────────────────────────────
    # Token classification (for metrics)
    # ──────────────────────────────────────────────────────────────────────

    def _build_token_type_lookup(self) -> Dict[int, str]:
        """Build a {token_id: type_str} lookup for all known tokens.

        Type strings:
            'sos', 'eos', 'pad', 'num_marker', 'row',
            'time_delta', 'time_delta_fused',
            'milestone', 'age',
            'class',
            'concept', 'bpe',
            'Q', 'L',
            'fused_concept_Q', 'fused_concept_L'
        """
        import re as _re

        _re_q       = _re.compile(r"^<\|Q(\d+)\|>$")
        _re_l       = _re.compile(r"^<\|L(\d+)\|>$")
        _re_age     = _re.compile(r"^<\|age_\d+\|>$")
        _re_ms      = _re.compile(r"^<\|ms_")
        # Build dynamic patterns from actual scale keys
        _sk_pat = "|".join(_re.escape(sk) for sk in self._scale_keys())
        _re_td_fuse = _re.compile(rf"^<\|delta_time_(?:{_sk_pat})_Q(\d+)\|>$")
        _re_td_mrk  = _re.compile(rf"^<\|delta_time_(?:{_sk_pat})\|>$")
        _all_special_re = _re.compile(
            rf"^<\|(?:sos|eos|pad|NUM|ROW|"
            rf"delta_time_(?:{_sk_pat})(?:_Q\d+)?|"
            r"Q\d+|L\d+|age_\d+|ms_.+)\|>$"
        )

        lookup: Dict[int, str] = {}

        # Classify everything in self.vocab (special + concept tokens)
        for tok_str, tok_id in self.vocab.items():
            if tok_str == "<|sos|>":
                lookup[tok_id] = "sos"
            elif tok_str == "<|eos|>":
                lookup[tok_id] = "eos"
            elif tok_str == "<|pad|>":
                lookup[tok_id] = "pad"
            elif tok_str == "<|NUM|>":
                lookup[tok_id] = "num_marker"
            elif tok_str == "<|ROW|>":
                lookup[tok_id] = "row"
            elif _re_td_fuse.match(tok_str):
                lookup[tok_id] = "time_delta_fused"
            elif _re_td_mrk.match(tok_str):
                lookup[tok_id] = "time_delta"
            elif _re_q.match(tok_str):
                lookup[tok_id] = "Q"
            elif _re_l.match(tok_str):
                lookup[tok_id] = "L"
            elif _re_age.match(tok_str):
                lookup[tok_id] = "age"
            elif _re_ms.match(tok_str):
                lookup[tok_id] = "milestone"
            elif "::" in tok_str:
                if "::L" in tok_str:
                    lookup[tok_id] = "fused_concept_L"
                elif "::Q" in tok_str:
                    lookup[tok_id] = "fused_concept_Q"
                else:
                    lookup[tok_id] = "concept"
            elif (tok_str.startswith("<|") and tok_str.endswith("|>")
                  and not _all_special_re.match(tok_str)):
                lookup[tok_id] = "class"
            else:
                # Plain concept token (no <|...|> wrapper)
                if not tok_str.startswith("<|"):
                    lookup[tok_id] = "concept"

        return lookup

    def classify_token_ids(self, ids: List[int]) -> List[str]:
        """Classify each token ID into a type string.

        Returns a list parallel to *ids* with one of:
            'sos', 'eos', 'pad', 'num_marker', 'row',
            'time_delta', 'time_delta_fused',
            'milestone', 'age', 'class',
            'concept', 'bpe',
            'Q', 'L',
            'fused_concept_Q', 'fused_concept_L'
        """
        if not hasattr(self, "_token_type_lookup"):
            self._token_type_lookup = self._build_token_type_lookup()
        lut = self._token_type_lookup
        return [lut.get(tid, "bpe") for tid in ids]

    def get_numeric_context(
        self, ids: List[int],
    ) -> List[Optional[dict]]:
        """Walk a token sequence and return the numeric params dict for each
        position where a density adjustment is needed.

        For time-delta positions returns ``time_delta_params["ip"|"op"]``.
        For row-level Q / L / NUM / fused_concept_Q / fused_concept_L
        returns the appropriate ``numeric_params[(cls, tv)]`` dict.
        All other positions return ``None``.

        Must be called **after** :meth:`classify_token_ids`.
        """
        import re as _re
        if not hasattr(self, "_token_type_lookup"):
            self._token_type_lookup = self._build_token_type_lookup()

        _sk_pat = "|".join(_re.escape(sk) for sk in self._scale_keys())
        _re_td_fuse = _re.compile(rf"^<\|delta_time_(?:{_sk_pat})_Q(\d+)\|>$")
        _re_td_mrk  = _re.compile(rf"^<\|delta_time_({_sk_pat})\|>$")

        types = self.classify_token_ids(ids)
        out: List[Optional[dict]] = [None] * len(ids)

        current_cls: Optional[str] = None
        current_tv: Optional[str] = None
        pending_td_key: Optional[str] = None  # "ip" or "op"
        bpe_pieces: List[str] = []
      
        for i, (tid, ttype) in enumerate(zip(ids, types)):
            tok = self.decode_token(tid)

            # Track class context
            if ttype == "class":
                current_cls = tok[2:-2]  # strip <| and |>
                current_tv = None
                pending_td_key = None
                bpe_pieces = []
                continue

            # Track text context
            if ttype == "concept":
                current_tv = tok
                bpe_pieces = []
                continue
              
            # Accumulate BPE subwords to reconstruct text_value
            if ttype == "bpe":
                bpe_pieces.append(tok)
                accumulated = "".join(bpe_pieces)
                ni = self.numeric_params.get((current_cls, accumulated))
                if ni is not None:
                    current_tv = accumulated
                    if ni.get("type") == "scaling":
                        out[i] = ni
                continue

            # Time-delta fused: params come from time_delta_params
            if ttype == "time_delta_fused":
                bpe_pieces = []
                if _re_td_fuse.match(tok):
                    # Extract scale key: <|delta_time_SK_Qn|> → SK
                    inner = tok[len("<|delta_time_"):-2]   # "SK_Qn"
                    sk = inner.rsplit("_Q", 1)[0]
                    td = self.time_delta_params.get(sk)
                    if td is not None:
                        out[i] = {**td, "_is_time_delta": True}
                continue

            # Time-delta marker: note the key for the next Q/NUM
            if ttype == "time_delta":
                bpe_pieces = []
                m = _re_td_mrk.match(tok)
                if m:
                    pending_td_key = m.group(1)
                continue

            # Q or NUM following a time-delta marker
            if ttype in ("Q", "num_marker") and pending_td_key is not None:
                bpe_pieces = []
                td = self.time_delta_params.get(pending_td_key)
                if td is not None:
                    out[i] = {**td, "_is_time_delta": True}
                pending_td_key = None
                continue

            # Row-level Q, L, num_marker
            if ttype in ("Q", "L", "num_marker"):
                bpe_pieces = []
                out[i] = self.numeric_params.get((current_cls, current_tv))
                continue

            # Fused concept tokens
            if ttype in ("fused_concept_Q", "fused_concept_L"):
                bpe_pieces = []
                tv_part = tok.rsplit("::", 1)[0]
                out[i] = self.numeric_params.get((current_cls, tv_part))
                current_tv = tv_part
                continue

            # Reset pending_td on anything else
            if ttype not in ("milestone", "age"):
                pending_td_key = None

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Debug encoding
    # ──────────────────────────────────────────────────────────────────────

    def encode_debug(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Like :meth:`encode` but returns the original DataFrame with a
        ``tokens`` column containing human-readable token strings per row.
        """
        if not self._trained:
            raise RuntimeError("Call train() before encode_debug().")
        self._validate_schema(df)

        df_e = self._precompute(df)
        if self.has_bpe_classes:
            self._build_bpe_cache(df_e)

        tokens_col: List[List[str]] = []
        last_id = None
        last_age: Optional[int] = None
        last_time: Optional[datetime] = None
        current_state: str = self.initial_state
        last_ms = None

        for row in df_e.iter_rows(named=True):
            rt: List[str] = []
            cls   = row["class"]
            tv    = row["text_value"]
            nv    = row["numeric_value"]
            ts    = row["time"]


            # ── Demographic birth-date row ────────────────────────────
            if cls == self.birth_date_class and tv == self.birth_date_text_value:
                if last_id is not None and row["id"] != last_id:
                    rt.append("<|eos|>")
                rt.append("<|sos|>")
                last_id = row["id"]
                last_time = None
                last_age = None
                last_ms = None
                current_state = self.initial_state
                # Age will be emitted at the first real event via Kalman check
                tokens_col.append(rt)
                continue

            # ── New patient ───────────────────────────────────────────
            if row["id"] != last_id:
                if last_id is not None:
                    rt.append("<|eos|>")
                rt.append("<|sos|>")
                last_id = row["id"]
                last_time = None
                last_age = None
                last_ms = None
                current_state = self.initial_state
                age = self._compute_age(row["id"], ts)
                if age is not None and 0 <= age <= _MAX_AGE:
                    rt.append(f"<|age_{age}|>")
                    last_age = age

            # ── Time delta + milestone ────────────────────────────────
            if ts is not None and last_time is not None:
                dt = (ts - last_time).total_seconds()
                if dt > 0:
                    sk = self._classify_delta_scale(dt)
                    td_p = self.time_delta_params.get(sk)
                    if td_p is not None:
                        if self.num_seq == "fused" and self.num_type == "discrete":
                            bi = int(np.digitize(dt, td_p["edges"], right=False))
                            rt.append(f"<|delta_time_{sk}_Q{bi}|>")
                        elif self.num_type == "discrete":
                            bi = int(np.digitize(dt, td_p["edges"], right=False))
                            rt.extend([
                                f"<|delta_time_{sk}|>",
                                f"<|Q{bi}|>"
                            ])
                        elif self.num_seq == "fused":
                            s = self._scale_td(dt, td_p)
                            rt.append(f"<|delta_time_{sk}|>({s:.3f})")
                        else:
                            s = self._scale_td(dt, td_p)
                            rt.extend([
                                f"<|delta_time_{sk}|>",
                                f"<|NUM|>({s:.3f})"
                            ])

                    # Milestone
                    ms_kind = self.milestone_per_state.get(current_state, "none")
                    if ms_kind != "none":
                        cur_boundary = _milestone_boundary(
                            ms_kind, ts, self.milestone_shift_start
                        )
                        if cur_boundary != last_ms:
                            ms_tok = _milestone_marker(
                                ms_kind, ts, self.milestone_shift_start
                            )
                            if ms_tok is not None:
                                rt.append(ms_tok)
                        last_ms = cur_boundary

            # ── Update state machine ─────────────────────────────────────
            for state, trigger_classes in self.state_transitions.items():
                if cls in trigger_classes:
                    current_state = state
                    last_ms = None  # reset milestone boundary on state change

            # ── Age milestone ─────────────────────────────────────────
            age = self._compute_age(row["id"], ts)
            if age is not None and 0 <= age <= _MAX_AGE and age != last_age:
                rt.append(f"<|age_{age}|>")
                last_age = age

            if ts is not None:
                last_time = ts

            rt.append(f"<|{cls}|>")

            text_mode = self._get_text_mode(cls, tv)
            eff_seq = (
                "fused"
                if self.num_seq == "fused" and text_mode == "concept"
                else "factored"
            )
            num_key  = (cls, tv)
            num_info = self.numeric_params.get(num_key)
            has_num  = (
                nv is not None
                and not (isinstance(nv, float) and math.isnan(nv))
            )

            if tv is not None:
                if text_mode == "concept":
                    if eff_seq == "fused" and has_num and num_info is not None:
                        if num_info["type"] == "level":
                            idx = self._level_idx(num_info, nv)
                            rt.append(f"{tv}::L{idx}")
                        elif num_info["type"] == "bins":
                            bi = int(np.digitize(
                                nv, num_info["edges"], right=False
                            ))
                            rt.append(f"{tv}::Q{bi}")
                        elif num_info["type"] == "scaling":
                            s = self.scale(num_info, nv)
                            rt.append(f"{tv}({s:.3f})")
                    else:
                        rt.append(tv)
                else:
                    cache_idx = row["_bpe_cache_idx"]
                    if cache_idx >= 0:
                        bpe_toks = self.bpe_cache[cache_idx]
                        rt.extend(self.bpe.decode([t]) for t in bpe_toks)
                        # Note: decode([t]) returns the text chunk for each BPE token

            if eff_seq == "factored" and has_num and num_info is not None:
                if num_info["type"] == "level":
                    idx = self._level_idx(num_info, nv)
                    rt.append(f"<|L{idx}|>")
                elif num_info["type"] == "bins":
                    bi = int(np.digitize(nv, num_info["edges"], right=False))
                    rt.append(f"<|Q{bi}|>")
                elif num_info["type"] == "scaling":
                    s = self.scale(num_info, nv)
                    rt.append(f"<|NUM|>({s:.3f})")

            tokens_col.append(rt)

        assert len(tokens_col) == len(df), (
            f"Token column length {len(tokens_col)} != DataFrame height {len(df)}"
        )
        return df.with_columns(pl.Series("tokens", tokens_col))

    # ──────────────────────────────────────────────────────────────────────
    # Decode
    # ──────────────────────────────────────────────────────────────────────

    def decode_token(self, token_id: int) -> str:
        if token_id in self.ivocab:
            return self.ivocab[token_id]
        if self.bpe is not None:
            return self.bpe.decode([token_id])
        return f"<UNK:{token_id}>"

    def decode(self, ids: List[int]) -> List[str]:
        return [self.decode_token(i) for i in ids]

    def decode_to_dataframe(
        self,
        ids: List[int],
        vals: Optional[List[float]] = None,
        reference_time: Optional[datetime] = None,
        start_patient_id: int = 0,
    ) -> "pl.DataFrame":
        """
        Invert a token-ID stream back into a Polars DataFrame.

        The output schema mirrors the encoder input:
        ``id``, ``time``, ``class``, ``text_value``, ``numeric_value``.

        Parameters
        ----------
        ids : list[int]
            Token ID sequence produced by :meth:`encode`.
        vals : list[float] | None
            Parallel float array from :meth:`encode` (required for
            ``num_type='continuous'``; may be omitted for ``'discrete'``).
        reference_time : datetime, optional
            Absolute base timestamp for the first event of each patient.
            Defaults to ``datetime(1970, 1, 1)``.  Because the encoder
            stores only *relative* deltas, absolute timestamps are not
            recoverable without this anchor.
        start_patient_id : int
            Integer ID assigned to the first patient sequence.
            Subsequent patients increment by 1.

        Returns
        -------
        pl.DataFrame
            Reconstructed frame sorted by ``(id, time)``.

        Notes
        -----
        * Birth-date rows used as ``<|sos|>`` triggers are not emitted.
        * Age and milestone tokens carry no recoverable payload and are
          silently dropped.
        * Bin numeric values use edge midpoints (lowest/highest edge for
          the outermost bins) — they are approximations of the originals.
        * Level numeric values are recovered exactly via the stored
          ``value_to_idx`` table.
        * Continuous (scaling) values are approximately recovered via the
          distribution-inverse transform; unscaling is only exact for
          ``minmax`` distribution.
        * BPE subword pieces are concatenated (no separator) to restore
          ``text_value``.
        * The ``inpatient`` column is not reconstructed.
        """
        import re as _re
        import math as _math
        from datetime import timedelta as _td

        if reference_time is None:
            reference_time = datetime(1970, 1, 1)

        # ── Pre-build token-classification helpers ────────────────────
        _HARD_SPECIALS = {
            "<|sos|>", "<|eos|>", "<|pad|>",
            "<|NUM|>", "<|ROW|>",
        }
        # Patterns for structured special tokens
        _re_q       = _re.compile(r"^<\|Q(\d+)\|>$")
        _re_l       = _re.compile(r"^<\|L(\d+)\|>$")
        _re_age     = _re.compile(r"^<\|age_\d+\|>$")
        _re_ms      = _re.compile(r"^<\|ms_")
        # Build dynamic patterns from actual scale keys
        _sk_pat = "|".join(_re.escape(sk) for sk in self._scale_keys())
        _re_td_fuse = _re.compile(rf"^<\|delta_time_({_sk_pat})_Q(\d+)\|>$")
        _re_td_mrk  = _re.compile(rf"^<\|delta_time_({_sk_pat})\|>$")

        # Class tokens: <|...|> tokens that are NOT any of the above
        _all_special_re = _re.compile(
            rf"^<\|(?:sos|eos|pad|NUM|ROW|"
            rf"delta_time_(?:{_sk_pat})(?:_Q\d+)?|"
            r"Q\d+|L\d+|age_\d+|ms_.+)\|>$"
        )
        _class_token_set = {
            tok for tok in self.vocab
            if tok.startswith("<|") and tok.endswith("|>")
            and not _all_special_re.match(tok)
        }

        def _bin_midpoint(edges: list, bin_idx: int) -> float:
            """Midpoint of bin *bin_idx* given quantile *edges*.

            ``np.digitize(x, edges, right=False)`` produces:
              bin 0      → x < edges[0]         → return edges[0]
              bin k      → edges[k-1] <= x < edges[k]  → midpoint
              bin N      → x >= edges[-1]        → return edges[-1]
            """
            n = len(edges)
            if n == 0:
                return 0.0
            if bin_idx <= 0:
                return float(edges[0])
            if bin_idx >= n:
                return float(edges[-1])
            return (float(edges[bin_idx - 1]) + float(edges[bin_idx])) / 2.0

        # ── State ─────────────────────────────────────────────────────
        rows: List[dict] = []
        patient_id: int = start_patient_id - 1   # incremented on first <|sos|>
        current_time: datetime = reference_time

        # Per-row accumulators
        pending_cls: Optional[str] = None
        pending_tv_pieces: List[str] = []
        pending_nv: Optional[float] = None        # fully resolved numeric
        pending_scaled_val: Optional[float] = None  # fused-scaling raw val
        pending_td_key: Optional[str] = None       # factored td: waiting for Q/NUM

        def _flush() -> None:
            nonlocal pending_cls, pending_tv_pieces, pending_nv, pending_scaled_val
            if pending_cls is None:
                return
            tv_str: Optional[str] = "".join(pending_tv_pieces) or None
            nv = pending_nv
            # Fused-scaling: concept token carried the scaled val in vals[i]
            if nv is None and pending_scaled_val is not None:
                num_info = self.numeric_params.get((pending_cls, tv_str))
                if num_info is not None and num_info.get("type") == "scaling":
                    nv = self.unscale(num_info, pending_scaled_val)
            rows.append({
                "id":            patient_id,
                "time":          current_time,
                "class":         pending_cls,
                "text_value":    tv_str,
                "numeric_value": nv,
            })
            pending_cls = None
            pending_tv_pieces = []
            pending_nv = None
            pending_scaled_val = None

        # ── Main decode loop ──────────────────────────────────────────
        for i, tok_id in enumerate(ids):
            tok = self.decode_token(tok_id)
            v_i = vals[i] if (vals is not None) else float("nan")

            # ── Structural / bookkeeping ──────────────────────────────
            if tok == "<|sos|>":
                _flush()
                patient_id += 1
                current_time = reference_time
                pending_td_key = None
                continue

            if tok in ("<|eos|>", "<|pad|>"):
                _flush()
                pending_td_key = None
                continue

            if tok == "<|ROW|>" or _re_age.match(tok) or _re_ms.match(tok):
                continue   # informational only

            # ── Fused time-delta (single combinatorial token) ─────────
            m = _re_td_fuse.match(tok)
            if m:
                td_key = m.group(1)
                bin_idx = int(m.group(2))
                td_p = self.time_delta_params.get(td_key)
                if td_p is not None:
                    dt = _bin_midpoint(td_p["edges"], bin_idx)
                    current_time = current_time + _td(seconds=dt)
                continue

            # ── Time-delta marker token ───────────────────────────────
            m = _re_td_mrk.match(tok)
            if m:
                td_key = m.group(1)
                td_p = self.time_delta_params.get(td_key)
                if td_p is not None:
                    if not _math.isnan(v_i):
                        # Fused-continuous: val IS stored at this position
                        dt = self.unscale(td_p, v_i)
                        current_time = current_time + _td(seconds=max(dt, 0.0))
                    else:
                        # Factored mode: next Q/NUM belongs to this delta
                        pending_td_key = td_key
                continue

            # ── Bin token ─────────────────────────────────────────────
            m = _re_q.match(tok)
            if m:
                bin_idx = int(m.group(1))
                if pending_td_key is not None:
                    # Factored-discrete time delta
                    td_p = self.time_delta_params.get(pending_td_key)
                    if td_p is not None:
                        dt = _bin_midpoint(td_p["edges"], bin_idx)
                        current_time = current_time + _td(seconds=dt)
                    pending_td_key = None
                else:
                    # Factored row-level bin numeric
                    tv_str = "".join(pending_tv_pieces) or None
                    num_info = self.numeric_params.get((pending_cls, tv_str))
                    if num_info is not None and "edges" in num_info:
                        pending_nv = _bin_midpoint(num_info["edges"], bin_idx)
                continue

            # ── Level token ───────────────────────────────────────────
            m = _re_l.match(tok)
            if m:
                level_idx = int(m.group(1))
                tv_str = "".join(pending_tv_pieces) or None
                num_info = self.numeric_params.get((pending_cls, tv_str))
                if num_info is not None and "values" in num_info:
                    lvls = num_info["values"]
                    idx = max(0, min(level_idx, len(lvls) - 1))
                    pending_nv = float(lvls[idx])
                continue

            # ── Continuous numeric token ──────────────────────────────
            if tok == "<|NUM|>":
                if pending_td_key is not None:
                    # Factored-continuous time delta
                    td_p = self.time_delta_params.get(pending_td_key)
                    if td_p is not None and not _math.isnan(v_i):
                        dt = self.unscale(td_p, v_i)
                        current_time = current_time + _td(seconds=max(dt, 0.0))
                    pending_td_key = None
                else:
                    # Factored-continuous row-level numeric
                    tv_str = "".join(pending_tv_pieces) or None
                    num_info = self.numeric_params.get((pending_cls, tv_str))
                    if num_info is not None and not _math.isnan(v_i):
                        pending_nv = self.unscale(num_info, v_i)
                continue

            # ── Class token ───────────────────────────────────────────
            if tok in _class_token_set:
                _flush()
                # Extract class name from <|cls|>
                pending_cls = tok[2:-2]
                pending_tv_pieces = []
                pending_nv = None
                pending_scaled_val = None
                pending_td_key = None
                continue

            # ── Fused concept+numeric token (tv::LN or tv::QN) ───────
            if "::" in tok:
                tv_part, qual = tok.rsplit("::", 1)
                pending_tv_pieces = [tv_part]
                if qual.startswith("L"):
                    level_idx = int(qual[1:])
                    num_info = self.numeric_params.get((pending_cls, tv_part))
                    if num_info is not None and "values" in num_info:
                        lvls = num_info["values"]
                        idx = max(0, min(level_idx, len(lvls) - 1))
                        pending_nv = float(lvls[idx])
                elif qual.startswith("Q"):
                    bin_idx = int(qual[1:])
                    num_info = self.numeric_params.get((pending_cls, tv_part))
                    if num_info is not None and "edges" in num_info:
                        pending_nv = _bin_midpoint(num_info["edges"], bin_idx)
                continue

            # ── Plain concept / BPE-subword text token ────────────────
            # Fused-scaling path: val at this position is the scaled nv
            if not _math.isnan(v_i):
                pending_scaled_val = v_i
            pending_tv_pieces.append(tok)

        # Flush final patient
        _flush()

        # ── Build DataFrame ───────────────────────────────────────────
        if not rows:
            return pl.DataFrame(
                schema={
                    "id":            pl.Int64,
                    "time":          pl.Datetime,
                    "class":         pl.Utf8,
                    "text_value":    pl.Utf8,
                    "numeric_value": pl.Float64,
                }
            )

        return pl.DataFrame(
            rows,
            schema={
                "id":            pl.Int64,
                "time":          pl.Datetime,
                "class":         pl.Utf8,
                "text_value":    pl.Utf8,
                "numeric_value": pl.Float64,
            },
        ).sort(["id", "time"])

    # ──────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Persist trained tokenizer to disk (JSON + optional BPE file)."""
        if not self._trained:
            raise RuntimeError("Cannot save an untrained tokenizer.")
        p = Path(path)

        ser_np = {}
        for k, v in self.numeric_params.items():
            sk = f"{k[0]}||{k[1]}"
            v_copy = dict(v)
            if "values" in v_copy:
                v_copy["values"] = [
                    float(x) if isinstance(x, (np.floating, np.integer)) else x
                    for x in v_copy["values"]
                ]
            if "value_to_idx" in v_copy:
                v_copy["value_to_idx"] = {
                    str(vk): vi for vk, vi in v_copy["value_to_idx"].items()
                }
            ser_np[sk] = v_copy

        # Serialize pair_text_modes with tuple keys as "cls||tv"
        ser_ptm = {
            f"{k[0]}||{k[1]}": v for k, v in self.pair_text_modes.items()
        }
        # Serialize text_mode_overrides (may have tuple keys)
        ser_tmo = {}
        for k, v in self.text_mode_overrides.items():
            if isinstance(k, tuple):
                ser_tmo[f"__tuple__{k[0]}||{k[1]}"] = v
            else:
                ser_tmo[k] = v

        # Serialize birth dates
        ser_bd = {
            str(k): v.isoformat() if hasattr(v, 'isoformat') else str(v)
            for k, v in self._birth_dates.items()
        }

        config = {
            "text_mode_default": self.text_mode_default,
            "text_mode_overrides": ser_tmo,
            "text_mode_threshold": self.text_mode_threshold,
            "num_type": self.num_type,
            "num_seq": self.num_seq,
            "n_bins": self.n_bins,
            "level_threshold": self.level_threshold,
            "bin_clip_min": self.bin_clip_min,
            "bin_clip_max": self.bin_clip_max,
            "continuous_clip_min": self.continuous_clip_min,
            "continuous_clip_max": self.continuous_clip_max,
            "distributions": self.distributions,
            "final_vocab_size": self.final_vocab_size,
            "split_pattern": self.split_pattern,
            "time_scales": self.time_scales,
            "time_scale_names": self.time_scale_names,
            "state_transitions": self.state_transitions,
            "initial_state": self.initial_state,
            "milestone_per_state": self.milestone_per_state,
            "milestone_shift_start": self.milestone_shift_start,
            "birth_date_class": self.birth_date_class,
            "birth_date_text_value": self.birth_date_text_value,
            "class_text_modes": self.class_text_modes,
            "pair_text_modes": ser_ptm,
            "numeric_params": ser_np,
            "time_delta_params": self.time_delta_params,
            "vocab": self.vocab,
            "has_bpe_classes": self.has_bpe_classes,
            "dist_recommendations": self._dist_recommendations,
            "birth_dates": ser_bd,
        }

        with open(p.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.bpe is not None:
            import pickle as _pkl
            with open(str(p.with_suffix(".bpe")), "wb") as _f:
                _pkl.dump(self.bpe, _f)

        print(f"Saved tokenizer to {p.with_suffix('.json')}")

    @classmethod
    def load(cls, path: str) -> "DBTokenizer":
        """Restore a trained tokenizer from disk."""
        p = Path(path)
        with open(p.with_suffix(".json"), "r") as f:
            cfg = json.load(f)

        # Deserialize text_mode_overrides (tuple keys)
        raw_tmo = cfg.get("text_mode_overrides", {})
        tmo: Dict = {}
        for k, v in raw_tmo.items():
            if k.startswith("__tuple__"):
                parts = k[len("__tuple__"):].split("||", 1)
                tmo[(parts[0], parts[1])] = v
            else:
                tmo[k] = v

        tok = cls(
            text_mode_default=cfg["text_mode_default"],
            text_mode_overrides=tmo,
            text_mode_threshold=cfg["text_mode_threshold"],
            num_type=cfg["num_type"],
            num_seq=cfg["num_seq"],
            n_bins=cfg["n_bins"],
            level_threshold=cfg["level_threshold"],
            bin_clip_min=cfg["bin_clip_min"],
            bin_clip_max=cfg["bin_clip_max"],
            continuous_clip_min=cfg["continuous_clip_min"],
            continuous_clip_max=cfg["continuous_clip_max"],
            distributions=cfg.get("distributions"),
            final_vocab_size=cfg["final_vocab_size"],
            split_pattern=cfg["split_pattern"],
            time_scales=cfg.get("time_scales", []),
            time_scale_names=cfg.get("time_scale_names"),
            state_transitions=cfg.get("state_transitions", {}),
            initial_state=cfg.get("initial_state", "default"),
            milestone_per_state=cfg.get("milestone_per_state", {}),
            milestone_shift_start=cfg.get("milestone_shift_start", 7),
            birth_date_class=cfg.get("birth_date_class", "demographic"),
            birth_date_text_value=cfg.get("birth_date_text_value", "birth_date"),
        )

        tok.class_text_modes = cfg["class_text_modes"]

        # Deserialize pair_text_modes
        raw_ptm = cfg.get("pair_text_modes", {})
        tok.pair_text_modes = {}
        for k, v in raw_ptm.items():
            parts = k.split("||", 1)
            tok.pair_text_modes[(parts[0], parts[1])] = v

        # Deserialise numeric_params
        tok.numeric_params = {}
        for sk, v in cfg["numeric_params"].items():
            parts = sk.split("||", 1)
            key = (parts[0], None if parts[1] == "None" else parts[1])
            if v.get("type") == "level" and "values" in v:
                v["value_to_idx"] = {
                    float(vk): vi for vk, vi in v["value_to_idx"].items()
                }
                v["values"] = [float(x) for x in v["values"]]
            tok.numeric_params[key] = v

        tok.time_delta_params = cfg["time_delta_params"]
        tok.vocab = cfg["vocab"]
        tok.ivocab = {int(v): k for k, v in tok.vocab.items()}
        tok.has_bpe_classes = cfg["has_bpe_classes"]
        tok._dist_recommendations = cfg.get("dist_recommendations", {})

        # Deserialize birth dates
        raw_bd = cfg.get("birth_dates", {})
        tok._birth_dates = {}
        for k, v in raw_bd.items():
            try:
                tok._birth_dates[int(k)] = datetime.fromisoformat(v)
            except (ValueError, TypeError):
                tok._birth_dates[k] = datetime.fromisoformat(v)

        if tok.has_bpe_classes:
            if not HAS_BPE_BACKEND:
                raise ImportError(
                    "tiktoken is required to load a BPE-enabled tokenizer."
                )
            import pickle as _pkl
            with open(str(p.with_suffix(".bpe")), "rb") as _f:
                tok.bpe = _pkl.load(_f)

        tok._trained = True
        print(f"Loaded tokenizer from {p.with_suffix('.json')}")
        return tok
