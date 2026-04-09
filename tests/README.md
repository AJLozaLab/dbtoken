# Bits-Per-Row Metric: Test Walkthrough

This document explains every test in `tests/test_metrics.py` and why each one is necessary to prove that density-adjusted BPR is the correct metric for comparing model performance across different tokenization schemes.

---

## Background

When we tokenize numeric data, different schemes produce different token vocabularies:

| Scheme | Numeric token | What the model predicts |
|--------|--------------|------------------------|
| **Discrete** (Q bins) | `<\|Q2\|>` — one of *Q* quantile bins | Categorical distribution over bins |
| **Continuous** (scaling) | `<\|NUM\|>` + a float value | Gaussian mean and variance in z-space |
| **Level** | `<\|L3\|>` — one of a few distinct values | Categorical distribution over levels |

Raw cross-entropy bits are **not comparable** across these schemes because a Q-bin token implicitly covers a *range* of values while a continuous prediction specifies a *density*.  The density adjustment converts all schemes to a common unit: **bits of information per unit of original measurement scale**.

$$
\text{density-adjusted bits} = -\log_2 \frac{p(Q_k)}{w_k} = -\log_2 p(Q_k) + \log_2 w_k = \text{raw} + \text{adj}
$$

For the continuous case, the Jacobian determinant of the scaling transform plays the same role as the bin width.

---

## Test Data

All tests share two minimal datasets built by helper functions:

- **`_make_mini_df()`** — 12 rows of `(lab, hemoglobin, numeric)` with values 10–21 at hourly intervals.  12 distinct values exceeds `level_threshold=10`, so the tokenizer creates quantile bins rather than levels.

- **`_make_level_df()`** — 12 rows with only 3 distinct numeric values (1, 2, 3), which stays below the level threshold and produces level tokens instead of bins.

Four tokenizer fixtures are trained on these datasets:

| Fixture | `num_type` | `num_seq` | Dataset |
|---------|-----------|----------|---------|
| `discrete_tok` | discrete | factored | mini |
| `continuous_tok` | continuous | factored | mini |
| `fused_tok` | discrete | fused | mini |
| `level_tok` | discrete | factored | level |

---

## Test 1 — Cross-Scheme Comparability (the "money shot")

**Class:** `TestCrossSchemeComparability`

**What it proves:** If a discrete model and a continuous model are equally good at predicting a numeric value, they produce *identical* density-adjusted bits.

**Setup:**
1. Encode the same data with both `discrete_tok` (Q bins) and `continuous_tok` (Gaussian scaling).
2. Construct two hypothetical "oracle" models:
   - Discrete oracle: assigns probability 1.0 to the correct bin → raw bits = 0.
   - Continuous oracle: Gaussian with $\sigma_z$ chosen so the original-space density exactly equals $1/w_k$.

**The math:**

For the discrete oracle:
$$-\log_2\!\left(\frac{p(Q_k)}{w_k}\right) = -\log_2\!\left(\frac{1}{w_k}\right) = \log_2(w_k)$$

For the continuous oracle with $p_x(x) = p_z(z) \cdot |dz/dx|$ set to $1/w_k$:
$$-\log_2(p_x) = -\log_2(1/w_k) = \log_2(w_k)$$

Both give $\log_2(w_k)$.  The test asserts this equality to $10^{-10}$ tolerance.

**Why it matters:** This is the whole reason the metric exists.  If this test fails, the metric cannot be used to compare tokenization strategies.

---

## Test 2 — Density Adjustment Math

**Class:** `TestDensityAdjustmentMath`

Verifies the two building blocks — `_bin_width` and `_jacobian_log_abs` — against hand-computed values.

| Test | What it checks |
|------|---------------|
| `test_bin_width_interior` | Interior bin: $w = \text{edges}[i] - \text{edges}[i-1]$ |
| `test_bin_width_first` | Boundary bin 0 uses neighbor width as proxy |
| `test_bin_width_last` | Boundary bin $N$ uses neighbor width as proxy |
| `test_bin_width_unequal` | Non-uniform edges: bins of width 1, 9, 90 |
| `test_jacobian_normal` | Normal: $\log|dz/dx| = -\frac{1}{2}\log(\text{var})$ |
| `test_jacobian_gamma` | Gamma: $\log|dz/dx| = -\frac{1}{2}\log(\alpha\beta^2)$ |
| `test_jacobian_minmax` | Minmax: $\log|dz/dx| = -\log(\max - \min)$ |
| `test_normal_jacobian_matches_bin_resolution` | **Key insight:** bin width 10 and $\sigma=10$ give *the same* adjustment ($\log_2 10 \approx 3.322$ bits).  This is the mathematical reason cross-scheme comparison works. |
| `test_jacobian_lognormal_value_dependent` | Lognormal Jacobian is value-dependent: $|dz/dx| = 1/(x \cdot \sigma)$, verified at $z=0$ and $z=1$. |

---

## Test 3 — Token Classification

**Class:** `TestTokenClassification`

Verifies that `classify_token_ids()` assigns the correct type string to every token.

| Test | Tokenizer | Checks |
|------|-----------|--------|
| `test_discrete_classification` | discrete | Every decoded token maps to the expected type (`sos`, `eos`, `time_delta`, `Q`, `class`, `concept`, etc.) |
| `test_continuous_has_num_markers` | continuous | Has `num_marker` tokens, zero `Q` tokens |
| `test_fused_has_fused_tokens` | fused | Produces `fused_concept_Q` or `time_delta_fused` tokens |

**Why it matters:** Classification drives all downstream logic — density adjustment, group routing, and reporting breakdowns all depend on correct token types.

---

## Test 4 — Numeric Context Tracking

**Class:** `TestNumericContext`

Verifies that `get_numeric_context()` attaches the right distribution parameters to each position in the sequence.

| Test | What it checks |
|------|---------------|
| `test_discrete_q_has_edges` | Q tokens have context with `"edges"`; structural tokens (sos, class, concept) have `None` |
| `test_time_q_vs_value_q_distinguished` | Q tokens after a time-delta marker get `_is_time_delta=True` context; Q tokens after a concept do not |
| `test_continuous_num_has_scaling_context` | NUM markers get context with `"type": "scaling"` |

**Why it matters:** A Q token after `<|delta_time_op|>` represents a *time-delta bin*, not a *lab-value bin*.  Using the wrong distribution parameters would corrupt the density adjustment.

---

## Test 5 — Group Routing

**Class:** `TestGroupAssignment`

Verifies that `_assign_groups()` routes each token to the correct reporting bucket.

| Test | Assertion |
|------|-----------|
| `test_q_after_time_delta_routes_to_time` | Q preceded by `time_delta` → group `"time"` |
| `test_value_q_routes_to_numeric` | Value Q (not time-delta context) → group `"numeric"` |
| `test_structural_tokens_masked` | `sos` → `"mask"`, `pad` → `"mask"`, `eos` → `"eos"` |

**Why it matters:** The final `bits_per_row` result reports sub-breakdowns (`time_bpr`, `numeric_bpr`, `class_bpr`, etc.).  Misrouting contaminates these breakdowns and distorts interpretation.

---

## Test 6 — Level Tokens Excluded from Density Adjustment

**Class:** `TestLevelTokens`

| Test | What it checks |
|------|---------------|
| `test_level_adj_is_zero` | Every `L` token has adjustment = 0.0 |
| `test_level_classified_correctly` | The level dataset produces `L`-type tokens |

**Why it matters:** Level tokens represent genuinely discrete values (e.g., a lab that only takes values {1, 2, 3}).  There is no bin width to divide by — the probability mass *is* the density.  Applying a density correction would produce nonsense.

---

## Test 7 — Fused vs Factored Same Density Adjustment

**Class:** `TestFusedVsFactoredAdjustment`

Encodes the same data with both factored (`<|lab|> hemoglobin <|Q2|>`) and fused (`hemoglobin::Q2`) tokenizers, then verifies that overlapping bin indices get **identical** adjustments.

**Why it matters:** Fused and factored are two ways to lay out the same information.  If the density adjustment differed between them, the metric would conflate tokenization layout with model quality.

---

## Test 8 — Integration: Adjustments at Correct Positions

**Class:** `TestDensityAdjustmentIntegration`

End-to-end tests on real encoded sequences:

| Test | What it checks |
|------|---------------|
| `test_discrete_adjustments_at_q_only` | Nonzero adjustment at Q positions; zero everywhere else |
| `test_continuous_adjustments_at_num_only` | Nonzero adjustment at NUM positions (and time-delta positions carrying continuous values) |
| `test_adjustment_values_analytically_correct` | One concrete Q token's adjustment equals $\log_2(w_k)$ computed from the token's actual bin edges |

**Why it matters:** Catches off-by-one errors, context tracking bugs, and missed token types that would silently corrupt the metric.

---

## Test 9 — Sign Convention Regression

**Class:** `TestSignConvention`

Pure arithmetic tests (no tokenizer needed) that verify the sign of the density adjustment:

| Test | Assertion |
|------|-----------|
| `test_wider_bin_costs_more_bits` | $w=10$, $p=0.5$: density bits $\approx 4.32$ > $w=1$: density bits $= 1.0$.  Wider bin → less precision → more bits. |
| `test_continuous_wider_sigma_costs_more` | Larger $\sigma$ → larger adjustment → more bits |

Both verify that the formula is `raw + adj` (not `raw - adj`).

**Why it matters:** A sign error would invert the metric's meaning — making imprecise schemes *appear better* than precise ones.  This test was added after catching exactly this bug during development.

---

## Summary

| # | Test Class | What it guarantees |
|---|-----------|-------------------|
| 1 | `TestCrossSchemeComparability` | Discrete and continuous oracles give identical density bits |
| 2 | `TestDensityAdjustmentMath` | Bin-width and Jacobian formulas are correct |
| 3 | `TestTokenClassification` | Every token gets the right type label |
| 4 | `TestNumericContext` | Time-delta vs value contexts are distinguished |
| 5 | `TestGroupAssignment` | Tokens route to correct reporting buckets |
| 6 | `TestLevelTokens` | Levels get zero density adjustment |
| 7 | `TestFusedVsFactoredAdjustment` | Layout doesn't affect adjustment |
| 8 | `TestDensityAdjustmentIntegration` | End-to-end adjustment correctness |
| 9 | `TestSignConvention` | Sign of adjustment is correct |

Together, these tests form a chain of evidence: the formulas are correct (2, 9), the classification pipeline feeds them correct inputs (3, 4, 5), edge cases are handled (6, 7), the integration works end-to-end (8), and the final result is genuinely comparable across schemes (1).
