"""
Bits-per-row metric for cross-tokenization-strategy model comparison.

Computes information-theoretic evaluation metrics that are comparable
across discrete (quantile-bin) and continuous (Gaussian) numeric
tokenization strategies.

Key idea:
  - Categorical tokens (class, concept, BPE, milestone, age, level):
    bits = −log₂ p(token)
  - Quantile-bin tokens (Q):
    density-adjusted bits = −log₂(p(Q_k) / w_k)   where w_k = bin width
  - Continuous tokens (NUM / fused-scaling):
    density-adjusted bits = −log₂ p_x(x)
                          = −(log p_z(z) + log|dz/dx|) / ln(2)
  - Fused time-delta tokens get the same treatment as Q/continuous.

All bits are aggregated per *data row* (one class token per row).

Usage::

    from dbtoken.metrics import bits_per_row
    result = bits_per_row(model, tokenizer, dataloader, model_type='discrete')
"""

import math
import re
import numpy as np
from typing import Optional, Dict, List

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

LN2 = math.log(2)


# ──────────────────────────────────────────────────────────────────────────
# Gaussian log-prob (copied from model code for portability)
# ──────────────────────────────────────────────────────────────────────────

def _gaussian_logprob(loc, scale, target):
    """Per-element Gaussian log-prob.  All tensors shape [B, T].

    At positions where target is NaN (non-numeric), returns
    log(1/sqrt(2π)) — the same constant the training loss uses.
    """
    mask = ~torch.isnan(target)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    dist = torch.distributions.normal.Normal(loc, scale)
    scalar_val = math.log(1.0 / math.sqrt(2 * math.pi))
    return torch.where(
        mask,
        dist.log_prob(safe_target),
        torch.full_like(loc, scalar_val),
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-token bits: discrete model
# ──────────────────────────────────────────────────────────────────────────

def _per_token_bits_discrete(model, xi, yi):
    """Compute per-position bits for a discrete (gpt2_v1-style) model.

    Parameters
    ----------
    model : Transformer
        Discrete next-token model with forward(idx, targets) -> (logits, loss).
    xi : Tensor[B, T]   — input token ids
    yi : Tensor[B, T]   — target token ids

    Returns
    -------
    bits : Tensor[B, T]  — −log₂ p(target) per position
    """
    logits, _ = model(xi)
    # cross_entropy with reduction='none' gives per-element nats
    bits = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        yi.view(-1),
        reduction="none",
    ).view(yi.shape) / LN2
    return bits


# ──────────────────────────────────────────────────────────────────────────
# Per-token bits: continuous model
# ──────────────────────────────────────────────────────────────────────────

def _per_token_bits_continuous(model, xi, xv, yi, yv):
    """Compute per-position bits for a continuous (multivariategpt_v2) model.

    Parameters
    ----------
    model : GPT
        Continuous model with forward(c, v, t_c, t_v).
    xi, xv : Tensor[B, T]  — input class ids and values
    yi, yv : Tensor[B, T]  — target class ids and values

    Returns
    -------
    c_bits : Tensor[B, T]  — class cross-entropy in bits
    v_bits : Tensor[B, T]  — value Gaussian neg-log-prob in bits
        (at non-numeric positions, this is zero — NOT the phantom constant)
    """
    x_c, x_v_l, x_v_s, _, _, _ = model(xi, xv, yi, yv)

    # Class cross-entropy (per position, in bits)
    c_bits = F.cross_entropy(
        x_c.view(-1, x_c.size(-1)),
        yi.view(-1),
        reduction="none",
        ignore_index=-1,
    ).view(yi.shape) / LN2

    # Value Gaussian neg-log-prob (per position, in bits)
    # x_v_l, x_v_s are [B, T] when targets are provided (training path)
    v_logprob = _gaussian_logprob(x_v_l, x_v_s, yv)
    v_bits_raw = -v_logprob / LN2

    # Zero out the phantom constant at non-numeric positions
    is_numeric = ~torch.isnan(yv)
    v_bits = torch.where(is_numeric, v_bits_raw, torch.zeros_like(v_bits_raw))

    return c_bits, v_bits


# ──────────────────────────────────────────────────────────────────────────
# Density adjustment
# ──────────────────────────────────────────────────────────────────────────

def _bin_width(edges, bin_idx):
    """Original-scale width of quantile bin *bin_idx* given *edges*.

    np.digitize(x, edges, right=False) yields:
      bin 0    : (−∞, edges[0])      — use edges[1]−edges[0] as proxy
      bin k    : [edges[k-1], edges[k])
      bin N    : [edges[-1], +∞)     — use edges[-1]−edges[-2] as proxy

    A minimum width of 1e-12 prevents log(0).
    """
    n = len(edges)
    if n <= 1:
        return 1.0  # degenerate: single edge, can't compute width
    if bin_idx <= 0:
        w = edges[1] - edges[0]
    elif bin_idx >= n:
        w = edges[-1] - edges[-2]
    else:
        w = edges[bin_idx] - edges[bin_idx - 1]
    return max(abs(w), 1e-12)


def _jacobian_log_abs(params, z_value=None):
    """Compute log|dz/dx| for the given distribution params.

    Returns a float.  For lognormal this depends on x, so we need
    the scaled value z to unscale back to x first.
    """
    dist = params.get("distribution")
    if dist == "normal":
        var = max(params.get("var", 1.0), 1e-8)
        return -0.5 * math.log(var)  # log(1/sqrt(var))
    elif dist == "lognormal":
        sigma2 = max(params.get("sigma2", 1.0), 1e-8)
        # |dz/dx| = 1/(x * sqrt(sigma2))
        # Need to unscale z -> x
        if z_value is not None:
            x = math.exp(z_value * math.sqrt(sigma2) + params.get("mu", 0.0))
            x = max(x, 1e-8)
            return -math.log(x) - 0.5 * math.log(sigma2)
        else:
            return -0.5 * math.log(sigma2)  # partial (missing -log(x))
    elif dist == "gamma":
        alpha = max(params.get("alpha", 1.0), 1e-8)
        beta = max(params.get("beta", 1.0), 1e-8)
        return -0.5 * math.log(alpha * beta ** 2)
    elif dist == "minmax":
        lo = params.get("min", 0.0)
        hi = params.get("max", 1.0)
        return -math.log(max(abs(hi - lo), 1e-8))
    return 0.0


_RE_Q_IDX = re.compile(r"Q(\d+)")
_RE_L_IDX = re.compile(r"L(\d+)")


def _density_adjustment(tokenizer, ids, vals, types, contexts):
    """Compute per-position density adjustment in bits.

    Parameters
    ----------
    tokenizer : DBTokenizer
    ids : list[int]                — token IDs (flat, length N)
    vals : list[float] | None     — parallel float values (continuous mode)
    types : list[str]             — token type labels from classify_token_ids
    contexts : list[dict | None]  — numeric param dicts from get_numeric_context

    Returns
    -------
    adj : list[float]  — additive correction in bits per position.
        For Q tokens:     +log₂(w_k)   (converts mass → density)
        For continuous:   −log₂|dz/dx|  (Jacobian correction)
        For everything else: 0.0
    """
    adj = [0.0] * len(ids)

    for i, (tid, ttype, ctx) in enumerate(zip(ids, types, contexts)):
        if ctx is None:
            continue

        tok = tokenizer.decode_token(tid)

        # --- Q token (factored or time-delta Q) ---
        if ttype == "Q":
            m = _RE_Q_IDX.search(tok)
            if m and "edges" in ctx:
                bin_idx = int(m.group(1))
                w = _bin_width(ctx["edges"], bin_idx)
                adj[i] = math.log2(w)  # mass → density: +log₂(w) in bits
            continue

        # --- Fused time-delta with bin (delta_time_ip_Q3) ---
        if ttype == "time_delta_fused":
            m = _RE_Q_IDX.search(tok)
            if m and "edges" in ctx:
                bin_idx = int(m.group(1))
                w = _bin_width(ctx["edges"], bin_idx)
                adj[i] = math.log2(w)
            continue

        # --- Fused concept+Q (hemoglobin::Q5) ---
        if ttype == "fused_concept_Q":
            m = _RE_Q_IDX.search(tok)
            if m and "edges" in ctx:
                bin_idx = int(m.group(1))
                w = _bin_width(ctx["edges"], bin_idx)
                adj[i] = math.log2(w)
            continue

        # --- Continuous NUM marker (factored) ---
        if ttype == "num_marker" and ctx.get("type") == "scaling":
            z = vals[i] if (vals is not None and not math.isnan(vals[i])) else None
            log_jac = _jacobian_log_abs(ctx, z_value=z)
            adj[i] = -log_jac / LN2  # −log₂|dz/dx|
            continue

        # --- Time-delta marker carrying a continuous value (fused) ---
        if ttype == "time_delta" and ctx is not None and ctx.get("type") == "scaling":
            z = vals[i] if (vals is not None and not math.isnan(vals[i])) else None
            if z is not None:
                log_jac = _jacobian_log_abs(ctx, z_value=z)
                adj[i] = -log_jac / LN2
            continue

        # --- Fused concept carrying a continuous value ---
        # (concept token where vals[i] is non-NaN and context is scaling)
        if ttype == "concept" and ctx is not None and ctx.get("type") == "scaling":
            z = vals[i] if (vals is not None and not math.isnan(vals[i])) else None
            if z is not None:
                log_jac = _jacobian_log_abs(ctx, z_value=z)
                adj[i] = -log_jac / LN2
            continue

    return adj


# ──────────────────────────────────────────────────────────────────────────
# Token-type grouping
# ──────────────────────────────────────────────────────────────────────────

_GROUP_MAP = {
    "sos":               "mask",       # mask from metric
    "pad":               "mask",
    "row":               "mask",
    "eos":               "eos",
    "time_delta":        "time",
    "time_delta_fused":  "time",
    "Q":                 "_context",    # Q is context-dependent (time vs numeric)
    "num_marker":        "_context",    # NUM is context-dependent
    "milestone":         "milestone",
    "age":               "age",
    "class":             "class",
    "concept":           "text",
    "bpe":               "text",
    "L":                 "level",
    "fused_concept_Q":   "fused",
    "fused_concept_L":   "fused",
}


def _assign_groups(types, contexts):
    """Assign each position to a reporting group.

    Q and NUM tokens are context-dependent: if they follow a time-delta
    marker (context is a time_delta_params entry), they go to 'time';
    otherwise they go to 'numeric'.
    """
    groups = []
    for ttype, ctx in zip(types, contexts):
        g = _GROUP_MAP.get(ttype, "text")
        if g == "_context":
            # Determine whether this is a time-delta numeric or row-level numeric
            if ctx is not None and ctx is not None:
                # time_delta_params dicts don't have a 'values' or 'value_to_idx'
                # key, but they always have 'type' in {'bins', 'scaling'}.
                # numeric_params dicts for levels have 'values'.
                # Simplest check: time_delta_params are stored under "ip"/"op"
                # and lack a "values" key.  But we just check whether ctx came
                # from time_delta_params by a duck-type check:
                # time_delta_params entries never have 'value_to_idx'.
                # This is reliable because only level-type numeric_params have it,
                # and Q/NUM are never level type.
                # Actually the cleanest: we already resolved this in
                # get_numeric_context — if ctx is a time_delta_params dict,
                # assign to 'time'; otherwise 'numeric'.
                # We can tag this by checking identity.
                pass
            groups.append("time" if _is_time_delta_ctx(ctx) else "numeric")
        else:
            groups.append(g)
    return groups


def _is_time_delta_ctx(ctx):
    """Check whether a numeric context dict originates from time_delta_params.

    get_numeric_context() tags time-delta contexts with ``_is_time_delta=True``.
    """
    if ctx is None:
        return False
    return ctx.get("_is_time_delta", False)


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def _no_grad_if_available(fn):
    """Apply @torch.no_grad() only when torch is available."""
    if HAS_TORCH:
        return torch.no_grad()(fn)
    return fn


@_no_grad_if_available
def bits_per_row(
    model,
    tokenizer,
    dataloader,
    model_type: str = "discrete",
    n_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Compute bits-per-row metric across a dataset.

    Parameters
    ----------
    model : nn.Module
        A discrete (gpt2_v1) or continuous (multivariategpt_v2) model.
    tokenizer : DBTokenizer
        Trained tokenizer (needed for token classification and density
        adjustment).
    dataloader : iterable
        Yields (Xi, Yi) for discrete or (Xi, Xv, Yi, Yv) for continuous.
    model_type : str
        ``'discrete'`` or ``'continuous'``.
    n_batches : int | None
        Max batches to evaluate.  ``None`` = all.

    Returns
    -------
    dict with keys:
        bpr             — cross-scheme comparable total (density-adjusted)
        bpr_raw         — within-scheme total (no density adjustment)
        eos_bpr         — end-of-sequence token bits / n_rows
        time_bpr        — time-delta bits (density-adjusted) / n_rows
        milestone_bpr   — milestone token bits / n_rows
        age_bpr         — age token bits / n_rows
        class_bpr       — class token bits / n_rows
        text_bpr        — concept + BPE bits / n_rows
        numeric_bpr     — density-adjusted numeric bits / n_rows
        level_bpr       — L-token bits / n_rows
        fused_bpr       — fused concept+numeric bits (density-adjusted) / n_rows
        n_rows          — total data rows (= class token count in targets)
        tokens_per_row  — average tokens per data row
    """
    model.eval()

    accum = {
        "eos": 0.0, "time": 0.0, "milestone": 0.0, "age": 0.0,
        "class": 0.0, "text": 0.0, "numeric": 0.0, "level": 0.0,
        "fused": 0.0,
    }
    accum_raw = dict(accum)  # same keys, tracks raw (no density adj)
    total_rows = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if n_batches is not None and batch_idx >= n_batches:
            break

        # ── Unpack batch ──────────────────────────────────────────────
        if model_type == "continuous":
            xi, xv, yi, yv = batch
            c_bits, v_bits = _per_token_bits_continuous(model, xi, xv, yi, yv)
            # Total per-position bits = class bits + value bits
            bits = c_bits + v_bits
        else:
            xi, yi = batch[0], batch[1]
            bits = _per_token_bits_discrete(model, xi, yi)
            yv = None

        B, T = yi.shape

        # ── Classify and get contexts (per sequence in batch) ─────────
        for b in range(B):
            yi_list = yi[b].tolist()
            yv_list = (
                yv[b].tolist() if yv is not None
                else [float("nan")] * T
            )
            bits_row = bits[b]  # Tensor[T]

            types = tokenizer.classify_token_ids(yi_list)
            contexts = tokenizer.get_numeric_context(yi_list)
            adj = _density_adjustment(tokenizer, yi_list, yv_list, types, contexts)
            groups = _assign_groups(types, contexts)

            for t in range(T):
                g = groups[t]
                if g == "mask":
                    continue

                raw_b = bits_row[t].item()
                adj_b = raw_b + adj[t]  # density-adjusted: −log₂(p/w) = raw + log₂(w)

                accum_raw[g] = accum_raw.get(g, 0.0) + raw_b
                accum[g] = accum.get(g, 0.0) + adj_b
                total_tokens += 1

                if g == "class":
                    total_rows += 1

    # ── Aggregate ─────────────────────────────────────────────────────
    n = max(total_rows, 1)

    result = {
        "bpr":           sum(accum.values()) / n,
        "bpr_raw":       sum(accum_raw.values()) / n,
        "eos_bpr":       accum["eos"] / n,
        "time_bpr":      accum["time"] / n,
        "milestone_bpr": accum["milestone"] / n,
        "age_bpr":       accum["age"] / n,
        "class_bpr":     accum["class"] / n,
        "text_bpr":      accum["text"] / n,
        "numeric_bpr":   accum["numeric"] / n,
        "level_bpr":     accum["level"] / n,
        "fused_bpr":     accum["fused"] / n,
        "n_rows":        total_rows,
        "tokens_per_row": total_tokens / n,
    }

    model.train()
    return result
