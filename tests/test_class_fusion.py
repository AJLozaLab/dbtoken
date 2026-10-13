"""Tests for fuse_class_to_concept and namespace_concepts_by_class."""

import json
import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from dbtoken import DBTokenizer
from dbtoken.metrics import _ROW_START_TYPES, _assign_groups, _density_adjustment

try:
    import rustbpe  # noqa: F401
    import tiktoken  # noqa: F401
    HAS_BPE = True
except ImportError:
    HAS_BPE = False


REF = datetime(2024, 1, 1)


def _frame(rows):
    """rows: iterable of (class, text_value, numeric_value)."""
    return pl.DataFrame(
        [
            {
                "id": 1,
                "time": REF + timedelta(hours=i),
                "class": cls,
                "text_value": text,
                "numeric_value": value,
            }
            for i, (cls, text, value) in enumerate(rows)
        ],
        schema_overrides={"numeric_value": pl.Float64},
    )


# Same piece "XX" appears under two delimited classes and one plain class
SHARED_ROWS = [
    ("abcd", "XX//aa", 1.0),
    ("efgh", "XX//bb", 2.0),
    ("efgh", "cc//XX", 3.0),
    ("ijkl", "XX", None),
    ("abcd", "p//XX", 1.0),
    ("efgh", "q//XX", 50.0),
] * 3
SPLITTERS = {"abcd": "//", "efgh": "//"}


def _row_tokens(tok, df):
    return tok.encode_debug(df)["tokens"].to_list()


def _roundtrip(tok, df):
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["class"].to_list() == df["class"].to_list()
    assert recon["text_value"].to_list() == df["text_value"].to_list()
    return ids, vals, recon


# ── Side check: default behaviour shares pieces across classes ─────────────

def test_default_shares_pieces_across_classes():
    df = _frame(SHARED_ROWS)
    tok = DBTokenizer(
        concept_splitters=SPLITTERS, num_seq="fused",
        level_threshold=0, n_bins=2,
    )
    tok.train(df)

    rows = _row_tokens(tok, df)
    # "XX" as a leading piece in abcd and efgh, and unsplit in ijkl
    assert rows[0][-2] == rows[1][-2] == rows[3][-1] == "XX"
    # abcd p//XX=1.0 and efgh q//XX=50.0 share the same fused-numeric token
    assert rows[4][-1] == rows[5][-1] == "XX::Q0"
    assert tok.concept_token_meta == {}


# ── namespace_concepts_by_class ────────────────────────────────────────────

def test_namespace_separates_pieces_per_class():
    df = _frame(SHARED_ROWS)
    tok = DBTokenizer(
        concept_splitters=SPLITTERS, num_seq="fused",
        level_threshold=0, n_bins=2,
        namespace_concepts_by_class=True,
    )
    tok.train(df)

    rows = _row_tokens(tok, df)
    assert rows[0][-2:] == ["abcd:XX", "abcd:aa::Q0"]
    assert rows[1][-2] == "efgh:XX"
    assert rows[3][-2:] == ["<|ijkl|>", "ijkl:XX"]
    assert rows[4][-1] == "abcd:XX::Q0"
    assert rows[5][-1] == "efgh:XX::Q0"
    assert "XX" not in tok.vocab
    assert len({tok.vocab[t] for t in ("abcd:XX", "efgh:XX", "ijkl:XX")}) == 3

    _, _, recon = _roundtrip(tok, df)
    assert all(v is not None for v in recon.filter(
        pl.col("class") != "ijkl")["numeric_value"].to_list())


# ── fuse_class_to_concept ──────────────────────────────────────────────────

def test_fusion_unsplit_saves_one_token_per_concept_row():
    rows = [
        ("lab", "hemoglobin", None),
        ("lab", "glucose", None),
        ("admit", None, None),
        ("lab", "hemoglobin", None),
    ]
    df = _frame(rows)
    base = DBTokenizer()
    base.train(df)
    fused = DBTokenizer(fuse_class_to_concept=True)
    fused.train(df)

    base_ids, _ = base.encode(df)
    fused_ids, _ = fused.encode(df)
    assert len(fused_ids) == len(base_ids) - 3

    toks = _row_tokens(fused, df)
    assert toks[0][-1] == "<|lab:hemoglobin|>"
    assert "<|lab|>" not in toks[0]
    # Null-text row keeps its standalone class token
    assert toks[2][-1] == "<|admit|>"
    _roundtrip(fused, df)


def test_fusion_delimited_fuses_first_piece_only():
    df = _frame([
        ("hier", "aa//bb//cc", None),
        ("hier", "//aa////bb//", None),
    ])
    tok = DBTokenizer(concept_splitters={"hier": "//"}, fuse_class_to_concept=True)
    tok.train(df)

    toks = _row_tokens(tok, df)
    assert toks[0][-3:] == ["<|hier:aa|>", "bb", "cc"]
    # Leading empty piece stays distinct from the plain class token
    assert toks[1][-5:] == ["<|hier:|>", "aa", "", "bb", ""]
    assert "<|hier:|>" != "<|hier|>"
    _roundtrip(tok, df)


def test_fusion_row_idx_attribution():
    df = _frame([("hier", "aa//bb", None), ("hier", "cc//dd", None)])
    tok = DBTokenizer(concept_splitters={"hier": "//"}, fuse_class_to_concept=True)
    tok.train(df)
    ids, _, row_idx = tok.encode(df, return_row_idx=True)
    decoded = tok.decode(ids)
    assert [decoded[i] for i, r in enumerate(row_idx) if r == 0] == [
        "<|hier:aa|>", "bb",
    ]
    assert [decoded[i] for i, r in enumerate(row_idx) if r == 1][-2:] == [
        "<|hier:cc|>", "dd",
    ]


@pytest.mark.parametrize("namespace", [False, True])
@pytest.mark.parametrize("num_type", ["discrete", "continuous"])
@pytest.mark.parametrize("num_seq", ["factored", "fused"])
@pytest.mark.parametrize("splitter", [None, "//"])
def test_fusion_numeric_modes(namespace, num_type, num_seq, splitter):
    text = "E11//9" if splitter else "E11"
    df = _frame([("dx", text, float(v)) for v in (1, 2, 3, 4)])
    tok = DBTokenizer(
        concept_splitters={"dx": splitter} if splitter else None,
        fuse_class_to_concept=True,
        namespace_concepts_by_class=namespace,
        num_type=num_type,
        num_seq=num_seq,
        level_threshold=0,
        n_bins=2,
        distributions={"minmax": ["dx"]},
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    types = tok.classify_token_ids(ids)
    contexts = tok.get_numeric_context(ids)
    target = tok.numeric_params[("dx", text)]

    assert "<|dx|>" not in decoded
    row_starts = [i for i, t in enumerate(types) if t in _ROW_START_TYPES]
    assert len(row_starts) == df.height
    assert all(decoded[i].startswith("<|dx:E11|>") for i in row_starts)

    if num_seq == "fused" and num_type == "discrete":
        kind = "fused_concept_Q" if splitter else "class_concept_Q"
        numeric_positions = [i for i, t in enumerate(types) if t == kind]
    elif num_seq == "fused":
        kind = "concept" if splitter else "class_concept"
        numeric_positions = [
            i for i, (t, v) in enumerate(zip(types, vals))
            if t == kind and not math.isnan(v)
        ]
    else:
        kind = "Q" if num_type == "discrete" else "num_marker"
        numeric_positions = [
            i for i, t in enumerate(types)
            if t == kind and contexts[i] is target
        ]
    assert len(numeric_positions) == df.height
    assert all(contexts[i] is target for i in numeric_positions)
    if namespace and splitter:
        assert all(decoded[i].startswith("dx:9") for i in numeric_positions
                   if num_seq == "fused")

    _, _, recon = _roundtrip(tok, df)
    assert all(v is not None for v in recon["numeric_value"].to_list())

    # Metrics helpers handle the new types and match the unfused baseline
    groups = _assign_groups(types, contexts)
    assert all(groups[i] in ("text", "fused") for i in row_starts)
    adj = _density_adjustment(tok, ids, vals, types, contexts)

    base = DBTokenizer(
        concept_splitters={"dx": splitter} if splitter else None,
        num_type=num_type, num_seq=num_seq, level_threshold=0, n_bins=2,
        distributions={"minmax": ["dx"]},
    )
    base.train(df)
    b_ids, b_vals = base.encode(df)
    b_types = base.classify_token_ids(b_ids)
    b_ctx = base.get_numeric_context(b_ids)
    b_adj = _density_adjustment(base, b_ids, b_vals, b_types, b_ctx)
    b_groups = _assign_groups(b_types, b_ctx)
    for g in ("numeric", "fused", "time"):
        assert sum(a for a, gg in zip(adj, groups) if gg == g) == pytest.approx(
            sum(a for a, gg in zip(b_adj, b_groups) if gg == g)
        )


def test_both_flags_against_shared_rows():
    df = _frame(SHARED_ROWS)
    tok = DBTokenizer(
        concept_splitters=SPLITTERS, num_seq="fused",
        level_threshold=0, n_bins=2,
        fuse_class_to_concept=True, namespace_concepts_by_class=True,
    )
    tok.train(df)
    rows = _row_tokens(tok, df)
    assert rows[0][-2:] == ["<|abcd:XX|>", "abcd:aa::Q0"]
    assert rows[2][-2:] == ["<|efgh:cc|>", "efgh:XX::Q0"]
    assert rows[3][-1] == "<|ijkl:XX|>"
    _roundtrip(tok, df)


@pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
def test_fusion_keeps_class_token_for_bpe_rows():
    df = _frame([("lab", "hemoglobin", None), ("note", "free text here", None)])
    tok = DBTokenizer(
        text_mode_overrides={"note": "bpe"},
        fuse_class_to_concept=True,
        final_vocab_size=512,
    )
    tok.train(df)
    toks = _row_tokens(tok, df)
    assert toks[0][-1] == "<|lab:hemoglobin|>"
    assert "<|note|>" in toks[1]
    _roundtrip(tok, df)


def test_collision_with_class_token_raises():
    # Class "a:b" yields <|a:b|>; class "a" with value "b" would fuse to the same
    df = _frame([("a", "b", None), ("a:b", None, None)])
    tok = DBTokenizer(fuse_class_to_concept=True)
    with pytest.raises(ValueError, match="collide"):
        tok.train(df)


# ── Serialization ──────────────────────────────────────────────────────────

def test_fusion_serialization_roundtrip(tmp_path):
    df = _frame(SHARED_ROWS)
    tok = DBTokenizer(
        concept_splitters=SPLITTERS, num_seq="fused",
        level_threshold=0, n_bins=2,
        fuse_class_to_concept=True, namespace_concepts_by_class=True,
    )
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    path = str(tmp_path / "fused_tok")
    tok.save(path)
    loaded = DBTokenizer.load(path)

    assert loaded.fuse_class_to_concept and loaded.namespace_concepts_by_class
    assert loaded.concept_token_meta == tok.concept_token_meta
    ids2, vals2 = loaded.encode(df)
    assert (ids2, vals2) == (ids1, vals1)
    recon = loaded.decode_to_dataframe(ids2, vals2, reference_time=REF)
    assert recon["text_value"].to_list() == df["text_value"].to_list()


def test_load_without_new_keys(tmp_path):
    df = _frame([("lab", "hemoglobin", None)])
    tok = DBTokenizer()
    tok.train(df)
    path = tmp_path / "old_tok"
    tok.save(str(path))

    cfg_path = path.with_suffix(".json")
    cfg = json.loads(cfg_path.read_text())
    for key in ("fuse_class_to_concept", "namespace_concepts_by_class",
                "concept_token_meta"):
        cfg.pop(key)
    cfg_path.write_text(json.dumps(cfg))

    loaded = DBTokenizer.load(str(path))
    assert not loaded.fuse_class_to_concept
    assert loaded.concept_token_meta == {}
    assert loaded.encode(df) == tok.encode(df)
