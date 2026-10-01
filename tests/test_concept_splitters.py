"""Tests for prescribed per-class concept splitting."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from dbtoken import DBTokenizer

try:
    import rustbpe  # noqa: F401
    import tiktoken  # noqa: F401
    HAS_BPE = True
except ImportError:
    HAS_BPE = False


REF = datetime(2024, 1, 1)


def _frame(text_values, numeric_values=None, cls="diagnosis"):
    if numeric_values is None:
        numeric_values = [None] * len(text_values)
    return pl.DataFrame(
        [
            {
                "id": 1,
                "time": REF + timedelta(hours=i),
                "class": cls,
                "text_value": text,
                "numeric_value": value,
            }
            for i, (text, value) in enumerate(zip(text_values, numeric_values))
        ],
        schema_overrides={"numeric_value": pl.Float64},
    )


def _row_text_tokens(tok, df):
    debug = tok.encode_debug(df)
    return [
        [part for part in row if not part.startswith("<|")]
        for row in debug["tokens"].to_list()
    ]


def test_delimiter_split_and_exact_roundtrip():
    df = _frame(["aa//bb//cc", "//aa////bb//"], cls="hierarchy")
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_threshold=0,
        concept_splitters={"hierarchy": "//"},
    )
    tok.train(df)

    assert tok.class_text_modes["hierarchy"] == "concept"
    assert _row_text_tokens(tok, df) == [
        ["aa", "bb", "cc"],
        ["", "aa", "", "bb", ""],
    ]

    ids, vals, row_idx = tok.encode(df, return_row_idx=True)
    decoded = tok.decode(ids)
    assert [decoded[i] for i, source in enumerate(row_idx) if source == 0] == [
        "<|hierarchy|>", "aa", "bb", "cc",
    ]
    assert [decoded[i] for i, source in enumerate(row_idx) if source == 1][-5:] == [
        "", "aa", "", "bb", "",
    ]

    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["text_value"].to_list() == df["text_value"].to_list()


def test_fixed_width_split_examples_and_roundtrip():
    df = _frame(["E11.9", "X12.673A"])
    tok = DBTokenizer(concept_splitters={"diagnosis": (2, 1, None)})
    tok.train(df)

    assert _row_text_tokens(tok, df) == [
        ["E1", "1", ".9"],
        ["X1", "2", ".673A"],
    ]
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["text_value"].to_list() == df["text_value"].to_list()


@pytest.mark.parametrize(
    "splitter, error",
    [
        ({"diagnosis": ""}, ValueError),
        ({"diagnosis": (2, 1)}, ValueError),
        ({"diagnosis": (2, None, None)}, ValueError),
        ({"diagnosis": (0, None)}, ValueError),
        ({"diagnosis": (True, None)}, ValueError),
        ({"diagnosis": object()}, TypeError),
        ({1: "//"}, TypeError),
    ],
)
def test_invalid_splitter_specs(splitter, error):
    with pytest.raises(error):
        DBTokenizer(concept_splitters=splitter)


def test_observed_piece_vocabulary_supports_recombination():
    train_df = _frame(["aa//bb", "cc//dd"], cls="hierarchy")
    tok = DBTokenizer(concept_splitters={"hierarchy": "//"})
    tok.train(train_df)

    known_parts = _frame(["aa//dd"], cls="hierarchy")
    ids, _ = tok.encode(known_parts)
    assert "aa" in tok.decode(ids)
    assert "dd" in tok.decode(ids)

    with pytest.raises(KeyError, match="not found in vocabulary"):
        tok.encode(_frame(["aa//unseen"], cls="hierarchy"))


@pytest.mark.parametrize("num_type", ["discrete", "continuous"])
@pytest.mark.parametrize("num_seq", ["factored", "fused"])
def test_split_concepts_with_numeric_modes(num_type, num_seq):
    df = _frame(["E11.9"] * 4, [1.0, 2.0, 3.0, 4.0])
    tok = DBTokenizer(
        concept_splitters={"diagnosis": (2, 1, None)},
        num_type=num_type,
        num_seq=num_seq,
        level_threshold=0,
        n_bins=2,
        distributions={"minmax": ["diagnosis"]},
    )
    tok.train(df)
    ids, vals = tok.encode(df)
    decoded = tok.decode(ids)
    types = tok.classify_token_ids(ids)
    contexts = tok.get_numeric_context(ids)

    if num_seq == "fused" and num_type == "discrete":
        numeric_positions = [i for i, kind in enumerate(types) if kind == "fused_concept_Q"]
        assert numeric_positions
        assert all(decoded[i].startswith(".9::Q") for i in numeric_positions)
    elif num_seq == "fused":
        numeric_positions = [
            i for i, (kind, value) in enumerate(zip(types, vals))
            if kind == "concept" and not math.isnan(value)
        ]
        assert numeric_positions
        assert all(decoded[i] == ".9" for i in numeric_positions)
    else:
        expected = "Q" if num_type == "discrete" else "num_marker"
        target_context = tok.numeric_params[("diagnosis", "E11.9")]
        numeric_positions = [
            i for i, kind in enumerate(types)
            if kind == expected and contexts[i] is target_context
        ]
        assert numeric_positions

    assert all(contexts[i] is tok.numeric_params[("diagnosis", "E11.9")]
               for i in numeric_positions)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["text_value"].to_list() == df["text_value"].to_list()
    assert all(value is not None for value in recon["numeric_value"].to_list())


def test_splitter_serialization_roundtrip(tmp_path):
    df = _frame(["aa//bb//cc"], cls="hierarchy")
    tok = DBTokenizer(concept_splitters={"hierarchy": "//", "diagnosis": [2, 1, None]})
    tok.train(df)
    ids1, vals1 = tok.encode(df)

    path = str(tmp_path / "split_tok")
    tok.save(path)
    loaded = DBTokenizer.load(path)
    ids2, vals2 = loaded.encode(df)

    assert loaded.concept_splitters == {
        "hierarchy": "//",
        "diagnosis": (2, 1, None),
    }
    assert (ids2, vals2) == (ids1, vals1)


@pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
def test_explicit_pair_override_takes_precedence_and_roundtrips():
    df = _frame(["aa//bb", "free text"], cls="hierarchy")
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_overrides={("hierarchy", "free text"): "bpe"},
        concept_splitters={"hierarchy": "//"},
        final_vocab_size=512,
    )
    tok.train(df)

    assert tok.class_text_modes["hierarchy"] == "concept"
    assert tok._get_text_mode("hierarchy", "free text") == "bpe"
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["text_value"].to_list() == df["text_value"].to_list()


@pytest.mark.skipif(not HAS_BPE, reason="BPE backend required")
def test_explicit_class_override_takes_precedence():
    df = _frame(["aa//bb"], cls="hierarchy")
    tok = DBTokenizer(
        text_mode_default="auto",
        text_mode_overrides={"hierarchy": "bpe"},
        concept_splitters={"hierarchy": "//"},
        final_vocab_size=512,
    )
    tok.train(df)

    assert tok.class_text_modes["hierarchy"] == "bpe"
    ids, vals = tok.encode(df)
    recon = tok.decode_to_dataframe(ids, vals, reference_time=REF)
    assert recon["text_value"].to_list() == ["aa//bb"]
