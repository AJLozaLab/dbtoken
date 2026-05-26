"""Tests for BPE vocab size edge cases: vocab < special tokens, large vocab, boundary values."""
import polars as pl
import pytest
from datetime import datetime, timedelta
from conftest import make_bpe_df, HAS_BPE
from tokenizer import DBTokenizer


@pytest.mark.skipif(not HAS_BPE, reason="rustbpe/tiktoken not installed")
class TestBPEVocabSize:

    def test_vocab_size_larger_than_needed(self):
        """Large final_vocab_size works (BPE merges fill the gap)."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=2048,
        )
        tok.train(df)

        assert tok.bpe is not None
        ids, _ = tok.encode(df)
        assert len(ids) > 0

    def test_vocab_size_very_small(self):
        """When final_vocab_size is tiny, BPE still gets at least 256 merges."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=10,  # way too small for special tokens
        )
        tok.train(df)

        # Should still work — BPE vocab clamped to min 256
        assert tok.bpe is not None
        ids, _ = tok.encode(df)
        assert len(ids) > 0

    def test_no_bpe_classes_sets_vocab_to_specials(self):
        """When no BPE classes, final_vocab_size is set to len(special tokens)."""
        rows = []
        for i in range(10):
            rows.append({
                "id": 1,
                "time": datetime(2020, 1, 1) + timedelta(hours=i),
                "class": "lab",
                "text_value": "glucose",
                "numeric_value": 100.0 + i,
            })
        df = pl.DataFrame(rows)
        tok = DBTokenizer(
            text_mode_default="concept",
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=4096,  # will be overridden
        )
        tok.train(df)

        assert tok.bpe is None
        # Vocab size should equal number of special/concept tokens, not 4096
        assert tok.final_vocab_size == len(tok.vocab)
        assert tok.final_vocab_size < 4096

    def test_vocab_contains_special_tokens(self):
        """BPE vocab includes all special tokens (Q, L, age, ms, class)."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=512,
            milestone_per_state={"default": "week"},
        )
        tok.train(df)

        # Core special tokens must be present
        for special in ["<|sos|>", "<|eos|>", "<|pad|>", "<|NUM|>",
                        "<|delta_time_0|>"]:
            assert special in tok.vocab, f"Missing special token: {special}"

        # Q-bin tokens
        for i in range(tok.n_bins + 1):
            assert f"<|Q{i}|>" in tok.vocab, f"Missing Q token: Q{i}"

        # Age tokens
        assert "<|age_0|>" in tok.vocab
        assert "<|age_120|>" in tok.vocab

    def test_bpe_and_concept_token_ids_dont_overlap(self):
        """BPE merge IDs and special/concept token IDs are distinct."""
        df = make_bpe_df()
        tok = DBTokenizer(
            text_mode_default="auto",
            text_mode_threshold=64,
            num_type="discrete",
            num_seq="factored",
            n_bins=5,
            final_vocab_size=512,
        )
        tok.train(df)

        special_ids = set(tok.vocab.values())
        # Encode a BPE text to get some BPE token IDs
        bpe_ids = tok.bpe.encode_ordinary("Patient presented with condition variant")
        for bid in bpe_ids:
            if bid in special_ids:
                # This should not happen — BPE merge IDs are below the special range
                assert False, f"BPE token ID {bid} overlaps with special token"
