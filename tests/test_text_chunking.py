from pathlib import Path

import pytest

from mlx_pocket_tts.conditioners import SentencePieceTokenizer
from mlx_pocket_tts.text_chunking import split_into_best_sentences


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(__file__).parents[1] / "models/english-public-mlx/tokenizer.model"
    if not path.exists():
        pytest.skip("converted tokenizer is not present")
    return SentencePieceTokenizer(4000, path)


def split(tokenizer, text, max_tokens=50):
    return split_into_best_sentences(tokenizer, text, max_tokens, False, False)


def test_long_comma_text_preserves_content(tokenizer):
    text = (
        "It was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness, it was the spring of hope, it was the winter of despair"
    )
    chunks = split(tokenizer, text, 20)
    assert len(chunks) > 1
    rejoined = " ".join(chunks).lower()
    assert "best of times" in rejoined
    assert "winter of despair" in rejoined


def test_decimals_are_not_split(tokenizer):
    chunks = split(tokenizer, "Version 2.0 is out. Pi is 3.14.", 50)
    assert "2.0" in chunks[0]
    assert "3.14" in chunks[0]


def test_empty_text_raises(tokenizer):
    with pytest.raises(ValueError, match="empty"):
        split(tokenizer, "")


def test_semicolon_removal(tokenizer):
    chunks = split_into_best_sentences(tokenizer, "One; two; three.", 50, False, True)
    assert ";" not in " ".join(chunks)
