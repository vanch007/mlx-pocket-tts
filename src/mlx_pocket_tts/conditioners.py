import logging
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece
import tokenizers

from .utils import download_if_necessary

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    tokens: mx.array


class SentencePieceTokenizer:
    def __init__(self, n_bins: int, tokenizer_path: str) -> None:
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        if n_bins != self.sp.vocab_size():
            raise ValueError(
                f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} "
                f"but n_bins={n_bins} was specified"
            )

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def __call__(self, text: str) -> TokenizedText:
        return TokenizedText(mx.array(self.encode(text), dtype=mx.int32)[None, :])


class JsonTokenizer:
    """A `tokenizer.json` (HuggingFace tokenizers) holding the SentencePiece vocabulary.

    The released v3.3+ configs point at this format (`tokenizer: tokenizers`); it yields the same
    ids as the SentencePiece model without the sentencepiece runtime.
    """

    def __init__(self, n_bins: int, tokenizer_path: str) -> None:
        logger.info("Loading tokenizers json from %s", tokenizer_path)
        local_path = download_if_necessary(tokenizer_path)
        self.tokenizer = tokenizers.Tokenizer.from_file(str(local_path))
        vocab_size = self.tokenizer.get_vocab_size()
        if n_bins != vocab_size:
            raise ValueError(f"tokenizer has vocab size={vocab_size} but n_bins={n_bins} was specified")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def __call__(self, text: str) -> TokenizedText:
        return TokenizedText(mx.array(self.encode(text), dtype=mx.int32)[None, :])


def build_tokenizer(n_bins: int, tokenizer_path: str, kind: str = "sentencepiece"):
    """Pick the backend from the config's `tokenizer` field, else from the file suffix."""
    if kind in ("tokenizers", "json") or str(tokenizer_path).split("@")[0].endswith(".json"):
        return JsonTokenizer(n_bins, tokenizer_path)
    return SentencePieceTokenizer(n_bins, tokenizer_path)


class LUTConditioner(nn.Module):
    def __init__(self, n_bins: int, tokenizer_path: str, dim: int, output_dim: int, tokenizer_kind: str = "sentencepiece"):
        super().__init__()
        self.tokenizer = build_tokenizer(n_bins, tokenizer_path, tokenizer_kind)
        self.dim = dim
        self.output_dim = output_dim
        self.embed = nn.Embedding(n_bins + 1, dim)
        self.output_proj = None if dim == output_dim else nn.Linear(dim, output_dim, bias=False)

    def prepare(self, text: str) -> TokenizedText:
        return self.tokenizer(text)

    def __call__(self, inputs: TokenizedText) -> mx.array:
        embeds = self.embed(inputs.tokens)
        if self.output_proj is not None:
            embeds = self.output_proj(embeds)
        return embeds
