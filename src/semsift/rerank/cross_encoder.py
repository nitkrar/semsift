"""A cross-encoder: rescores the top candidates by reading query and passage together."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from typing import Callable, Sequence

from .rerankers import Candidate

#: Pairs per inference call.
_BATCH = 32


class CrossEncoder:
    """Rescores the top `top` candidates with a model that reads each
    (query, passage) pair together and returns one relevance logit.

    A candidate's new score is `weight * sigmoid(logit)` plus
    `(1 - weight)` times its incoming score, min-max normalised over the
    top; equal pair scores keep the incoming order. Candidates beyond `top`
    are not returned: their incoming scores are not on the same scale.

    The model runs through onnxruntime (the `onnx` extra) from a Hugging
    Face repository holding `tokenizer.json` and an ONNX graph. `scorer`
    replaces the model with any function from (query, passage) pairs to
    logits. Logits are cached per (model, query, passage), up to
    `cache_size` pairs.

    `providers` defaults to CPU: on Apple silicon CoreML takes only part of
    this graph and runs about three times slower than CPU.
    """

    needs = frozenset({"text"})

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L6-v2", *,
                 top: int = 30, weight: float = 1.0, cache_size: int = 4096,
                 local_only: bool = False, filename: str = "onnx/model.onnx",
                 providers: str = "cpu",
                 scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None
                 ) -> None:
        if isinstance(top, bool) or not isinstance(top, int) or top <= 0:
            raise ValueError("top must be a positive integer")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self.model, self.top, self.weight, self.cache_size = model, top, weight, cache_size
        self._cache: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        self._scorer = scorer or _OnnxPairs(model, local_only=local_only,
                                            filename=filename, providers=providers)

    def logits(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Raw relevance logits for (query, passage) pairs, cached."""
        out: list[float | None] = [self._cache_get((self.model, q, p)) for q, p in pairs]
        todo = [n for n, v in enumerate(out) if v is None]
        if todo:
            scored = list(self._scorer([pairs[n] for n in todo]))
            if len(scored) != len(todo):
                raise ValueError(f"{len(scored)} scores for {len(todo)} pairs")
            for n, s in zip(todo, scored):
                out[n] = float(s)
                self._cache_put((self.model, *pairs[n]), float(s))
        return [float(v) for v in out]

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        top = list(candidates)[:self.top]
        if not top:
            return []
        if any(c.text is None for c in top):
            raise ValueError("CrossEncoder needs candidate text")
        logits = self.logits([(query, c.text) for c in top])
        low, high = min(c.score for c in top), max(c.score for c in top)
        norm = [1.0 if high == low else (c.score - low) / (high - low) for c in top]
        scored = [replace(c, score=self.weight * _sigmoid(x) + (1.0 - self.weight) * n)
                  for c, x, n in zip(top, logits, norm)]
        return sorted(scored, key=lambda c: -c.score)

    def _cache_get(self, key):
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_put(self, key, value: float) -> None:
        if not self.cache_size:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class _OnnxPairs:
    """(query, passage) pairs to logits through an ONNX sequence classifier."""

    def __init__(self, repo_id: str, *, local_only: bool, filename: str, providers: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:      # pragma: no cover - environment
            raise ImportError(
                "CrossEncoder needs the onnx extra: pip install 'semsift[onnx]'"
            ) from exc
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        from ..embed.backends import _session

        get = lambda f: hf_hub_download(repo_id, f, local_files_only=local_only)  # noqa: E731
        self._tok = Tokenizer.from_file(get("tokenizer.json"))
        self._tok.enable_truncation(512)
        self._tok.enable_padding()
        self._sess = _session(ort, get(filename), providers)
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import numpy as np

        out: list[float] = []
        for i in range(0, len(pairs), _BATCH):
            enc = self._tok.encode_batch(list(pairs[i:i + _BATCH]))
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            logits = self._sess.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            out.extend(float(row[0]) for row in logits)
        return out
