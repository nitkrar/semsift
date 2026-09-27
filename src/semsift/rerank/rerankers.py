"""Rerankers: take candidates, return them re-scored, reordered or fewer.

A reranker never adds ids. The order it returns is the ranking. Recency
and Rules re-score and sort by score, keeping the incoming order among
equal scores; MMR reorders without re-scoring, so it goes last.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..store import filters as f


@dataclass(frozen=True)
class Candidate:
    id: int
    score: float
    text: str | None = None
    metadata: Mapping[str, Any] | None = None
    vector: tuple[float, ...] | None = None


class Reranker(Protocol):
    #: Which of "text", "metadata", "vector" the composer must fetch.
    needs: frozenset[str]

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]: ...


def _by_score(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: -c.score)


class Recency:
    """Multiplies each score by `0.5 ** (age / half_life)`.

    `field` holds UTC epoch seconds. A missing timestamp multiplies by
    `missing`; a timestamp in the future counts as age 0.
    """

    needs = frozenset({"metadata"})

    def __init__(self, field: str, *, half_life: float, missing: float = 1.0,
                 clock: Callable[[], float] = time.time) -> None:
        if not half_life > 0:
            raise ValueError("half_life must be positive")
        self.field, self.half_life, self.missing, self.clock = field, half_life, missing, clock

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        now = self.clock()
        out = []
        for c in candidates:
            at = (c.metadata or {}).get(self.field)
            factor = (self.missing if at is None
                      else 0.5 ** (max(0.0, now - at) / self.half_life))
            out.append(replace(c, score=c.score * factor))
        return _by_score(out)


@dataclass(frozen=True)
class Rule:
    """Multiply by `factor` when `when` matches a candidate's metadata."""

    when: f.Filter
    factor: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.factor) and self.factor > 0):
            raise ValueError(f"a rule factor must be finite and positive, got {self.factor}")


def _sqlite_glob(pattern: str, value: str) -> bool:
    """Match SQLite GLOB syntax, whose negated classes start with `^`."""
    pattern = pattern.split("\0", 1)[0]
    value = value.split("\0", 1)[0]

    def char_class(start: int, char: str) -> tuple[bool, int] | None:
        i = start
        inverted = i < len(pattern) and pattern[i] == "^"
        if inverted:
            i += 1
        seen = False
        previous = None
        if i < len(pattern) and pattern[i] == "]":
            seen = char == "]"
            i += 1
        while i < len(pattern) and pattern[i] != "]":
            current = pattern[i]
            if (current == "-" and previous is not None
                    and i + 1 < len(pattern) and pattern[i + 1] != "]"):
                end = pattern[i + 1]
                seen = seen or previous <= char <= end
                previous = None
                i += 2
            else:
                seen = seen or char == current
                previous = current
                i += 1
        if i == len(pattern):
            return None
        return seen != inverted, i + 1

    @lru_cache(maxsize=None)
    def match(pattern_at: int, value_at: int) -> bool:
        while pattern_at < len(pattern):
            token = pattern[pattern_at]
            if token == "*":
                while pattern_at < len(pattern) and pattern[pattern_at] == "*":
                    pattern_at += 1
                if pattern_at == len(pattern):
                    return True
                return any(match(pattern_at, i)
                           for i in range(value_at, len(value) + 1))
            if value_at == len(value):
                return False
            if token == "?":
                pattern_at += 1
                value_at += 1
                continue
            if token == "[":
                parsed = char_class(pattern_at + 1, value[value_at])
                if parsed is None or not parsed[0]:
                    return False
                pattern_at = parsed[1]
                value_at += 1
                continue
            if token != value[value_at]:
                return False
            pattern_at += 1
            value_at += 1
        return value_at == len(value)

    return match(0, 0)


def _matches(node, meta: Mapping[str, Any]) -> bool | None:
    """SQL's three-valued truth: a comparison with a missing value is None."""
    if isinstance(node, f.Cmp):
        value = meta.get(node.field)
        if value is None:
            return None
        if node.op == "glob":
            return _sqlite_glob(node.value, value)
        comparisons = {"eq": lambda: value == node.value,
                       "gt": lambda: value > node.value,
                       "gte": lambda: value >= node.value,
                       "lt": lambda: value < node.value,
                       "lte": lambda: value <= node.value}
        try:
            return comparisons[node.op]()
        except KeyError as exc:
            raise ValueError(f"unknown comparison {node.op!r}") from exc
    if isinstance(node, f.In):
        value = meta.get(node.field)
        return None if value is None else value in node.values
    if isinstance(node, f.Between):
        value = meta.get(node.field)
        return None if value is None else node.low <= value <= node.high
    if isinstance(node, f.IsNull):
        return meta.get(node.field) is None
    if isinstance(node, (f.All, f.Any_)):
        if not node.parts:
            raise ValueError("and_/or_ need at least one filter")
        parts = [_matches(p, meta) for p in node.parts]
        decisive = False if isinstance(node, f.All) else True
        if decisive in parts:
            return decisive
        return None if None in parts else not decisive
    if isinstance(node, f.Not):
        # As the store compiles it: NOT coalesce(inner, 0).
        return _matches(node.inner, meta) is not True
    if isinstance(node, f.IdSet):
        raise ValueError("an id set selects from tables; a rule reads metadata only")
    raise TypeError(f"not a filter: {node!r}")


class Rules:
    """Multiplies each score by every matching rule's factor."""

    needs = frozenset({"metadata"})

    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = tuple(rules)

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        out = []
        for c in candidates:
            factor = 1.0
            for rule in self.rules:
                if _matches(rule.when, c.metadata or {}) is True:
                    factor *= rule.factor
            out.append(replace(c, score=c.score * factor))
        return _by_score(out)


class Multiply:
    """Multiplies each score by `fn(candidate)`.

    For weights a consumer computes in code, such as penalties by path;
    `Rules` does the same from rules held as data. `needs` names the
    candidate fields `fn` reads. Each factor must be finite and positive.
    """

    def __init__(self, fn: Callable[[Candidate], float],
                 needs: frozenset[str] | set[str] = frozenset()) -> None:
        self.fn = fn
        self.needs = frozenset(needs)

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        out = []
        for c in candidates:
            factor = self.fn(c)
            if not (math.isfinite(factor) and factor > 0):
                raise ValueError(f"a factor must be finite and positive, got {factor}")
            out.append(replace(c, score=c.score * factor))
        return _by_score(out)


class MMR:
    """Maximal marginal relevance: relevance traded against redundancy.

    Picks greedily by `balance * relevance - (1 - balance) * max similarity
    to what is already picked`. Relevance is min-max normalised first, so
    it and cosine similarity share a scale whatever produced the scores.
    Scores are left as they were; only the order changes.
    """

    needs = frozenset({"vector"})

    def __init__(self, balance: float = 0.5) -> None:
        if not 0.0 <= balance <= 1.0:
            raise ValueError("balance must be in [0, 1]")
        self.balance = balance

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        import numpy as np

        if not candidates:
            return []
        if any(c.vector is None for c in candidates):
            raise ValueError("MMR needs a vector for every candidate")
        vectors = np.asarray([c.vector for c in candidates], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1.0, norms)
        scores = np.asarray([c.score for c in candidates], dtype="float64")
        span = scores.max() - scores.min()
        relevance = np.ones_like(scores) if span == 0 else (scores - scores.min()) / span

        chosen: list[int] = []
        remaining = list(range(len(candidates)))
        best_sim = np.full(len(candidates), -np.inf)
        while remaining:
            redundancy = np.where(np.isinf(best_sim), 0.0, best_sim)
            value = self.balance * relevance - (1 - self.balance) * redundancy
            pick = max(remaining, key=lambda i: (value[i], -i))
            chosen.append(pick)
            remaining.remove(pick)
            best_sim = np.maximum(best_sim, vectors @ vectors[pick])
        return [candidates[i] for i in chosen]
