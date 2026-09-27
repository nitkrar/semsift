"""Ranked lists and the fusers that merge them."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, NamedTuple, Sequence

#: The standard RRF constant. Larger values flatten the gap between
#: adjacent ranks.
RRF_K = 60


class Scored(NamedTuple):
    """One item from one source.

    `score` is higher-is-better whatever the source; `raw` is the
    source's own value, kept for display and never read by a fuser.
    """

    id: int
    score: float
    raw: float


class Evidence(NamedTuple):
    """One source's contribution to an item."""

    source: str
    rank: int
    score: float
    raw: float


@dataclass(frozen=True)
class RankedList:
    """One source's items, best first, and facts about how reliable they are."""

    source: str
    items: tuple[Scored, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.warnings, tuple)
                or any(type(warning) is not str for warning in self.warnings)):
            raise TypeError(f"{self.source}: warnings must be a tuple of strings")
        ids = [s.id for s in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.source}: an id appears twice")
        if any(not math.isfinite(s.score) for s in self.items):
            raise ValueError(f"{self.source}: scores must be finite")
        if any(not math.isfinite(s.raw) for s in self.items):
            raise ValueError(f"{self.source}: raw values must be finite")
        if any(a.score < b.score or (a.score == b.score and a.id > b.id)
               for a, b in zip(self.items, self.items[1:])):
            raise ValueError(f"{self.source}: items are not best first")


@dataclass(frozen=True)
class Fused:
    """The merged ranking, best first, and where each id came from.

    `evidence[id]` holds source, one-based rank, score and raw value for
    every input list containing the id, in input order, whether or not the
    fuser used that list.
    """

    items: tuple[tuple[int, float], ...]
    evidence: Mapping[int, tuple[Evidence, ...]] = field(default_factory=dict)


def _validate_lists(lists: Sequence[RankedList]) -> None:
    sources = [rl.source for rl in lists]
    if len(set(sources)) != len(sources):
        raise ValueError("source names must be unique")


def _weight(source: str, weights: Mapping[str, float] | None) -> float:
    weight = 1.0 if weights is None else weights.get(source, 1.0)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{source}: weight must be finite and non-negative")
    return weight


def _fused(scores: Mapping[int, float], lists: Sequence[RankedList]) -> Fused:
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("fused scores must be finite")
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    evidence: dict[int, list[Evidence]] = {}
    for rl in lists:
        for rank, s in enumerate(rl.items, start=1):
            if s.id in scores:
                evidence.setdefault(s.id, []).append(
                    Evidence(rl.source, rank, s.score, s.raw))
    return Fused(tuple(order), {k: tuple(v) for k, v in evidence.items()})


def rrf(lists: Sequence[RankedList], *, k: int = RRF_K,
        weights: Mapping[str, float] | None = None) -> Fused:
    """Reciprocal rank fusion: each list adds `weight / (k + rank)`.

    Only ranks count, so sources with incomparable scores fuse without
    normalising. A source missing from `weights` weighs 1.0; weight 0
    drops it.
    """
    _validate_lists(lists)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    scores: dict[int, float] = {}
    for rl in lists:
        w = _weight(rl.source, weights)
        if w == 0.0:
            continue
        for rank, s in enumerate(rl.items, start=1):
            scores[s.id] = scores.get(s.id, 0.0) + w / (k + rank)
    return _fused(scores, lists)


NORMALISATIONS = ("min-max", "max", "sum", "z-score")


def _normalise(values: list[float], how: str) -> list[float]:
    scale = max(abs(v) for v in values)
    if how == "min-max":
        low, high = min(values), max(values)
        if high == low:
            return [1.0] * len(values)
        scaled = [v / scale for v in values]
        low, high = min(scaled), max(scaled)
        return [(v - low) / (high - low) for v in scaled]
    if how == "max":
        return [0.0] * len(values) if scale == 0 else [v / scale for v in values]
    if how == "sum":
        if scale == 0:
            return [0.0] * len(values)
        scaled = [v / scale for v in values]
        total = sum(abs(v) for v in scaled)
        return [v / total for v in scaled]
    if scale == 0:
        return [0.0] * len(values)
    scaled = [v / scale for v in values]
    mean = sum(scaled) / len(scaled)
    std = math.sqrt(sum((v - mean) ** 2 for v in scaled) / len(scaled))
    return [0.0] * len(values) if std == 0 else [
        (v - mean) / std for v in scaled]


def blend(lists: Sequence[RankedList], *,
          weights: Mapping[str, float] | None = None,
          normalise: str = "min-max") -> Fused:
    """Weighted sum of per-source normalised scores.

    Each list is normalised on its own. An id a source did not return
    adds nothing for that source.
    """
    _validate_lists(lists)
    if normalise not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {normalise!r};"
                         f" expected one of {', '.join(NORMALISATIONS)}")
    scores: dict[int, float] = {}
    for rl in lists:
        w = _weight(rl.source, weights)
        if not rl.items or w == 0.0:
            continue
        normed = _normalise([s.score for s in rl.items], normalise)
        for s, n in zip(rl.items, normed):
            scores[s.id] = scores.get(s.id, 0.0) + w * n
    return _fused(scores, lists)


def first(lists: Sequence[RankedList]) -> Fused:
    """The first non-empty list as it stands."""
    _validate_lists(lists)
    for rl in lists:
        if rl.items:
            return _fused({s.id: s.score for s in rl.items}, lists)
    return Fused(())
