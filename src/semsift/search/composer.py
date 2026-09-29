"""The composer: sources → fuse → hydrate → rerank → hits."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..fuse import Evidence, Fused, RankedList, rrf
from ..rerank import Candidate, Reranker
from ..store import Store
from ..store import filters as f


class Source(Protocol):
    name: str

    def search(self, query: str, k: int, filter: f.Filter | None) -> RankedList: ...


class VectorSource:
    name = "vector"

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(self, query: str, k: int, filter: f.Filter | None) -> RankedList:
        return self.store.search_vector(query, k, filter)


class KeywordSource:
    name = "keyword"

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(self, query: str, k: int, filter: f.Filter | None) -> RankedList:
        return self.store.search_keyword(query, k, filter)


@dataclass(frozen=True)
class Hit:
    id: int
    score: float
    text: str
    metadata: Mapping[str, Any]
    #: The declared fields named as citation fields, e.g. a key and a span.
    citation: Mapping[str, Any]
    evidence: tuple


@dataclass(frozen=True)
class Result:
    hits: tuple[Hit, ...]
    #: Facts about a degraded result. A filter failure raises instead.
    warnings: tuple[str, ...]


class Search:
    """Runs sources, fuses them, hydrates the top candidates and reranks.

    `base_filter` is ANDed with every request's filter, given to every
    source, and applied again when candidates are hydrated, so a source
    that ignores it cannot widen the result.

    After reranking, the hits are packed for the caller's context:
    `distinct_by` keeps the best-ranked hit for each value of a declared
    field (a document key, say), leaving hits without a value ungrouped,
    and `max_chars` stops before the hit whose text would take the total
    past the budget. The first hit is always kept. `k` then counts what
    is left. Packing can leave fewer than `k` hits when the candidate
    depth holds few distinct values; raise `depth` for more.
    """

    def __init__(self, store: Store, *, sources: Sequence[Source],
                 fuser: Callable[[Sequence[RankedList]], Fused] = rrf,
                 rerankers: Sequence[Reranker] = (), depth: int | None = None,
                 base_filter: f.Filter | None = None,
                 citation: Sequence[str] = (), distinct_by: str | None = None,
                 max_chars: int | None = None) -> None:
        if (depth is not None
                and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 0)):
            raise ValueError("depth must be a non-negative integer")
        unknown = set(citation) - set(store.kinds)
        if unknown:
            raise ValueError(f"citation fields {sorted(unknown)} are not declared")
        if distinct_by is not None and distinct_by not in store.kinds:
            raise ValueError(f"distinct_by field {distinct_by!r} is not declared")
        if max_chars is not None and (isinstance(max_chars, bool)
                                      or not isinstance(max_chars, int) or max_chars <= 0):
            raise ValueError("max_chars must be a positive integer")
        self.store = store
        self.sources = tuple(sources)
        self.fuser = fuser
        self.rerankers = tuple(rerankers)
        self.depth = depth
        self.base_filter = base_filter
        self.citation = tuple(citation)
        self.distinct_by = distinct_by
        self.max_chars = max_chars

    def _filter(self, request: f.Filter | None) -> f.Filter | None:
        parts = [p for p in (self.base_filter, request) if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else f.and_(*parts)

    def run(self, query: str, k: int, filter: f.Filter | None = None) -> Result:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        flt = self._filter(filter)
        # Compiled once up front so a bad filter raises before any source
        # runs, whatever the sources do with it.
        f.compile_filter(flt, self.store.kinds)
        depth = 4 * k if self.depth is None else self.depth
        warnings: list[str] = []
        lists: list[RankedList] = []
        errors: list[Exception] = []
        for source in self.sources:
            try:
                ranked = source.search(query, depth, flt)
                lists.append(ranked)
                warnings.extend(ranked.warnings)
            except (f.FilterError, sqlite3.Error):
                # A filter or database failure may mean the scope was not
                # applied; it is never downgraded to a warning.
                raise
            except Exception as exc:
                errors.append(exc)
                warnings.append(f"source {source.name!r} failed and was skipped: {exc}")
        if errors and not lists:
            raise errors[0]

        fused = self.fuser(lists)
        self._validate_fused(fused, lists)
        top = [i for i, _ in fused.items[:depth]]
        needs = {"text", "metadata"}
        for r in self.rerankers:
            needs |= set(r.needs)
        allowed, records = self._hydrate(top, flt, needs)
        if len(allowed) < len(top):
            warnings.append(f"{len(top) - len(allowed)} candidates were outside"
                            " the filter or not in the store and were dropped")
        scores = dict(fused.items)
        candidates = [Candidate(i, scores[i], records[i].text, records[i].metadata,
                                records[i].vector)
                      for i in top if i in records]
        hydrated = len(candidates)

        for reranker in self.rerankers:
            before = {c.id: c for c in candidates}
            candidates = list(reranker.rerank(query, candidates))
            if any(not isinstance(c, Candidate) for c in candidates):
                raise ValueError(f"{type(reranker).__name__} returned a non-candidate")
            after = [c.id for c in candidates]
            if len(set(after)) != len(after) or not set(after) <= set(before):
                raise ValueError(f"{type(reranker).__name__} added or repeated ids")
            if any(not math.isfinite(c.score) for c in candidates):
                raise ValueError(f"{type(reranker).__name__} returned a non-finite score")
            if any((c.text, c.metadata, c.vector)
                   != (before[c.id].text, before[c.id].metadata, before[c.id].vector)
                   for c in candidates):
                raise ValueError(f"{type(reranker).__name__} replaced candidate fields")
        if len(candidates) < k and hydrated >= k:
            warnings.append(f"reranking left {len(candidates)} of {k} hits")
        candidates = self._pack(candidates, k, warnings)

        hits = tuple(
            Hit(c.id, c.score, records[c.id].text, records[c.id].metadata,
                {n: records[c.id].metadata.get(n) for n in self.citation},
                tuple(fused.evidence.get(c.id, ())))
            for c in candidates[:k])
        return Result(hits, tuple(warnings))

    def _pack(self, candidates: list[Candidate], k: int,
              warnings: list[str]) -> list[Candidate]:
        """The best hit per `distinct_by` value, then as many as fit `max_chars`."""
        if self.distinct_by is not None:
            seen: set = set()
            kept = []
            for c in candidates:
                value = c.metadata.get(self.distinct_by)
                if value is not None:
                    if value in seen:
                        continue
                    seen.add(value)
                kept.append(c)
            candidates = kept
        if self.max_chars is not None and k:
            total, kept = 0, []
            for c in candidates:
                if kept and total + len(c.text) > self.max_chars:
                    break
                total += len(c.text)
                kept.append(c)
            if len(kept) == 1 and total > self.max_chars:
                warnings.append(f"the first hit alone is {total} characters,"
                                f" over max_chars {self.max_chars}")
            candidates = kept
        return candidates

    @staticmethod
    def _validate_fused(fused: Fused, lists: Sequence[RankedList]) -> None:
        ids = [item_id for item_id, _ in fused.items]
        if len(set(ids)) != len(ids):
            raise ValueError("fuser repeated an id")
        if any(not math.isfinite(score) for _, score in fused.items):
            raise ValueError("fuser returned a non-finite score")
        if any(a_score < b_score or (a_score == b_score and a_id > b_id)
               for (a_id, a_score), (b_id, b_score)
               in zip(fused.items, fused.items[1:])):
            raise ValueError("fuser items are not best first")
        source_ids = {item.id for ranked in lists for item in ranked.items}
        if not set(ids) <= source_ids:
            raise ValueError("fuser added ids that no source returned")
        for item_id in ids:
            expected = tuple(
                Evidence(ranked.source, rank, item.score, item.raw)
                for ranked in lists
                for rank, item in enumerate(ranked.items, start=1)
                if item.id == item_id)
            if tuple(fused.evidence.get(item_id, ())) != expected:
                raise ValueError(f"fuser returned incorrect evidence for id {item_id}")

    def _hydrate(self, ids: Sequence[int], flt: f.Filter | None,
                 needs: set[str]) -> tuple[set[int], dict]:
        """Scope-check and fetch against one SQLite snapshot."""
        owns_transaction = not self.store.conn.in_transaction
        if owns_transaction:
            self.store.conn.execute("BEGIN")
        try:
            allowed = self._allowed(ids, flt)
            records = self.store.fetch([i for i in ids if i in allowed], needs)
            return allowed, records
        finally:
            if owns_transaction:
                self.store.conn.execute("ROLLBACK")

    def _allowed(self, ids: Sequence[int], flt: f.Filter | None) -> set[int]:
        """The subset of `ids` in the store and inside the filter."""
        extra = f.IdSet(f"SELECT value FROM json_each(?)", (_json_ids(ids),))
        return self.store.select_ids(f.and_(flt, extra) if flt is not None else extra)


def _json_ids(ids: Sequence[int]) -> str:
    import json

    return json.dumps(list(ids))
