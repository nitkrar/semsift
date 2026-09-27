"""Nearest-neighbour search through vicinity, with deterministic ties."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass
class VectorIndex:
    ids: tuple[int, ...]
    vicinity: object
    nbytes: int


def build(ids: Sequence[int], matrix) -> VectorIndex:
    """An index over `matrix` rows named by `ids`; `matrix` may be float16."""
    import numpy as np
    from vicinity import Vicinity

    vectors = np.asarray(matrix, dtype="float32")
    vic = Vicinity.from_vectors_and_items(vectors, list(ids),
                                          backend_type="basic", metric="cosine")
    return VectorIndex(tuple(ids), vic, vectors.nbytes)


def query(index: VectorIndex, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
    """The best `k` as (id, cosine similarity), ordered by (-score, id).

    vicinity chooses arbitrarily among equal distances at its cut, so the
    fetch widens until the distance at `k` differs from the distance at
    the cut, or covers every row. Every row tied with the k-th is then in
    the fetched set, and sorting by id makes the choice deterministic.
    """
    import numpy as np

    n = len(index.ids)
    if n == 0 or k <= 0:
        return []
    q = np.asarray([vector], dtype="float32")
    fetch = min(n, 2 * k)
    while True:
        hits = index.vicinity.query(q, k=fetch)[0]
        if fetch >= n or len(hits) < fetch or hits[min(k, len(hits)) - 1][1] != hits[-1][1]:
            break
        fetch = min(n, fetch * 2)
    scored = sorted(((int(i), 1.0 - float(d)) for i, d in hits),
                    key=lambda p: (-p[1], p[0]))
    return scored[:k]


class Cache:
    """Built indexes by filter key, least recently used first out.

    An entry holds the store generation it was built at and is valid for
    that generation only.
    """

    def __init__(self, max_entries: int = 8, max_bytes: int = 256 << 20) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[int, VectorIndex]] = OrderedDict()

    def get(self, key: Hashable, generation: int) -> VectorIndex | None:
        entry = self._entries.get(key)
        if entry is None or entry[0] != generation:
            return None
        self._entries.move_to_end(key)
        return entry[1]

    def put(self, key: Hashable, generation: int, index: VectorIndex) -> None:
        if index.nbytes > self.max_bytes:
            return
        self._entries[key] = (generation, index)
        self._entries.move_to_end(key)
        while (len(self._entries) > self.max_entries
               or sum(e[1].nbytes for e in self._entries.values()) > self.max_bytes):
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
