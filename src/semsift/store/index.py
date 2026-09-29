"""Nearest-neighbour search over stored vectors, with deterministic ties."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Hashable, Protocol, Sequence


class Backend(Protocol):
    """A searchable set of vectors, built once per filter and store generation.

    `query` returns the best `k` as (id, cosine similarity), ordered by
    (-score, id): equal scores resolve by id, so a result never depends on
    the order rows were loaded in.
    """

    nbytes: int

    def query(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]: ...


#: Builds a backend from row ids and their (n, dims) matrix.
BackendFactory = Callable[[Sequence[int], object], Backend]


def _prepare(ids: Sequence[int], matrix):
    """(ids as int64, rows as unit-length float32, width or None when empty)."""
    import numpy as np

    id_col = np.asarray(ids, dtype="int64")
    if len(id_col) == 0:
        return id_col, np.zeros((0, 0), dtype="float32"), None
    vectors = np.asarray(matrix, dtype="float32")
    if vectors.ndim != 2 or len(id_col) != vectors.shape[0]:
        raise ValueError(f"{len(id_col)} ids for a matrix of shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return id_col, vectors / np.where(norms == 0, 1.0, norms), vectors.shape[1]


def _unit_query(vector: Sequence[float], width: int):
    import numpy as np

    q = np.asarray(vector, dtype="float32")
    if q.shape != (width,):
        raise ValueError(f"query shape {q.shape} against stored width {width}")
    norm = float(np.sqrt(np.add.reduce(q.astype("float64") ** 2)))
    return q / norm if norm else q


def _ordered(pairs) -> list[tuple[int, float]]:
    return sorted(((int(i), float(s)) for i, s in pairs), key=lambda p: (-p[1], p[0]))


class Exhaustive:
    """Exact cosine similarity against every stored vector; numpy only.

    Rows are widened to float32 and normalised once at build. A query is
    one matrix-vector product; exact at the corpus sizes semsift targets.
    """

    def __init__(self, ids: Sequence[int], matrix) -> None:
        self._ids, self._vectors, self._width = _prepare(ids, matrix)
        self.nbytes = self._vectors.nbytes + self._ids.nbytes

    def query(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        import numpy as np

        n = len(self._ids)
        if n == 0 or k <= 0:
            return []
        # Summed by numpy rather than a BLAS matrix product, whose kernels
        # differ by shape, CPU and platform, so a row's score is the same
        # wherever the index is searched.
        scores = np.add.reduce(self._vectors * _unit_query(vector, self._width), axis=1)
        if k < n - 1:
            cut = np.argpartition(-scores, k)[:k + 1]
            # Everything tied with the weakest of that set, so the id
            # tiebreak below sees the candidates a full sort would.
            pool = np.flatnonzero(scores >= scores[cut].min())
        else:
            pool = np.arange(n)
        # lexsort's last key is primary: (-score, id).
        order = pool[np.lexsort((self._ids[pool], -scores[pool]))][:k]
        return [(int(self._ids[i]), float(scores[i])) for i in order]


class HNSW:
    """Approximate search over an hnswlib graph; needs the `hnsw` extra.

    Built per filter and store generation like any backend, so it pays
    off where a filter still leaves many vectors. `m` and
    `ef_construction` shape the graph; `ef_search` trades speed for
    recall and is raised to `k` when smaller. Ties resolve by id among
    the results the graph returns.
    """

    def __init__(self, ids: Sequence[int], matrix, *, m: int = 16,
                 ef_construction: int = 200, ef_search: int = 64) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError("HNSW needs the hnsw extra: pip install 'semsift[hnsw]'") from exc
        import numpy as np

        self._ids, vectors, self._width = _prepare(ids, matrix)
        self._ef = ef_search
        self._index = None
        self.nbytes = vectors.nbytes * 2 + self._ids.nbytes
        if self._width is not None:
            # Rows are unit length, so inner product is cosine similarity.
            self._index = hnswlib.Index(space="ip", dim=self._width)
            self._index.init_index(max_elements=len(self._ids), M=m,
                                   ef_construction=ef_construction, random_seed=0)
            self._index.add_items(vectors, np.arange(len(self._ids)))

    def query(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        n = len(self._ids)
        if n == 0 or k <= 0:
            return []
        q = _unit_query(vector, self._width)
        k = min(k, n)
        self._index.set_ef(max(self._ef, k))
        labels, distances = self._index.knn_query(q, k=k)
        return _ordered((self._ids[j], 1.0 - d) for j, d in zip(labels[0], distances[0]))


class USearch:
    """Approximate search over a usearch HNSW index; needs the `usearch` extra.

    `connectivity`, `expansion_add` and `expansion_search` are usearch's
    graph degree and build- and search-time breadth. Ties resolve by id
    among the results the index returns.
    """

    def __init__(self, ids: Sequence[int], matrix, *, connectivity: int = 16,
                 expansion_add: int = 128, expansion_search: int = 64) -> None:
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise ImportError("USearch needs the usearch extra:"
                              " pip install 'semsift[usearch]'") from exc
        import numpy as np

        self._ids, vectors, self._width = _prepare(ids, matrix)
        self._index = None
        self.nbytes = vectors.nbytes * 2 + self._ids.nbytes
        if self._width is not None:
            self._index = Index(ndim=self._width, metric="ip", dtype="f32",
                                connectivity=connectivity, expansion_add=expansion_add,
                                expansion_search=expansion_search)
            self._index.add(np.arange(len(self._ids), dtype="uint64"), vectors)

    def query(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        n = len(self._ids)
        if n == 0 or k <= 0:
            return []
        matches = self._index.search(_unit_query(vector, self._width), min(k, n))
        return _ordered((self._ids[int(j)], 1.0 - float(d))
                        for j, d in zip(matches.keys, matches.distances))


def build(ids: Sequence[int], matrix, backend: BackendFactory = Exhaustive) -> Backend:
    """A backend over `matrix` rows named by `ids`; `matrix` may be float16."""
    return backend(ids, matrix)


def query(index: Backend, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
    return index.query(vector, k)


class Cache:
    """Built indexes by filter key, least recently used first out.

    An entry holds the store generation it was built at and is valid for
    that generation only.
    """

    def __init__(self, max_entries: int = 8, max_bytes: int = 256 << 20) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[int, Backend]] = OrderedDict()

    def get(self, key: Hashable, generation: int) -> Backend | None:
        entry = self._entries.get(key)
        if entry is None or entry[0] != generation:
            return None
        self._entries.move_to_end(key)
        return entry[1]

    def put(self, key: Hashable, generation: int, index: Backend) -> None:
        if index.nbytes > self.max_bytes:
            return
        self._entries[key] = (generation, index)
        self._entries.move_to_end(key)
        while (len(self._entries) > self.max_entries
               or sum(e[1].nbytes for e in self._entries.values()) > self.max_bytes):
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
