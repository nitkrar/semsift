"""One SQLite store holding items, their vectors and a keyword index."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..embed.space import VectorSpace
from ..fuse import RankedList, Scored
from . import index as vindex
from .codec import VECTOR_DTYPE, pack, unpack
from .filters import Filter, compile_filter

_IDENT = re.compile(r"[a-z][a-z0-9_]*\Z")
_KINDS = {"int": "INTEGER", "float": "REAL", "text": "TEXT", "bool": "INTEGER"}
_RESERVED = {"id", "text", "keyword_text", "extra"}
_TOKENIZERS = {"unicode61": "unicode61", "porter": "porter unicode61",
               "trigram": "trigram"}
#: SQLite's default host-parameter limit is 999 on older builds.
_BATCH = 500
_MIN_ID = -(1 << 63)
_MAX_ID = (1 << 63) - 1


@dataclass(frozen=True)
class Field:
    """Declared metadata: filterable, typed, returned on fetch."""

    name: str
    kind: str
    indexed: bool = False


@dataclass(frozen=True)
class Item:
    """What a consumer stores. `keyword_text` defaults to `text`."""

    id: int
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    keyword_text: str | None = None


@dataclass(frozen=True)
class Vectors:
    """Vectors for items, in item order, with the space they were made in."""

    space: VectorSpace
    rows: tuple[tuple[float, ...], ...]
    #: The canary texts' vectors from the same encoder; `embed` fills it.
    canary: tuple[tuple[float, ...], ...] | None = None


#: Fixed texts whose vectors are recorded with a store's first vectors.
#: Re-encoding them later shows whether the encoder still produces the
#: same outputs, which its identity alone cannot.
CANARY_TEXTS = ("The landlord renewed the lease; rent is due on the first.",
                "def add(a, b):\n    return a + b",
                "when is the next dentist appointment")


class StaleVectors(RuntimeError):
    """The encoder's outputs moved too far from the stored vectors to rank by."""


@dataclass(frozen=True)
class Health:
    """How far the encoder has moved from the stored vectors.

    `state` is `ok`, `drifted`, `stale`, `unchecked` (no canaries or no
    encoder) or `empty`. `similarity` is the lowest canary cosine.
    """

    state: str
    similarity: float | None
    embedded_at: float | None


@dataclass(frozen=True)
class Reembedding:
    """New vectors for every item, from `prepare_reembed`, for `apply_reembed`."""

    generation: int
    space: VectorSpace
    ids: tuple[int, ...]
    rows: tuple[tuple[float, ...], ...]
    canary: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Record:
    """What `fetch` returns; a field not asked for is None."""

    id: int
    text: str | None = None
    metadata: dict | None = None
    vector: tuple[float, ...] | None = None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _space_json(space: VectorSpace | None) -> str | None:
    return None if space is None else json.dumps(asdict(space), sort_keys=True)


def _space_from(text: str | None) -> VectorSpace | None:
    return None if text is None else VectorSpace(**json.loads(text))


class Store:
    """Items, vectors and an FTS5 keyword index in the caller's database.

    The store never commits. Writes require an open transaction and join
    it, so the caller's own rows commit or roll back with them.
    """

    def __init__(self, conn: sqlite3.Connection, prefix: str,
                 fields: Sequence[Field], *, encoder=None,
                 tokenizer: str = "unicode61", cache: vindex.Cache | None = None,
                 drift_warn: float = 0.9999, drift_stale: float = 0.99) -> None:
        if not isinstance(prefix, str) or not _IDENT.match(prefix):
            raise ValueError(f"prefix {prefix!r} must match [a-z][a-z0-9_]*")
        if not isinstance(tokenizer, str) or tokenizer not in _TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {sorted(_TOKENIZERS)}")
        fields = tuple(fields)
        for f in fields:
            if not isinstance(f, Field):
                raise TypeError(f"fields must contain Field values; got {f!r}")
            if (not isinstance(f.name, str) or not _IDENT.match(f.name)
                    or f.name in _RESERVED):
                raise ValueError(f"field name {f.name!r} is not allowed")
            if not isinstance(f.kind, str) or f.kind not in _KINDS:
                raise ValueError(f"field {f.name!r} has unknown kind {f.kind!r}")
            if type(f.indexed) is not bool:
                raise TypeError(f"field {f.name!r} indexed must be bool")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError("field names repeat")
        if not 0.0 <= drift_stale <= drift_warn <= 1.0:
            raise ValueError("need 0 <= drift_stale <= drift_warn <= 1")
        self.drift_warn, self.drift_stale = drift_warn, drift_stale
        self.conn = conn
        self.prefix = prefix
        self.fields = tuple(fields)
        self.kinds = {f.name: f.kind for f in fields}
        self.encoder = encoder
        self.tokenizer = tokenizer
        self._cache = cache or vindex.Cache()
        self._cache_namespace = object()
        self._t = {name: f'"{prefix}_{name}"' for name in ("meta", "items", "vectors", "fts")}
        self._declaration = json.dumps(
            {"fields": [asdict(f) for f in fields], "tokenizer": tokenizer})
        if self._table_exists():
            stored = conn.execute(f"SELECT declaration FROM {self._t['meta']}").fetchone()[0]
            if stored != self._declaration:
                raise ValueError(f"store {prefix!r} was created with {stored}")
        else:
            self._create(_TOKENIZERS[tokenizer])

    # -- schema -----------------------------------------------------------

    def _table_exists(self) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (f"{self.prefix}_meta",)).fetchone() is not None

    def _create(self, tokenize: str) -> None:
        # One statement at a time: executescript would commit whatever
        # transaction the caller has open.
        t = self._t
        cols = "".join(f', "{f.name}" {_KINDS[f.kind]}' for f in self.fields)
        statements = [
            f"CREATE TABLE {t['meta']} (id INTEGER PRIMARY KEY CHECK (id = 1),"
            " declaration TEXT NOT NULL, space TEXT, canary TEXT, embedded_at REAL,"
            " generation INTEGER NOT NULL DEFAULT 0)",
            f"INSERT INTO {t['meta']} (id, declaration) VALUES (1, ?)",
            f"CREATE TABLE {t['items']} (id INTEGER PRIMARY KEY, text TEXT NOT NULL,"
            f" keyword_text TEXT NOT NULL{cols}, extra TEXT NOT NULL DEFAULT '{{}}')",
            f"CREATE TABLE {t['vectors']} (id INTEGER PRIMARY KEY, vec BLOB NOT NULL)",
            f"CREATE VIRTUAL TABLE {t['fts']} USING fts5(keyword_text,"
            f" tokenize='{tokenize}')",
        ]
        statements += [
            f'CREATE INDEX "{self.prefix}_items_{f.name}" ON {t["items"]} ("{f.name}")'
            for f in self.fields if f.indexed]
        savepoint = f'"semsift_{self.prefix}_schema"'
        self.conn.execute(f"SAVEPOINT {savepoint}")
        try:
            for sql in statements:
                self.conn.execute(
                    sql, (self._declaration,) if "VALUES (1, ?)" in sql else ())
        except BaseException:
            self.conn.execute(f"ROLLBACK TO {savepoint}")
            self.conn.execute(f"RELEASE {savepoint}")
            raise
        self.conn.execute(f"RELEASE {savepoint}")

    # -- state ------------------------------------------------------------

    @property
    def space(self) -> VectorSpace | None:
        """The space of the stored vectors; None while the store is empty."""
        return _space_from(self.conn.execute(
            f"SELECT space FROM {self._t['meta']}").fetchone()[0])

    def _generation(self) -> int:
        return self.conn.execute(f"SELECT generation FROM {self._t['meta']}").fetchone()[0]

    def _check_space(self) -> None:
        stored = self.space
        if stored is not None and self.encoder is not None and self.encoder.space != stored:
            raise ValueError(f"store {self.prefix!r} holds vectors in {stored};"
                             f" the encoder writes {self.encoder.space}")

    def _require_transaction(self) -> None:
        if not self.conn.in_transaction:
            raise RuntimeError("writes need an open transaction; the store never commits")

    def _bump(self, space: VectorSpace | None = None, *, reset_space: bool = False,
              canary: tuple | None = None) -> None:
        sets = "generation = generation + 1"
        params: list = []
        if reset_space:
            sets += ", space = NULL, canary = NULL, embedded_at = NULL"
        elif space is not None:
            # The first vectors fix the space, their canaries and the time.
            sets += (", embedded_at = CASE WHEN space IS NULL THEN ? ELSE embedded_at END"
                     ", canary = CASE WHEN space IS NULL THEN ? ELSE canary END"
                     ", space = coalesce(space, ?)")
            params = [time.time(), None if canary is None else json.dumps(canary),
                      _space_json(space)]
        self.conn.execute(f"UPDATE {self._t['meta']} SET {sets}", params)
        self._cache.clear()

    # -- writes -----------------------------------------------------------

    def embed(self, items: Sequence[Item]) -> Vectors:
        """Encode `items` with the store's encoder. Touches no table."""
        if self.encoder is None:
            raise RuntimeError(f"store {self.prefix!r} has no encoder")
        self._check_space()
        rows = self.encoder.encode([it.text for it in items])
        canary = self._canary() if self.space is None else None
        return Vectors(self.encoder.space, tuple(tuple(map(float, r)) for r in rows), canary)

    def _canary(self, pool: ThreadPoolExecutor | None = None) -> tuple[tuple[float, ...], ...]:
        """Each canary text's vector from `encode`.

        One text per call: an encoder's output can depend on what else
        shares its batch, and the check must compare like with like.
        """
        calls = [(self.encoder.encode, t) for t in CANARY_TEXTS]
        if pool is None:
            results = [encode([text]) for encode, text in calls]
        else:
            futures = [pool.submit(encode, [text]) for encode, text in calls]
            results = [f.result() for f in futures]
        canary = []
        for rows in results:
            if len(rows) != 1:
                raise ValueError(f"{len(rows)} vectors for 1 canary text")
            canary.append(tuple(map(float, rows[0])))
        out = tuple(canary)
        self._blobs_for(self.encoder.space, out)
        return out

    def upsert(self, items: Sequence[Item], vectors: Vectors | None = None) -> None:
        """Write items and their vectors, replacing rows with the same ids."""
        self._require_transaction()
        self._check_space()
        items = tuple(items)
        blobs = self._blobs(items, vectors)
        prepared = [self._prepare(item) for item in items]
        if not items:
            return
        t = self._t
        names = [f.name for f in self.fields]
        cols = "".join(f', "{n}"' for n in names)
        marks = ", ?" * len(names)
        updates = "".join(f', "{n}" = excluded."{n}"' for n in names)
        for i, (item, (declared, extra, keyword)) in enumerate(zip(items, prepared)):
            self.conn.execute(
                f"INSERT INTO {t['items']} (id, text, keyword_text{cols}, extra)"
                f" VALUES (?, ?, ?{marks}, ?) ON CONFLICT(id) DO UPDATE SET"
                f" text = excluded.text, keyword_text = excluded.keyword_text{updates},"
                " extra = excluded.extra",
                (item.id, item.text, keyword, *declared, extra))
            self.conn.execute(f"DELETE FROM {t['fts']} WHERE rowid = ?", (item.id,))
            self.conn.execute(f"INSERT INTO {t['fts']} (rowid, keyword_text) VALUES (?, ?)",
                              (item.id, keyword))
            if blobs is not None:
                self.conn.execute(
                    f"INSERT INTO {t['vectors']} (id, vec) VALUES (?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET vec = excluded.vec",
                    (item.id, blobs[i]))
            else:
                self.conn.execute(f"DELETE FROM {t['vectors']} WHERE id = ?", (item.id,))
        reset_space = blobs is None and not self._has_vectors()
        self._bump(vectors.space if vectors is not None else None,
                   reset_space=reset_space,
                   canary=vectors.canary if vectors is not None else None)

    def _blobs(self, items: Sequence[Item], vectors: Vectors | None) -> list[bytes] | None:
        if vectors is None:
            if self.encoder is not None:
                raise ValueError("this store embeds; pass the vectors from embed()")
            return None
        if len(vectors.rows) != len(items):
            raise ValueError(f"{len(vectors.rows)} vectors for {len(items)} items")
        expected = self.space or (self.encoder.space if self.encoder else None)
        if expected is not None and vectors.space != expected:
            raise ValueError(f"vectors are in {vectors.space}; the store needs {expected}")
        if vectors.canary is not None:
            if len(vectors.canary) != len(CANARY_TEXTS):
                raise ValueError(
                    f"{len(vectors.canary)} canary vectors for {len(CANARY_TEXTS)} texts")
            self._blobs_for(vectors.space, vectors.canary)
        return self._blobs_for(vectors.space, vectors.rows)

    @staticmethod
    def _blobs_for(space: VectorSpace, rows) -> list[bytes]:
        for row in rows:
            if space.dims and len(row) != space.dims:
                raise ValueError(f"a vector of width {len(row)} in a space of"
                                 f" width {space.dims}")
        return [pack(row) for row in rows]

    def _prepare(self, item: Item) -> tuple[list, str, str]:
        from .filters import _value

        if type(item.id) is not int:
            raise TypeError(f"item id must be int; got {item.id!r}")
        if not _MIN_ID <= item.id <= _MAX_ID:
            raise ValueError(f"item id {item.id} is outside SQLite's integer range")
        if type(item.text) is not str:
            raise TypeError(f"item text must be str; got {item.text!r}")
        if item.keyword_text is not None and type(item.keyword_text) is not str:
            raise TypeError(f"item keyword_text must be str or None; got {item.keyword_text!r}")
        if not isinstance(item.metadata, Mapping):
            raise TypeError(f"item metadata must be a mapping; got {item.metadata!r}")
        if any(type(name) is not str for name in item.metadata):
            raise TypeError("metadata keys must be strings")
        declared = []
        for f in self.fields:
            value = item.metadata.get(f.name)
            declared.append(None if value is None else _value(f.kind, f.name, value))
        extra = {k: v for k, v in item.metadata.items() if k not in self.kinds}
        encoded = json.dumps(extra, sort_keys=True, allow_nan=False)
        keyword = item.text if item.keyword_text is None else item.keyword_text
        return declared, encoded, keyword

    def _has_vectors(self) -> bool:
        return self.conn.execute(
            f"SELECT 1 FROM {self._t['vectors']} LIMIT 1").fetchone() is not None

    def remove(self, ids: Iterable[int]) -> None:
        """Delete items, their vectors and their keyword rows."""
        self._require_transaction()
        self._check_space()
        ids = list(ids)
        for item_id in ids:
            if type(item_id) is not int:
                raise TypeError(f"item id must be int; got {item_id!r}")
            if not _MIN_ID <= item_id <= _MAX_ID:
                raise ValueError(f"item id {item_id} is outside SQLite's integer range")
        if not ids:
            return
        for start in range(0, len(ids), _BATCH):
            chunk = ids[start:start + _BATCH]
            marks = ", ".join("?" * len(chunk))
            for table, key in (("items", "id"), ("vectors", "id"), ("fts", "rowid")):
                self.conn.execute(
                    f"DELETE FROM {self._t[table]} WHERE {key} IN ({marks})", chunk)
        self._bump(reset_space=not self._has_vectors())

    def clear(self) -> None:
        """Delete everything and forget the vector space, to re-embed."""
        self._require_transaction()
        self._check_space()
        for table in ("items", "vectors", "fts"):
            self.conn.execute(f"DELETE FROM {self._t[table]}")
        self._bump(reset_space=True)

    # -- searches ---------------------------------------------------------

    def search_keyword(self, query: str, k: int, filter: Filter | None = None) -> RankedList:
        """FTS5 BM25, best first. `raw` is bm25(), where lower is better."""
        self._check_space()
        where = compile_filter(filter, self.kinds)
        if k <= 0:
            return RankedList("keyword", ())
        if self.tokenizer == "trigram":
            terms = query.replace("\x00", " ").split()
            if not terms:
                return RankedList("keyword", ())
            match = " OR ".join(
                '"' + term.replace('"', '""') + '"' for term in terms)
        else:
            words = re.findall(r"\w+", query)
            if not words:
                return RankedList("keyword", ())
            # Each word quoted, so FTS5 operators in the query are plain words.
            match = " OR ".join(f'"{w}"' for w in words)
        t = self._t
        rows = self.conn.execute(
            f"SELECT f.rowid, bm25({t['fts']}) FROM {t['fts']} f"
            f" JOIN {t['items']} i ON i.id = f.rowid"
            f" WHERE {t['fts']} MATCH ? AND ({where.sql})"
            f" ORDER BY bm25({t['fts']}), f.rowid LIMIT ?",
            (match, *where.params, k)).fetchall()
        return RankedList("keyword", tuple(Scored(i, -r, r) for i, r in rows))

    def search_vector(self, query: str, k: int, filter: Filter | None = None) -> RankedList:
        """Cosine similarity to `query` encoded with `encode_query`, best first."""
        if self.encoder is None:
            raise RuntimeError(f"store {self.prefix!r} has no encoder")
        self._check_space()
        where = compile_filter(filter, self.kinds)
        space = self.space
        if space is None:
            return RankedList("vector", ())
        # The canary check and the query encode run together: each search
        # re-checks the encoder, which a model swapped behind an unchanged
        # identity can fail at any time, without waiting for it first.
        meta = self._meta()
        with ThreadPoolExecutor(max_workers=1 + len(CANARY_TEXTS)) as pool:
            query_future: Future = pool.submit(self.encoder.encode_query, [query])
            health = self._assess(meta, pool)
            if health.state == "stale":
                query_future.cancel()
                raise StaleVectors(
                    f"store {self.prefix!r}: the encoder's outputs moved from the stored"
                    f" vectors (canary similarity {health.similarity:.4f}); re-embed needed")
            query_rows = query_future.result()
            if len(query_rows) != 1:
                raise ValueError(f"{len(query_rows)} vectors for 1 query text")
            vector = query_rows[0]
        if space.dims and len(vector) != space.dims:
            raise ValueError(f"query width {len(vector)}; stored width {space.dims}")
        warnings = ()
        if health.state == "drifted":
            warnings = (f"store {self.prefix!r}: the encoder's outputs drifted from the"
                        f" stored vectors (canary similarity {health.similarity:.4f});"
                        " re-embed recommended",)
        if k <= 0:
            return RankedList("vector", (), warnings)
        idx = self._index(where)
        hits = vindex.query(idx, vector, k)
        return RankedList("vector", tuple(Scored(i, s, s) for i, s in hits), warnings)

    def health(self) -> Health:
        """Compare the encoder's canary vectors, both sides, with the stored ones."""
        self._check_space()
        with ThreadPoolExecutor(max_workers=len(CANARY_TEXTS)) as pool:
            return self._assess(self._meta(), pool)

    def _meta(self) -> tuple:
        # Read on the calling thread: a connection is not shared with the pool.
        return self.conn.execute(
            f"SELECT space, canary, embedded_at FROM {self._t['meta']}").fetchone()

    def _assess(self, meta: tuple, pool: ThreadPoolExecutor) -> Health:
        space, canary, embedded_at = meta
        if space is None:
            health = Health("empty", None, None)
        elif canary is None or self.encoder is None or self.encoder.space != _space_from(space):
            health = Health("unchecked", None, embedded_at)
        else:
            similarity = min(_cosine(a, b)
                             for a, b in zip(json.loads(canary), self._canary(pool), strict=True))
            state = ("ok" if similarity >= self.drift_warn
                     else "drifted" if similarity >= self.drift_stale else "stale")
            health = Health(state, similarity, embedded_at)
        return health

    def prepare_reembed(self, batch: int = 256) -> Reembedding:
        """Encode every stored text with the store's encoder. Touches no table.

        Works whether or not the encoder's space matches the stored one,
        so it is also how a store moves to a new model.
        """
        if self.encoder is None:
            raise RuntimeError(f"store {self.prefix!r} has no encoder")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError("batch must be a positive integer")
        generation = self._generation()
        ids: list[int] = []
        vectors: list[tuple[float, ...]] = []
        last_id: int | None = None
        while True:
            if last_id is None:
                rows = self.conn.execute(
                    f"SELECT id, text FROM {self._t['items']} ORDER BY id LIMIT ?",
                    (batch,)).fetchall()
            else:
                rows = self.conn.execute(
                    f"SELECT id, text FROM {self._t['items']} WHERE id > ?"
                    " ORDER BY id LIMIT ?", (last_id, batch)).fetchall()
            if not rows:
                break
            encoded = self.encoder.encode([text for _, text in rows])
            if len(encoded) != len(rows):
                raise ValueError(f"{len(encoded)} vectors for {len(rows)} items")
            ids.extend(item_id for item_id, _ in rows)
            vectors.extend(tuple(map(float, v)) for v in encoded)
            last_id = rows[-1][0]
        return Reembedding(generation, self.encoder.space, tuple(ids), tuple(vectors),
                           self._canary() if ids else ())

    def apply_reembed(self, plan: Reembedding) -> None:
        """Replace every vector with `plan`'s, in the caller's transaction."""
        self._require_transaction()
        if self._generation() != plan.generation:
            raise RuntimeError("the store changed after prepare_reembed; prepare again")
        if len(plan.rows) != len(plan.ids):
            raise ValueError(f"{len(plan.rows)} vectors for {len(plan.ids)} items")
        item_count = self.conn.execute(
            f"SELECT count(*) FROM {self._t['items']}").fetchone()[0]
        if len(plan.ids) != item_count:
            raise RuntimeError("the re-embedding plan does not cover all stored items")
        if plan.ids:
            if len(plan.canary) != len(CANARY_TEXTS):
                raise ValueError(
                    f"{len(plan.canary)} canary vectors for {len(CANARY_TEXTS)} texts")
            self._blobs_for(plan.space, plan.canary)
        blobs = self._blobs_for(plan.space, plan.rows)
        t = self._t
        self.conn.execute(f"DELETE FROM {t['vectors']}")
        self.conn.executemany(f"INSERT INTO {t['vectors']} (id, vec) VALUES (?, ?)",
                              zip(plan.ids, blobs))
        if plan.ids:
            self.conn.execute(
                f"UPDATE {t['meta']} SET space = ?, canary = ?, embedded_at = ?,"
                " generation = generation + 1",
                (_space_json(plan.space), json.dumps(plan.canary), time.time()))
            self._cache.clear()
        else:
            self._bump(reset_space=True)

    def _index(self, where) -> vindex.VectorIndex:
        # Inside a transaction the rows may yet roll back, so nothing built
        # there is kept.
        cacheable = where.cacheable and not self.conn.in_transaction
        key = (self._cache_namespace, where.sql, where.params)
        generation = self._generation()
        if cacheable:
            hit = self._cache.get(key, generation)
            if hit is not None:
                return hit
        t = self._t
        rows = self.conn.execute(
            f"SELECT v.id, v.vec FROM {t['vectors']} v JOIN {t['items']} i ON i.id = v.id"
            f" WHERE {where.sql} ORDER BY v.id", where.params).fetchall()
        idx = vindex.build([r[0] for r in rows], unpack([r[1] for r in rows]))
        if cacheable:
            self._cache.put(key, generation, idx)
        return idx

    def select_ids(self, filter: Filter | None) -> set[int]:
        """Ids of items matching `filter`."""
        self._check_space()
        where = compile_filter(filter, self.kinds)
        return {r[0] for r in self.conn.execute(
            f"SELECT i.id FROM {self._t['items']} i WHERE {where.sql}", where.params)}

    # -- hydration --------------------------------------------------------

    def fetch(self, ids: Sequence[int], fields: set[str]) -> dict[int, Record]:
        """Records for `ids` with the asked-for fields: text, metadata, vector."""
        unknown = set(fields) - {"text", "metadata", "vector"}
        if unknown:
            raise ValueError(f"unknown fetch fields {sorted(unknown)}")
        self._check_space()
        t = self._t
        names = [f.name for f in self.fields]
        selected = ["i.id"]
        if "text" in fields:
            selected.append("i.text")
        if "metadata" in fields:
            selected.append("i.extra")
            selected.extend(f'i."{n}"' for n in names)
        if "vector" in fields:
            selected.append("v.vec")
        join = (f" LEFT JOIN {t['vectors']} v ON v.id = i.id"
                if "vector" in fields else "")
        out: dict[int, Record] = {}
        ids = list(ids)
        for start in range(0, len(ids), _BATCH):
            chunk = ids[start:start + _BATCH]
            rows = self.conn.execute(
                f"SELECT {', '.join(selected)} FROM {t['items']} i{join}"
                f" WHERE i.id IN ({', '.join('?' * len(chunk))})", chunk).fetchall()
            for row in rows:
                item_id = row[0]
                at = 1
                text = row[at] if "text" in fields else None
                at += int("text" in fields)
                metadata = None
                if "metadata" in fields:
                    extra = row[at]
                    at += 1
                    values = row[at:at + len(names)]
                    at += len(names)
                    metadata = {n: (bool(v) if v is not None and self.kinds[n] == "bool"
                                    else v) for n, v in zip(names, values)}
                    metadata.update(json.loads(extra))
                vector = None
                if "vector" in fields and row[at] is not None:
                    vector = tuple(float(x) for x in unpack([row[at]])[0])
                out[item_id] = Record(item_id, text, metadata, vector)
        return out
