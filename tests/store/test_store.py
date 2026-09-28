"""The SQLite store: writes, both searches, filters, vector space, cache."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from semsift.embed import FakeEncoder, VectorSpace
from semsift.store import Field, Item, StaleKeywords, Store, Vectors
from semsift.store import filters as f
from semsift.store import index as vindex


class TableEncoder:
    """Vectors chosen per text, so rankings in tests are exact.

    Texts outside the table (the store's canary texts) get the first
    axis, so they encode but never matter to a ranking.
    """

    def __init__(self, table: dict[str, list[float]], model: str = "table") -> None:
        self.table = table
        dims = len(next(iter(table.values())))
        self.space = VectorSpace(model=model, backend="fake", variant="",
                                 dims=dims, doc_prefix="", pooling="")

    def encode(self, texts):
        dims = self.space.dims
        return [self.table.get(t, [1.0] + [0.0] * (dims - 1)) for t in texts]

    def encode_query(self, texts):
        return self.encode(texts)


FIELDS = [Field("account", "text", indexed=True), Field("at", "int"),
          Field("done", "bool"), Field("path", "text")]


def memory() -> sqlite3.Connection:
    return sqlite3.connect(":memory:", isolation_level=None)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = memory()
        self.enc = FakeEncoder(dims=8)
        self.store = Store(self.conn, "docs", FIELDS, encoder=self.enc)

    def put(self, *items: Item, store: Store | None = None) -> None:
        store = store or self.store
        vectors = store.embed(items)
        self.conn.execute("BEGIN")
        store.upsert(items, vectors)
        self.conn.execute("COMMIT")

    def kw(self, query: str, k: int = 10, filter=None, store: Store | None = None) -> list[int]:
        store = store or self.store
        return [s.id for s in store.search_keyword(query, k, filter).items]


class ConstructionTests(unittest.TestCase):
    def test_a_prefix_that_is_not_an_identifier_is_refused(self) -> None:
        for bad in ("Docs", "1docs", "docs; drop table x", "", "a-b"):
            with self.assertRaises(ValueError, msg=bad):
                Store(memory(), bad, [])

    def test_field_names_types_and_tokenizers_are_checked(self) -> None:
        for fields in ([Field("Bad", "text")], [Field("text", "text")],
                       [Field("a", "blob")], [Field("a", "int"), Field("a", "int")]):
            with self.assertRaises(ValueError, msg=repr(fields)):
                Store(memory(), "s", fields)
        with self.assertRaises(ValueError):
            Store(memory(), "s", [], tokenizer="porter; drop")

    def test_reopening_with_other_fields_is_refused(self) -> None:
        conn = memory()
        Store(conn, "s", [Field("a", "int")])
        with self.assertRaises(ValueError):
            Store(conn, "s", [Field("a", "text")])
        Store(conn, "s", [Field("a", "int")])

    def test_creating_a_store_does_not_commit_the_callers_transaction(self) -> None:
        conn = memory()
        conn.execute("CREATE TABLE mine (x)")
        conn.execute("BEGIN")
        conn.execute("INSERT INTO mine VALUES (1)")
        Store(conn, "s", [Field("a", "int")])
        conn.execute("ROLLBACK")
        self.assertEqual(0, conn.execute("SELECT count(*) FROM mine").fetchone()[0])

    def test_creating_a_store_does_not_leave_an_implicit_transaction(self) -> None:
        conn = sqlite3.connect(":memory:")
        store = Store(conn, "s", [])
        self.assertFalse(conn.in_transaction)
        with self.assertRaises(RuntimeError):
            store.upsert([Item(1, "alpha")])

    def test_failed_schema_creation_leaves_no_partial_store(self) -> None:
        conn = memory()

        def deny_fts(action, _arg1, _arg2, _database, _source):
            if action == sqlite3.SQLITE_CREATE_VTABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_fts)
        with self.assertRaises(sqlite3.DatabaseError):
            Store(conn, "s", [])
        conn.set_authorizer(None)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 's_%'").fetchall()
        self.assertEqual([], tables)

    def test_two_prefixes_in_one_database_are_independent(self) -> None:
        conn = memory()
        a = Store(conn, "a", [], encoder=FakeEncoder(dims=4))
        b = Store(conn, "b", [], encoder=FakeEncoder(dims=4))
        item = Item(1, "shared words here")
        conn.execute("BEGIN")
        a.upsert([item], a.embed([item]))
        conn.execute("COMMIT")
        self.assertEqual([1], [s.id for s in a.search_keyword("shared", 5).items])
        self.assertEqual((), b.search_keyword("shared", 5).items)


class WriteTests(Fixture):
    def test_writes_outside_a_transaction_are_refused(self) -> None:
        item = Item(1, "alpha")
        with self.assertRaises(RuntimeError):
            self.store.upsert([item], self.store.embed([item]))
        with self.assertRaises(RuntimeError):
            self.store.remove([1])

    def test_a_rollback_undoes_the_item_its_vector_and_its_keyword_row(self) -> None:
        item = Item(1, "alpha beta")
        vectors = self.store.embed([item])
        self.conn.execute("BEGIN")
        self.store.upsert([item], vectors)
        self.conn.execute("ROLLBACK")
        self.assertEqual([], self.kw("alpha"))
        self.assertEqual((), self.store.search_vector("alpha beta", 5).items)
        self.assertEqual({}, self.store.fetch([1], {"text"}))

    def test_a_rollback_restores_replaced_item_vector_and_keyword_row(self) -> None:
        self.put(Item(1, "old words"))
        old_vector = self.store.fetch([1], {"vector"})[1].vector
        replacement = Item(1, "new words")
        self.conn.execute("BEGIN")
        self.store.upsert([replacement], self.store.embed([replacement]))
        self.conn.execute("ROLLBACK")
        self.assertEqual([1], self.kw("old"))
        self.assertEqual([], self.kw("new"))
        restored = self.store.fetch([1], {"text", "vector"})[1]
        self.assertEqual("old words", restored.text)
        self.assertEqual(old_vector, restored.vector)

    def test_replacing_an_item_replaces_its_keyword_terms(self) -> None:
        self.put(Item(1, "old wording"))
        self.put(Item(1, "new wording"))
        self.assertEqual([], self.kw("old"))
        self.assertEqual([1], self.kw("new"))
        self.assertEqual(1, len(self.store.search_vector("x", 10).items))

    def test_remove_drops_the_item_from_both_searches(self) -> None:
        self.put(Item(1, "alpha"), Item(2, "alpha"))
        self.conn.execute("BEGIN")
        self.store.remove([1])
        self.conn.execute("COMMIT")
        self.assertEqual([2], self.kw("alpha"))
        self.assertEqual([2], [s.id for s in self.store.search_vector("alpha", 5).items])

    def test_keyword_text_is_searched_and_text_is_embedded(self) -> None:
        self.put(Item(1, "body only", keyword_text="body only pathword"))
        self.assertEqual([1], self.kw("pathword"))
        stored = self.store.fetch([1], {"vector"})[1].vector
        expected = self.enc.encode(["body only"])[0]
        self.assertGreater(sum(a * b for a, b in zip(stored, expected))
                           / (sum(a * a for a in expected) ** 0.5), 0.999)

    def test_keywords_are_searched_alongside_text_but_not_embedded(self) -> None:
        self.put(Item(1, "body only", keywords="pathword"))
        self.assertEqual([1], self.kw("pathword"))
        self.assertEqual([1], self.kw("body"))
        stored = self.store.fetch([1], {"vector"})[1].vector
        expected = self.enc.encode(["body only"])[0]
        self.assertGreater(sum(a * b for a, b in zip(stored, expected))
                           / (sum(a * a for a in expected) ** 0.5), 0.999)

    def test_text_is_stored_once(self) -> None:
        import random

        rng = random.Random(0)
        vocabulary = [f"w{i}" for i in range(200)]
        text = " ".join(rng.choice(vocabulary) for _ in range(60_000))
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(Path(tmp) / "s.db", isolation_level=None)
            store = Store(conn, "s", [])
            conn.execute("BEGIN")
            store.upsert([Item(1, text, keywords="some path words")])
            conn.execute("COMMIT")
            size = (conn.execute("PRAGMA page_count").fetchone()[0]
                    * conn.execute("PRAGMA page_size").fetchone()[0])
            conn.close()
        # One copy plus the inverted index; a second copy would pass 2x.
        self.assertLess(size, 2 * len(text))

    def test_the_keyword_index_matches_the_items_after_every_write(self) -> None:
        def check() -> None:
            self.conn.execute(
                "INSERT INTO docs_fts(docs_fts, rank) VALUES ('integrity-check', 1)")

        self.put(Item(1, "alpha", keywords="one"), Item(2, "beta"),
                 Item(3, "gamma", keyword_text="override"))
        check()
        self.put(Item(1, "alpha two", keywords="uno"), Item(2, "beta", keywords="new"),
                 Item(3, "gamma"))
        check()
        self.conn.execute("BEGIN")
        self.store.upsert([Item(2, "rolled")], self.store.embed([Item(2, "rolled")]))
        self.conn.execute("ROLLBACK")
        check()
        self.conn.execute("BEGIN")
        self.store.remove([1])
        self.conn.execute("COMMIT")
        check()
        self.assertEqual([], self.kw("uno"))
        self.assertEqual([2], self.kw("new"))
        self.assertEqual([3], self.kw("gamma"))
        self.assertEqual([], self.kw("override"))
        self.conn.execute("BEGIN")
        self.store.clear()
        self.conn.execute("COMMIT")
        check()

    def test_deferred_keywords_are_searchable_after_a_sync(self) -> None:
        self.put(Item(1, "alpha"), Item(2, "beta"))
        self.conn.execute("BEGIN")
        self.store.defer_keywords()
        self.store.upsert([Item(1, "gamma")], self.store.embed([Item(1, "gamma")]))
        self.store.remove([2])
        self.store.upsert([Item(3, "delta")], self.store.embed([Item(3, "delta")]))
        self.conn.execute("COMMIT")
        with self.assertRaises(StaleKeywords):
            self.kw("gamma")
        self.conn.execute("BEGIN")
        self.store.sync_keywords()
        self.conn.execute("COMMIT")
        self.assertEqual([1], self.kw("gamma"))
        self.assertEqual([], self.kw("alpha beta"))
        self.assertEqual([3], self.kw("delta"))
        self.conn.execute(
            "INSERT INTO docs_fts(docs_fts, rank) VALUES ('integrity-check', 1)")

    def test_a_committed_deferral_outlives_the_store_object(self) -> None:
        self.put(Item(1, "alpha"))
        self.conn.execute("BEGIN")
        self.store.defer_keywords()
        self.store.upsert([Item(2, "beta")], self.store.embed([Item(2, "beta")]))
        self.conn.execute("COMMIT")
        reopened = Store(self.conn, "docs", FIELDS, encoder=self.enc)
        self.assertTrue(reopened.keywords_stale)
        with self.assertRaises(StaleKeywords):
            self.kw("beta", store=reopened)

    def test_a_rolled_back_deferral_leaves_the_index_in_step(self) -> None:
        self.put(Item(1, "alpha"))
        self.conn.execute("BEGIN")
        self.store.defer_keywords()
        self.store.upsert([Item(2, "beta")], self.store.embed([Item(2, "beta")]))
        self.conn.execute("ROLLBACK")
        self.assertFalse(self.store.keywords_stale)
        self.put(Item(3, "gamma"))
        self.assertEqual([3], self.kw("gamma"))
        self.conn.execute(
            "INSERT INTO docs_fts(docs_fts, rank) VALUES ('integrity-check', 1)")

    def test_declared_metadata_is_typed_and_extra_is_kept(self) -> None:
        self.put(Item(1, "alpha", {"account": "work", "at": 5, "done": True,
                                   "colour": "red"}))
        got = self.store.fetch([1], {"text", "metadata"})[1]
        self.assertEqual("alpha", got.text)
        self.assertEqual({"account": "work", "at": 5, "done": True,
                          "path": None, "colour": "red"}, got.metadata)
        self.conn.execute("BEGIN")
        for bad in ({"at": "5"}, {"done": 1}, {"account": 3}, {"at": True}):
            item = Item(2, "beta", bad)
            with self.assertRaises(TypeError, msg=repr(bad)):
                self.store.upsert([item], self.store.embed([item]))
        self.conn.execute("ROLLBACK")

    def test_unrepresentable_numeric_values_are_rejected(self) -> None:
        store = Store(self.conn, "numbers", [
            Field("weight", "float"), Field("count", "int")])
        huge = 10 ** 10_000
        self.conn.execute("BEGIN")
        with self.assertRaises(TypeError):
            store.upsert([Item(1, "alpha", {"weight": huge})])
        with self.assertRaises(ValueError):
            store.upsert([Item(1, "alpha", {"count": huge})])
        self.conn.execute("ROLLBACK")
        with self.assertRaises(f.FilterError):
            store.search_keyword("alpha", 1, f.eq("weight", huge))
        with self.assertRaises(f.FilterError):
            store.search_keyword("alpha", 1, f.eq("count", huge))

    def test_embed_touches_no_table(self) -> None:
        vectors = self.store.embed([Item(1, "alpha")])
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.enc.space, vectors.space)

    def test_vectors_must_match_the_items(self) -> None:
        items = [Item(1, "a"), Item(2, "b")]
        one = self.store.embed(items[:1])
        self.conn.execute("BEGIN")
        with self.assertRaises(ValueError):
            self.store.upsert(items, one)
        wrong_width = Vectors(self.enc.space, ((1.0, 0.0),))
        with self.assertRaises(ValueError):
            self.store.upsert(items[:1], wrong_width)
        self.conn.execute("ROLLBACK")

    def test_invalid_extra_json_does_not_partially_write_a_batch(self) -> None:
        store = Store(self.conn, "plain", [])
        self.conn.execute("BEGIN")
        with self.assertRaises(ValueError):
            store.upsert([
                Item(1, "valid"),
                Item(2, "invalid", {"weight": float("nan")}),
            ])
        self.conn.execute("COMMIT")
        self.assertEqual({}, store.fetch([1, 2], {"text"}))

    def test_item_identity_and_text_are_checked_before_writing(self) -> None:
        store = Store(self.conn, "plain", [])
        bad = [
            Item(True, "alpha"),
            Item(1 << 63, "alpha"),
            Item(1, 42),
            Item(1, "alpha", keyword_text=42),
        ]
        self.conn.execute("BEGIN")
        for item in bad:
            with self.assertRaises((TypeError, ValueError), msg=repr(item)):
                store.upsert([item])
        for item_id in (True, 1 << 63):
            with self.assertRaises((TypeError, ValueError), msg=repr(item_id)):
                store.remove([item_id])
        self.conn.execute("ROLLBACK")


class VectorSpaceTests(Fixture):
    def test_an_empty_store_adopts_the_encoders_space(self) -> None:
        self.assertIsNone(self.store.space)
        self.put(Item(1, "alpha"))
        self.assertEqual(self.enc.space, self.store.space)

    def test_vectors_from_another_space_are_refused(self) -> None:
        self.put(Item(1, "alpha"))
        other = FakeEncoder(dims=8, doc_prefix="p: ")
        vectors = Vectors(other.space, tuple(tuple(v) for v in other.encode(["b"])))
        self.conn.execute("BEGIN")
        with self.assertRaises(ValueError):
            self.store.upsert([Item(2, "b")], vectors)
        self.conn.execute("ROLLBACK")

    def test_a_store_opened_with_another_encoder_refuses_reads_and_writes(self) -> None:
        self.put(Item(1, "alpha"))
        other = Store(self.conn, "docs", FIELDS, encoder=FakeEncoder(dims=16))
        with self.assertRaises(ValueError):
            other.search_vector("alpha", 5)
        with self.assertRaises(ValueError):
            other.search_keyword("alpha", 5)
        with self.assertRaises(ValueError):
            other.embed([Item(2, "b")])

        self.conn.execute("BEGIN")
        with self.assertRaises(ValueError):
            other.remove([1])
        with self.assertRaises(ValueError):
            other.clear()
        self.conn.execute("ROLLBACK")

    def test_clear_empties_the_store_and_forgets_its_space(self) -> None:
        self.put(Item(1, "alpha"))
        self.conn.execute("BEGIN")
        self.store.clear()
        self.conn.execute("COMMIT")
        self.assertIsNone(self.store.space)
        self.assertEqual([], self.kw("alpha"))

    def test_removing_the_last_vector_forgets_its_space(self) -> None:
        self.put(Item(1, "alpha"))
        self.conn.execute("BEGIN")
        self.store.remove([1])
        self.conn.execute("COMMIT")
        self.assertIsNone(self.store.space)

    def test_an_empty_upsert_does_not_adopt_a_space(self) -> None:
        vectors = self.store.embed([])
        self.conn.execute("BEGIN")
        self.store.upsert([], vectors)
        self.conn.execute("COMMIT")
        self.assertIsNone(self.store.space)

    def test_keyword_only_replacement_drops_the_stale_vector(self) -> None:
        self.put(Item(1, "old text"))
        keyword_only = Store(self.conn, "docs", FIELDS)
        self.conn.execute("BEGIN")
        keyword_only.upsert([Item(1, "new text")])
        self.conn.execute("COMMIT")
        self.assertIsNone(keyword_only.fetch([1], {"vector"})[1].vector)
        self.assertIsNone(keyword_only.space)
        self.assertEqual((), self.store.search_vector("old text", 5).items)

    def test_a_keyword_only_store_has_no_vector_search(self) -> None:
        store = Store(self.conn, "plain", [])
        self.conn.execute("BEGIN")
        store.upsert([Item(1, "alpha")])
        self.conn.execute("COMMIT")
        self.assertEqual([1], self.kw("alpha", store=store))
        with self.assertRaises(RuntimeError):
            store.search_vector("alpha", 5)


class VectorSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = memory()
        table = {"east": [1.0, 0.0], "north": [0.0, 1.0], "north-east": [1.0, 1.0],
                 "west": [-1.0, 0.0], "same": [0.5, 0.5]}
        self.store = Store(self.conn, "v", FIELDS, encoder=TableEncoder(table))

    def put(self, pairs, **meta) -> None:
        items = [Item(i, t, dict(meta)) for i, t in pairs]
        vectors = self.store.embed(items)
        self.conn.execute("BEGIN")
        self.store.upsert(items, vectors)
        self.conn.execute("COMMIT")

    def ids(self, query, k=10, filter=None):
        return [s.id for s in self.store.search_vector(query, k, filter).items]

    def test_ranks_by_cosine_best_first(self) -> None:
        self.put([(1, "north"), (2, "east"), (3, "north-east"), (4, "west")])
        got = self.store.search_vector("east", 4).items
        self.assertEqual([2, 3, 1, 4], [s.id for s in got])
        self.assertAlmostEqual(1.0, got[0].score, places=3)
        self.assertAlmostEqual(-1.0, got[3].score, places=3)
        self.assertEqual(got[0].score, got[0].raw)

    def test_ties_resolve_by_id_even_beyond_any_fixed_margin(self) -> None:
        pairs = [(i, "same") for i in (97, 3, 55, 12, 80, 41, 7, 66, 29, 18) * 1]
        pairs += [(100 + i, "same") for i in range(60)]
        self.put(pairs)
        self.assertEqual([3, 7, 12, 18, 29], self.ids("north-east", 5))

    def test_k_larger_than_the_store_returns_everything(self) -> None:
        self.put([(1, "north"), (2, "east")])
        self.assertEqual(2, len(self.ids("east", 50)))

    def test_an_empty_store_and_an_empty_filter_return_nothing(self) -> None:
        self.assertEqual([], self.ids("east"))
        self.put([(1, "east")], account="work")
        self.assertEqual([], self.ids("east", filter=f.eq("account", "home")))

    def test_filters_apply_before_ranking(self) -> None:
        self.put([(1, "east")], account="home")
        self.put([(2, "north")], account="work")
        self.assertEqual([2], self.ids("east", 1, f.eq("account", "work")))

    def test_non_positive_k_returns_no_hits(self) -> None:
        self.put([(1, "east")])
        for k in (0, -1):
            self.assertEqual([], self.ids("east", k))
            self.assertEqual((), self.store.search_keyword("east", k).items)

    def test_query_width_must_match_the_stored_space(self) -> None:
        self.put([(1, "east")])
        self.store.encoder.table["too-wide"] = [1.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            self.store.search_vector("too-wide", 1)


class KeywordSearchTests(Fixture):
    def test_best_first_with_raw_bm25_kept(self) -> None:
        self.put(Item(1, "ledger ledger ledger settle"), Item(2, "a ledger mention"))
        got = self.store.search_keyword("ledger", 5).items
        self.assertEqual([1, 2], [s.id for s in got])
        self.assertLess(got[0].raw, 0)
        self.assertEqual(-got[0].raw, got[0].score)

    def test_queries_with_no_words_return_nothing(self) -> None:
        self.put(Item(1, "alpha"))
        for q in ("", "   ", "!!! ???"):
            self.assertEqual([], self.kw(q), q)

    def test_fts_syntax_in_a_query_is_treated_as_words(self) -> None:
        self.put(Item(1, "alpha and beta"))
        for q in ('alpha AND', '"alpha', 'alpha*', 'NEAR(alpha', 'alpha OR -beta'):
            self.assertEqual([1], self.kw(q), q)

    def test_any_query_word_matches(self) -> None:
        self.put(Item(1, "alpha"), Item(2, "beta"))
        self.assertEqual({1, 2}, set(self.kw("alpha beta")))

    def test_ties_resolve_by_id(self) -> None:
        self.put(Item(9, "same words"), Item(4, "same words"))
        self.assertEqual([4, 9], self.kw("same"))

    def test_trigram_search_preserves_punctuation_in_the_substring(self) -> None:
        store = Store(self.conn, "grams", [], tokenizer="trigram")
        self.conn.execute("BEGIN")
        store.upsert([Item(1, "foo-bar")])
        self.conn.execute("COMMIT")
        self.assertEqual([1], self.kw("oo-b", store=store))

    def test_trigram_search_matches_any_whitespace_separated_term(self) -> None:
        store = Store(self.conn, "grams", [], tokenizer="trigram")
        self.conn.execute("BEGIN")
        store.upsert([Item(1, "foo-bar")])
        self.conn.execute("COMMIT")
        self.assertEqual([1], self.kw("foo nonsense", store=store))


class FilterTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        self.put(Item(1, "word", {"account": "work", "at": 10, "done": True, "path": "src/a.py"}),
                 Item(2, "word", {"account": "home", "at": 20, "done": False, "path": "tests/b.py"}),
                 Item(3, "word", {"account": None, "at": 30, "path": "docs/c.md"}))

    def got(self, flt) -> set[int]:
        return set(self.kw("word", filter=flt))

    def test_comparisons(self) -> None:
        self.assertEqual({1}, self.got(f.eq("account", "work")))
        self.assertEqual({1, 2}, self.got(f.in_("account", ["work", "home"])))
        self.assertEqual({2, 3}, self.got(f.gt("at", 10)))
        self.assertEqual({1, 2}, self.got(f.lte("at", 20)))
        self.assertEqual({2}, self.got(f.between("at", 15, 25)))
        self.assertEqual({2}, self.got(f.glob("path", "tests/*")))
        self.assertEqual({3}, self.got(f.is_null("account")))
        self.assertEqual({1}, self.got(f.eq("done", True)))

    def test_combinators(self) -> None:
        self.assertEqual({1}, self.got(f.and_(f.gt("at", 5), f.eq("account", "work"))))
        self.assertEqual({1, 3}, self.got(f.or_(f.eq("account", "work"), f.gte("at", 30))))

    def test_negation_includes_missing_values(self) -> None:
        self.assertEqual({2, 3}, self.got(f.ne("account", "work")))
        self.assertEqual({2, 3}, self.got(f.not_(f.eq("account", "work"))))

    def test_bad_filters_raise_rather_than_widen(self) -> None:
        for flt in (f.eq("nope", 1), f.eq("at", "10"), f.glob("at", "1*"),
                    f.in_("account", []), f.eq("done", 1)):
            with self.assertRaises((ValueError, TypeError), msg=repr(flt)):
                self.kw("word", filter=flt)

    def test_an_id_set_selects_through_the_consumers_tables(self) -> None:
        self.conn.execute("CREATE TABLE pinned (item INTEGER, owner TEXT)")
        self.conn.execute("INSERT INTO pinned VALUES (2, 'me'), (3, 'you')")
        pinned = f.IdSet("SELECT item FROM pinned WHERE owner = ?", ("me",))
        self.assertEqual({2}, self.got(pinned))
        self.assertEqual({2}, set(s.id for s in
                                  self.store.search_vector("word", 5, pinned).items))

    def test_an_id_set_binds_its_parameters(self) -> None:
        owner = "me' OR 1=1 --"
        self.conn.execute("CREATE TABLE pinned (item INTEGER, owner TEXT)")
        self.conn.executemany("INSERT INTO pinned VALUES (?, ?)",
                              [(1, "me"), (2, owner)])
        pinned = f.IdSet("SELECT item FROM pinned WHERE owner = ?", (owner,))
        self.assertEqual({2}, self.got(pinned))


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "s.db"
        self.enc = FakeEncoder(dims=8)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def open(self) -> tuple[sqlite3.Connection, Store]:
        conn = sqlite3.connect(self.path, isolation_level=None)
        return conn, Store(conn, "s", [], encoder=self.enc)

    @staticmethod
    def write(conn, store, *items) -> None:
        vectors = store.embed(items)
        conn.execute("BEGIN")
        store.upsert(items, vectors)
        conn.execute("COMMIT")

    def test_a_committed_write_is_seen_by_the_next_search(self) -> None:
        conn, store = self.open()
        self.write(conn, store, Item(1, "alpha"))
        self.assertEqual(1, len(store.search_vector("alpha", 5).items))
        self.write(conn, store, Item(2, "beta"))
        self.assertEqual(2, len(store.search_vector("alpha", 5).items))

    def test_a_write_through_another_connection_is_seen(self) -> None:
        conn_a, a = self.open()
        conn_b, b = self.open()
        self.write(conn_a, a, Item(1, "alpha"))
        self.assertEqual(1, len(a.search_vector("alpha", 5).items))
        self.write(conn_b, b, Item(2, "beta"))
        self.assertEqual(2, len(a.search_vector("alpha", 5).items))

    def test_a_rolled_back_write_is_not_served_from_cache(self) -> None:
        """The rollback returns the generation to its old value, and a later
        commit elsewhere reaches the value the rolled-back write had: an
        index built inside the transaction would then look current."""
        conn, store = self.open()
        self.write(conn, store, Item(1, "alpha"))
        item = Item(2, "beta")
        vectors = store.embed([item])
        conn.execute("BEGIN")
        store.upsert([item], vectors)
        self.assertEqual({1, 2}, {s.id for s in store.search_vector("alpha", 5).items})
        conn.execute("ROLLBACK")
        conn_b, b = self.open()
        self.write(conn_b, b, Item(3, "gamma"))
        self.assertEqual({1, 3}, {s.id for s in store.search_vector("alpha", 5).items})

    def test_a_shared_cache_does_not_mix_stores(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        cache = vindex.Cache()
        a = Store(conn, "a", [], encoder=self.enc, cache=cache)
        b = Store(conn, "b", [], encoder=self.enc, cache=cache)
        self.write(conn, a, Item(1, "alpha"))
        self.write(conn, b, Item(2, "beta"))
        self.assertEqual([1], [s.id for s in a.search_vector("alpha", 5).items])
        self.assertEqual([2], [s.id for s in b.search_vector("alpha", 5).items])

    def test_cache_evicts_least_recently_used_entries_to_both_limits(self) -> None:
        def index(nbytes):
            from types import SimpleNamespace

            return SimpleNamespace(nbytes=nbytes)

        by_entries = vindex.Cache(max_entries=1, max_bytes=100)
        by_entries.put("a", 1, index(4))
        by_entries.put("b", 1, index(4))
        self.assertIsNone(by_entries.get("a", 1))
        self.assertIsNotNone(by_entries.get("b", 1))

        by_bytes = vindex.Cache(max_entries=3, max_bytes=8)
        by_bytes.put("a", 1, index(4))
        by_bytes.put("b", 1, index(4))
        self.assertIsNotNone(by_bytes.get("a", 1))
        by_bytes.put("c", 1, index(4))
        self.assertIsNone(by_bytes.get("b", 1))
        self.assertIsNotNone(by_bytes.get("a", 1))
        self.assertIsNotNone(by_bytes.get("c", 1))


class FetchTests(Fixture):
    def test_fetch_returns_only_what_was_asked_for_and_skips_unknown_ids(self) -> None:
        self.put(Item(1, "alpha", {"account": "work"}))
        got = self.store.fetch([1, 99], {"text"})
        self.assertEqual([1], list(got))
        self.assertEqual("alpha", got[1].text)
        self.assertIsNone(got[1].metadata)
        self.assertIsNone(got[1].vector)
        self.assertEqual(8, len(self.store.fetch([1], {"vector"})[1].vector))

    def test_fetch_does_not_read_unrequested_vector_blobs(self) -> None:
        self.put(Item(1, "alpha", {"account": "work"}))

        def deny_vectors(action, table, column, _database, _source):
            if (action == sqlite3.SQLITE_READ
                    and table == "docs_vectors" and column == "vec"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(deny_vectors)
        try:
            self.assertEqual("alpha", self.store.fetch([1], {"text"})[1].text)
        finally:
            self.conn.set_authorizer(None)

    def test_fetch_batches_more_ids_than_an_old_sqlite_parameter_limit(self) -> None:
        store = Store(self.conn, "plain", [])
        items = [Item(i, f"item {i}") for i in range(1, 1002)]
        self.conn.execute("BEGIN")
        store.upsert(items)
        self.conn.execute("COMMIT")
        got = store.fetch([item.id for item in items], {"text"})
        self.assertEqual(len(items), len(got))
        self.assertEqual("item 1001", got[1001].text)



class BackendTests(unittest.TestCase):
    def test_a_store_searches_through_the_backend_it_is_given(self) -> None:
        from semsift.store.index import Exhaustive

        built = []

        def recording(ids, matrix):
            built.append(list(ids))
            return Exhaustive(ids, matrix)

        conn = memory()
        store = Store(conn, "b", [], encoder=FakeEncoder(dims=8), backend=recording)
        items = [Item(1, "alpha"), Item(2, "beta")]
        vectors = store.embed(items)
        conn.execute("BEGIN")
        store.upsert(items, vectors)
        conn.execute("COMMIT")
        self.assertEqual({1, 2}, {s.id for s in store.search_vector("alpha", 5).items})
        self.assertEqual([[1, 2]], built)

if __name__ == "__main__":
    unittest.main()
