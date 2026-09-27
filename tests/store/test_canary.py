"""Canary texts catch an encoder whose outputs changed under the same identity."""

from __future__ import annotations

from dataclasses import replace
import sqlite3
import threading
import unittest

from semsift.embed import FakeEncoder
from semsift.search import KeywordSource, Search, VectorSource
from semsift.store import CANARY_TEXTS, Field, Item, StaleVectors, Store, Vectors


class Shifted:
    """FakeEncoder's space and outputs, with every output rotated by `mix`.

    mix 0 reproduces the base; a small mix drifts; a large one is a
    different model hiding behind the same identity.
    """

    def __init__(self, mix: float, dims: int = 8) -> None:
        self.base = FakeEncoder(dims=dims)
        self.other = FakeEncoder(dims=dims, doc_prefix="other: ", query_prefix="other: ")
        self.mix = mix
        self.space = self.base.space

    def _blend(self, a, b):
        return [[(1 - self.mix) * x + self.mix * y for x, y in zip(u, v)] for u, v in zip(a, b)]

    def encode(self, texts):
        return self._blend(self.base.encode(texts), self.other.encode(texts))

    def encode_query(self, texts):
        return self._blend(self.base.encode_query(texts), self.other.encode_query(texts))


ITEMS = [Item(1, "alpha words"), Item(2, "beta words")]


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        first = Store(self.conn, "s", [Field("key", "text")], encoder=FakeEncoder(dims=8))
        vectors = first.embed(ITEMS)
        self.conn.execute("BEGIN")
        first.upsert(ITEMS, vectors)
        self.conn.execute("COMMIT")

    def reopen(self, encoder, **kw) -> Store:
        return Store(self.conn, "s", [Field("key", "text")], encoder=encoder, **kw)


class DriftTests(Fixture):
    def test_the_first_write_records_canaries_and_a_time(self) -> None:
        store = self.reopen(FakeEncoder(dims=8))
        self.assertEqual("ok", store.health().state)
        self.assertIsNotNone(store.health().embedded_at)

    def test_an_unchanged_encoder_is_ok_and_searches_quietly(self) -> None:
        store = self.reopen(Shifted(0.0))
        self.assertEqual("ok", store.health().state)
        self.assertEqual((), store.search_vector("alpha", 5).warnings)

    def test_a_small_drift_searches_with_a_warning(self) -> None:
        store = self.reopen(Shifted(0.02))
        health = store.health()
        self.assertEqual("drifted", health.state)
        self.assertTrue(0.99 <= health.similarity < 0.9999)
        ranked = store.search_vector("alpha", 5)
        self.assertEqual(2, len(ranked.items))
        self.assertIn("re-embed", ranked.warnings[0])

    def test_a_large_drift_refuses_vector_search(self) -> None:
        store = self.reopen(Shifted(0.6))
        self.assertEqual("stale", store.health().state)
        with self.assertRaises(StaleVectors):
            store.search_vector("alpha", 5)
        self.assertEqual([1], [s.id for s in store.search_keyword("alpha", 5).items])

    def test_thresholds_can_be_overridden(self) -> None:
        store = self.reopen(Shifted(0.02), drift_warn=0.9, drift_stale=0.5)
        self.assertEqual("ok", store.health().state)

    def test_thresholds_must_be_finite_ordered_probabilities(self) -> None:
        invalid = ((float("nan"), 0.5), (float("inf"), 0.5),
                   (0.5, 0.6), (1.1, 0.5), (0.5, -0.1))
        for warn, stale in invalid:
            with self.subTest(warn=warn, stale=stale), self.assertRaises(ValueError):
                self.reopen(Shifted(0.0), drift_warn=warn, drift_stale=stale)

    def test_vectors_written_without_canaries_are_not_checked(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        enc = FakeEncoder(dims=8)
        store = Store(conn, "p", [], encoder=enc)
        rows = tuple(tuple(v) for v in enc.encode(["x"]))
        conn.execute("BEGIN")
        store.upsert([Item(1, "x")], Vectors(enc.space, rows))
        conn.execute("COMMIT")
        self.assertEqual("unchecked", Store(conn, "p", [], encoder=Shifted(0.6)).health().state)

    def test_health_refuses_an_encoder_from_another_space(self) -> None:
        with self.assertRaises(ValueError):
            self.reopen(FakeEncoder(dims=16)).health()

    def test_each_canary_encode_must_return_exactly_one_vector(self) -> None:
        store = self.reopen(Shifted(0.0))
        encode = store.encoder.encode

        def duplicates_the_row(texts):
            rows = encode(texts)
            return rows + rows if len(texts) == 1 else rows

        store.encoder.encode = duplicates_the_row
        with self.assertRaisesRegex(ValueError, "canary text"):
            store.health()

    def test_precomputed_canaries_must_cover_every_fixed_text(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        enc = FakeEncoder(dims=8)
        store = Store(conn, "p", [], encoder=enc)
        rows = tuple(tuple(v) for v in enc.encode(["x"]))
        one_canary = (tuple(enc.encode([CANARY_TEXTS[0]])[0]),)
        conn.execute("BEGIN")
        with self.assertRaisesRegex(ValueError, "canary vectors"):
            store.upsert([Item(1, "x")], Vectors(enc.space, rows, one_canary))
        self.assertEqual({}, store.fetch([1], {"text"}))
        conn.execute("ROLLBACK")


class PerSearchTests(Fixture):
    def test_every_search_checks_canaries_through_encode_only(self) -> None:
        store = self.reopen(Shifted(0.0))
        calls = []
        enc = store.encoder
        doc, query = enc.encode, enc.encode_query
        enc.encode = lambda texts: calls.append(("doc", tuple(texts))) or doc(texts)
        enc.encode_query = lambda texts: calls.append(("query", tuple(texts))) or query(texts)
        store.search_vector("alpha", 5)
        store.search_vector("alpha", 5)
        canary_calls = [c for c in calls if c[1][0] in CANARY_TEXTS]
        # The canary encodes run in parallel, so their order is not fixed.
        self.assertEqual(sorted([("doc", (t,)) for t in CANARY_TEXTS] * 2), sorted(canary_calls))

    def test_a_model_swapped_after_a_search_is_caught_without_a_write(self) -> None:
        enc = Shifted(0.0)
        store = self.reopen(enc)
        store.search_vector("alpha", 5)
        enc.mix = 0.6
        with self.assertRaises(StaleVectors):
            store.search_vector("alpha", 5)

    def test_a_non_positive_search_still_checks_canaries(self) -> None:
        store = self.reopen(Shifted(0.6))
        with self.assertRaises(StaleVectors):
            store.search_vector("alpha", 0)

    def test_the_check_runs_alongside_the_query_encode(self) -> None:
        started = threading.Barrier(1 + len(CANARY_TEXTS), timeout=2)

        class Synchronized(Shifted):
            def encode(self, texts):
                started.wait()
                return super().encode(texts)

            def encode_query(self, texts):
                started.wait()
                return super().encode_query(texts)

        store = self.reopen(Synchronized(0.0))
        store.search_vector("alpha", 5)

    def test_query_encode_must_return_exactly_one_vector(self) -> None:
        store = self.reopen(Shifted(0.0))
        encode_query = store.encoder.encode_query

        def duplicates_the_query(texts):
            rows = encode_query(texts)
            return rows + rows if texts == ["alpha"] else rows

        store.encoder.encode_query = duplicates_the_query
        with self.assertRaisesRegex(ValueError, "query text"):
            store.search_vector("alpha", 5)


class SearchWarningTests(Fixture):
    def test_the_composer_passes_a_drift_warning_through(self) -> None:
        store = self.reopen(Shifted(0.02))
        res = Search(store, sources=[VectorSource(store), KeywordSource(store)]).run("alpha", 2)
        self.assertTrue(any("re-embed" in w for w in res.warnings))

    def test_stale_vectors_leave_keyword_results_and_a_warning(self) -> None:
        store = self.reopen(Shifted(0.6))
        res = Search(store, sources=[VectorSource(store), KeywordSource(store)]).run("alpha", 2)
        self.assertEqual([1], [h.id for h in res.hits])
        self.assertTrue(any("re-embed" in w for w in res.warnings))


class ReembedTests(Fixture):
    def reembed(self, store: Store) -> None:
        plan = store.prepare_reembed()
        self.conn.execute("BEGIN")
        store.apply_reembed(plan)
        self.conn.execute("COMMIT")

    def test_reembedding_restores_health_and_rewrites_every_vector(self) -> None:
        enc = Shifted(0.6)
        store = self.reopen(enc)
        before = store.health().embedded_at
        self.reembed(store)
        self.assertEqual("ok", store.health().state)
        self.assertGreaterEqual(store.health().embedded_at, before)
        got = store.fetch([1], {"vector"})[1].vector
        want = enc.encode(["alpha words"])[0]
        norm = sum(x * x for x in want) ** 0.5
        self.assertGreater(sum(a * b / norm for a, b in zip(got, want)), 0.999)
        self.assertEqual(2, len(store.search_vector("alpha", 5).items))

    def test_reembedding_moves_a_store_to_a_new_vector_space(self) -> None:
        store = self.reopen(FakeEncoder(dims=16))
        with self.assertRaises(ValueError):
            store.search_vector("alpha", 5)
        self.reembed(store)
        self.assertEqual(16, store.space.dims)
        self.assertEqual(2, len(store.search_vector("alpha", 5).items))

    def test_a_write_between_prepare_and_apply_is_refused(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        self.conn.execute("BEGIN")
        store.remove([2])
        with self.assertRaises(RuntimeError):
            store.apply_reembed(plan)
        self.conn.execute("ROLLBACK")

    def test_an_added_item_between_prepare_and_apply_is_refused(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        item = Item(3, "gamma words")
        vectors = store.embed([item])
        self.conn.execute("BEGIN")
        store.upsert([item], vectors)
        with self.assertRaises(RuntimeError):
            store.apply_reembed(plan)
        self.conn.execute("ROLLBACK")

    def test_prepare_touches_no_table_and_apply_needs_a_transaction(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        self.assertFalse(self.conn.in_transaction)
        with self.assertRaises(RuntimeError):
            store.apply_reembed(plan)

    def test_reembedding_an_empty_store_does_not_adopt_a_space(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        store = Store(conn, "empty", [], encoder=FakeEncoder(dims=8))

        def unexpected_encode(texts):
            raise AssertionError("an empty store has nothing to encode")

        store.encoder.encode = unexpected_encode
        plan = store.prepare_reembed()
        conn.execute("BEGIN")
        store.apply_reembed(plan)
        conn.execute("COMMIT")
        self.assertIsNone(store.space)
        self.assertEqual("empty", store.health().state)

    def test_reembedding_adds_canaries_to_an_unchecked_store(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        enc = FakeEncoder(dims=8)
        store = Store(conn, "plain", [])
        rows = tuple(tuple(v) for v in enc.encode(["alpha"]))
        conn.execute("BEGIN")
        store.upsert([Item(1, "alpha")], Vectors(enc.space, rows))
        conn.execute("COMMIT")
        checked = Store(conn, "plain", [], encoder=enc)
        self.assertEqual("unchecked", checked.health().state)
        plan = checked.prepare_reembed()
        conn.execute("BEGIN")
        checked.apply_reembed(plan)
        conn.execute("COMMIT")
        self.assertEqual("ok", checked.health().state)

    def test_prepare_rejects_a_non_positive_or_non_integer_batch(self) -> None:
        store = self.reopen(Shifted(0.6))
        for batch in (-1, 0, True, 1.5):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                store.prepare_reembed(batch=batch)

    def test_prepare_rejects_an_encoder_that_drops_rows(self) -> None:
        store = self.reopen(Shifted(0.6))
        encode = store.encoder.encode

        def drops_the_last_row(texts):
            rows = encode(texts)
            return rows[:-1] if len(texts) > 1 else rows

        store.encoder.encode = drops_the_last_row
        with self.assertRaisesRegex(ValueError, "vectors for"):
            store.prepare_reembed()

    def test_apply_rejects_a_plan_with_missing_rows_before_writing(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        broken = replace(plan, rows=plan.rows[:-1])
        before = store.fetch([1, 2], {"vector"})
        self.conn.execute("BEGIN")
        with self.assertRaisesRegex(ValueError, "vectors for"):
            store.apply_reembed(broken)
        self.assertEqual(before, store.fetch([1, 2], {"vector"}))
        self.conn.execute("ROLLBACK")

    def test_apply_rejects_a_plan_for_only_some_stored_items(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        broken = replace(plan, ids=plan.ids[:-1], rows=plan.rows[:-1])
        before = store.fetch([1, 2], {"vector"})
        self.conn.execute("BEGIN")
        with self.assertRaisesRegex(RuntimeError, "stored items"):
            store.apply_reembed(broken)
        self.assertEqual(before, store.fetch([1, 2], {"vector"}))
        self.conn.execute("ROLLBACK")

    def test_apply_rejects_an_incomplete_canary_set_before_writing(self) -> None:
        store = self.reopen(Shifted(0.6))
        plan = store.prepare_reembed()
        broken = replace(plan, canary=plan.canary[:1])
        before = store.fetch([1, 2], {"vector"})
        self.conn.execute("BEGIN")
        with self.assertRaisesRegex(ValueError, "canary vectors"):
            store.apply_reembed(broken)
        self.assertEqual(before, store.fetch([1, 2], {"vector"}))
        self.conn.execute("ROLLBACK")


if __name__ == "__main__":
    unittest.main()
