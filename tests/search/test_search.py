"""The composer: sources, base filter, fusion, hydration, rerankers."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest

from semsift.embed import VectorSpace
from semsift.fuse import Evidence, Fused, RankedList, Scored, rrf
from semsift.rerank import Candidate
from semsift.search import KeywordSource, Search, VectorSource
from semsift.store import Field, Item, Store
from semsift.store import filters as f


class TableEncoder:
    def __init__(self, table):
        self.table = table
        self.space = VectorSpace("table", "fake", "", 2, "", "")

    def encode(self, texts):
        return [self.table.get(t, [0.0, 1.0]) for t in texts]

    def encode_query(self, texts):
        return self.encode(texts)


class Recorder:
    """A source that returns fixed items and remembers the filter it got."""

    name = "mine"

    def __init__(self, ids, fail: Exception | None = None):
        self.ids, self.fail, self.seen, self.depths = ids, fail, [], []

    def search(self, query, k, filter):
        self.seen.append(filter)
        self.depths.append(k)
        if self.fail:
            raise self.fail
        return RankedList(self.name, tuple(Scored(i, 1.0 / (n + 1), 0.0)
                                           for n, i in enumerate(self.ids[:k])))


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        enc = TableEncoder({"query": [1.0, 0.0], "vector match": [1.0, 0.0],
                            "both match": [0.9, 0.1]})
        self.store = Store(self.conn, "s", [Field("account", "text"), Field("key", "text"),
                                            Field("span", "text")], encoder=enc)
        items = [Item(1, "keyword query words padded out with several more words", {"account": "work", "key": "doc1", "span": "1-3"}),
                 Item(2, "vector match", {"account": "work", "key": "doc2", "span": "4-6"}),
                 Item(3, "both match", {"account": "work", "key": "doc3", "span": "7-9"},
                      keyword_text="both match query"),
                 Item(4, "query elsewhere", {"account": "home", "key": "doc4", "span": "1-1"})]
        vectors = self.store.embed(items)
        self.conn.execute("BEGIN")
        self.store.upsert(items, vectors)
        self.conn.execute("COMMIT")

    def search(self, **kw) -> Search:
        kw.setdefault("sources", [VectorSource(self.store), KeywordSource(self.store)])
        return Search(self.store, **kw)


class CompositionTests(Fixture):
    def test_hits_come_from_either_source_and_agreement_ranks_first(self) -> None:
        res = self.search(citation=("key", "span")).run(
            "query", 3, filter=f.eq("account", "work"))
        ids = [h.id for h in res.hits]
        self.assertEqual(3, ids[0])
        self.assertIn(1, ids)
        self.assertIn(2, ids)
        top = res.hits[0]
        self.assertEqual("both match", top.text)
        self.assertEqual({"key": "doc3", "span": "7-9"}, top.citation)
        self.assertEqual({"vector", "keyword"}, {e.source for e in top.evidence})
        self.assertEqual((), res.warnings)

    def test_the_base_filter_reaches_every_source_and_cannot_be_widened(self) -> None:
        mine = Recorder([4, 1])
        s = self.search(sources=[VectorSource(self.store), KeywordSource(self.store), mine],
                        base_filter=f.eq("account", "work"))
        res = s.run("query", 10, filter=f.or_(f.eq("account", "home"), f.eq("account", "work")))
        self.assertNotIn(4, [h.id for h in res.hits])
        self.assertIsInstance(mine.seen[0], f.All)
        self.assertIn(f.eq("account", "work"), mine.seen[0].parts)

    def test_a_consumer_id_outside_the_filter_is_dropped_at_hydration(self) -> None:
        # The consumer source ignores the filter it was given; hydration
        # applies the combined filter again, so its hit cannot leak.
        s = self.search(sources=[Recorder([4])], base_filter=f.eq("account", "work"))
        self.assertEqual((), s.run("query", 5).hits)

    def test_scope_recheck_and_hydration_share_one_database_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/search.sqlite"
            conn = sqlite3.connect(path, isolation_level=None)
            conn.execute("PRAGMA journal_mode = WAL")
            store = Store(conn, "s", [Field("account", "text")])
            conn.execute("BEGIN")
            store.upsert([Item(1, "private", {"account": "work"})])
            conn.execute("COMMIT")
            other = sqlite3.connect(path, isolation_level=None)
            original = store.select_ids

            def move_after_scope_check(filter):
                allowed = original(filter)
                other.execute('UPDATE "s_items" SET account = ? WHERE id = 1',
                              ("home",))
                return allowed

            store.select_ids = move_after_scope_check
            try:
                result = Search(store, sources=[Recorder([1])],
                                base_filter=f.eq("account", "work")).run("q", 1)
                self.assertEqual("work", result.hits[0].metadata["account"])
            finally:
                other.close()
                conn.close()

    def test_explicit_zero_candidate_depth_is_honoured(self) -> None:
        source = Recorder([1])
        result = self.search(sources=[source], depth=0).run("query", 1)
        self.assertEqual([0], source.depths)
        self.assertEqual((), result.hits)

    def test_depth_and_requested_hit_count_are_non_negative_integers(self) -> None:
        for bad in (-1, True, 1.5):
            with self.subTest(depth=bad), self.assertRaises(ValueError):
                self.search(depth=bad)
            with self.subTest(k=bad), self.assertRaises(ValueError):
                self.search(sources=[Recorder([1])]).run("query", bad)

    def test_hydration_drops_are_not_reported_as_reranking(self) -> None:
        result = self.search(sources=[Recorder([4])],
                             base_filter=f.eq("account", "work")).run("query", 1)
        self.assertEqual(1, len(result.warnings))
        self.assertIn("outside the filter", result.warnings[0])


class FailureTests(Fixture):
    def test_a_failing_source_is_skipped_with_a_warning(self) -> None:
        s = self.search(sources=[Recorder([], fail=OSError("server down")),
                                 KeywordSource(self.store)])
        res = s.run("query", 5)
        self.assertTrue(res.hits)
        self.assertEqual(1, len(res.warnings))
        self.assertIn("server down", res.warnings[0])

    def test_when_every_source_fails_the_search_raises(self) -> None:
        s = self.search(sources=[Recorder([], fail=OSError("down"))])
        with self.assertRaises(OSError):
            s.run("query", 5)

    def test_a_bad_filter_raises_instead_of_warning(self) -> None:
        """Even when another source succeeds with nothing to hydrate."""
        sources = lambda: [KeywordSource(self.store), Recorder([])]  # noqa: E731
        with self.assertRaises((ValueError, TypeError)):
            self.search(sources=sources()).run("query", 5, filter=f.eq("nope", 1))
        with self.assertRaises((ValueError, TypeError)):
            self.search(sources=sources(), base_filter=f.eq("account", 3)).run("query", 5)

    def test_a_broken_id_subquery_raises_instead_of_warning(self) -> None:
        broken = f.IdSet("SELECT id FROM no_such_table")
        with self.assertRaises(sqlite3.Error):
            self.search(sources=[KeywordSource(self.store), Recorder([])]).run(
                "query", 5, filter=broken)

    def test_a_broken_id_subquery_raises_even_with_no_candidates(self) -> None:
        broken = f.IdSet("SELECT id FROM no_such_table")
        with self.assertRaises(sqlite3.Error):
            self.search(sources=[Recorder([])]).run("query", 5, filter=broken)


class RerankerContractTests(Fixture):
    def test_rerankers_get_the_fields_they_declare(self) -> None:
        seen = {}

        class Peek:
            needs = frozenset({"vector"})

            def rerank(self, query, candidates):
                seen["vector"] = all(c.vector is not None for c in candidates)
                return list(candidates)

        self.search(rerankers=[Peek()]).run("query", 3)
        self.assertTrue(seen["vector"])

    def test_a_reranker_that_adds_or_repeats_ids_is_refused(self) -> None:
        class Adds:
            needs = frozenset()

            def rerank(self, query, candidates):
                return list(candidates) + [Candidate(99, 1.0)]

        class Repeats:
            needs = frozenset()

            def rerank(self, query, candidates):
                return list(candidates) + list(candidates[:1])

        for bad in (Adds(), Repeats()):
            with self.assertRaises(ValueError):
                self.search(rerankers=[bad]).run("query", 3)

    def test_a_reranker_that_returns_a_non_finite_score_is_refused(self) -> None:
        class NonFinite:
            needs = frozenset()

            def rerank(self, query, candidates):
                return [Candidate(c.id, float("nan")) for c in candidates]

        with self.assertRaisesRegex(ValueError, "finite"):
            self.search(rerankers=[NonFinite()]).run("query", 3)

    def test_a_reranker_cannot_replace_hydrated_fields(self) -> None:
        class ReplacesText:
            needs = frozenset()

            def rerank(self, query, candidates):
                return [Candidate(c.id, c.score, "forged", c.metadata, c.vector)
                        for c in candidates]

        with self.assertRaisesRegex(ValueError, "fields"):
            self.search(rerankers=[ReplacesText()]).run("query", 3)

    def test_fewer_than_k_after_reranking_is_a_warning(self) -> None:
        class DropAll:
            needs = frozenset()

            def rerank(self, query, candidates):
                return list(candidates[:1])

        res = self.search(rerankers=[DropAll()]).run("query", 3)
        self.assertEqual(1, len(res.hits))
        self.assertEqual(1, len(res.warnings))

    def test_only_the_candidate_depth_is_reranked(self) -> None:
        counts = []

        class Count:
            needs = frozenset()

            def rerank(self, query, candidates):
                counts.append(len(candidates))
                return list(candidates)

        self.search(rerankers=[Count()], depth=2).run("query", 1)
        self.assertEqual([2], counts)

    def test_the_fuser_is_pluggable(self) -> None:
        calls = []

        def spy(lists):
            calls.append([rl.source for rl in lists])
            return rrf(lists)

        self.search(fuser=spy).run("query", 3)
        self.assertEqual([["vector", "keyword"]], calls)

    def test_a_fuser_cannot_expand_the_candidate_set(self) -> None:
        def expands(lists):
            return Fused(((4, 1.0),))

        with self.assertRaisesRegex(ValueError, "added"):
            self.search(sources=[Recorder([1])], fuser=expands).run("query", 1)

    def test_a_fuser_must_return_unique_finite_best_first_items(self) -> None:
        bad_items = (
            ((1, 1.0), (1, 0.5)),
            ((1, float("nan")),),
            ((1, 0.5), (2, 1.0)),
            ((2, 1.0), (1, 1.0)),
        )
        for items in bad_items:
            with self.subTest(items=items), self.assertRaises(ValueError):
                self.search(sources=[Recorder([1, 2])],
                            fuser=lambda lists, items=items: Fused(items)).run("query", 2)

    def test_a_fuser_must_preserve_source_evidence(self) -> None:
        def loses_evidence(lists):
            return Fused(((1, 1.0),))

        with self.assertRaisesRegex(ValueError, "evidence"):
            self.search(sources=[Recorder([1])], fuser=loses_evidence).run("query", 1)

        evidence = (Evidence("mine", 1, 1.0, 0.0),)
        result = self.search(
            sources=[Recorder([1])],
            fuser=lambda lists: Fused(((1, 1.0),), {1: evidence}),
        ).run("query", 1)
        self.assertEqual(evidence, result.hits[0].evidence)


class ContextPackingTests(unittest.TestCase):
    """What reaches the caller's context: one hit per document, within a
    character budget, chosen after reranking."""

    #: id -> (document, text). The source ranks them in id order.
    ROWS = {1: ("mail-a", "a" * 30), 2: ("mail-a", "b" * 30), 3: ("mail-b", "c" * 30),
            4: (None, "d" * 30), 5: (None, "e" * 30), 6: ("mail-c", "f" * 30)}

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.store = Store(self.conn, "p", [Field("doc", "text")])
        self.conn.execute("BEGIN")
        self.store.upsert([Item(i, text, {"doc": doc}) for i, (doc, text) in self.ROWS.items()])
        self.conn.execute("COMMIT")

    def run_search(self, k, **kw):
        search = Search(self.store, sources=[Recorder(sorted(self.ROWS))], depth=6, **kw)
        return search.run("q", k)

    def test_one_hit_per_document_keeps_the_best_ranked(self) -> None:
        res = self.run_search(10, distinct_by="doc")
        self.assertEqual([1, 3, 4, 5, 6], [h.id for h in res.hits])

    def test_hits_without_the_field_are_never_grouped(self) -> None:
        res = self.run_search(10, distinct_by="doc")
        self.assertEqual([4, 5], [h.id for h in res.hits if h.metadata["doc"] is None])

    def test_k_counts_documents_not_chunks(self) -> None:
        # Unpacked, the top two are both chunks of mail-a.
        self.assertEqual([1, 2], [h.id for h in self.run_search(2).hits])
        self.assertEqual([1, 3], [h.id for h in self.run_search(2, distinct_by="doc").hits])

    def test_the_budget_stops_before_the_hit_that_would_overflow(self) -> None:
        res = self.run_search(10, max_chars=95)
        self.assertEqual([1, 2, 3], [h.id for h in res.hits])
        self.assertEqual((), res.warnings)

    def test_the_first_hit_is_kept_over_budget_with_a_warning(self) -> None:
        res = self.run_search(10, max_chars=10)
        self.assertEqual([1], [h.id for h in res.hits])
        self.assertEqual(1, len(res.warnings))
        self.assertIn("max_chars", res.warnings[0])

    def test_packing_and_budget_combine(self) -> None:
        res = self.run_search(10, distinct_by="doc", max_chars=65)
        self.assertEqual([1, 3], [h.id for h in res.hits])

    def test_the_distinct_field_must_be_declared(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct_by"):
            Search(self.store, sources=[], distinct_by="missing")

    def test_the_budget_is_a_positive_integer(self) -> None:
        for bad in (0, -1, True, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "max_chars"):
                    Search(self.store, sources=[], max_chars=bad)


if __name__ == "__main__":
    unittest.main()
