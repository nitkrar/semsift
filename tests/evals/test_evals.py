"""The evaluation harness: metrics, run files, splits, tuning, baselines."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from semsift.evals import (RunSet, evaluate, hit_at, mrr, ndcg_at, read_runs,
                          source_run, split, tune, write_runs)
from semsift.fuse import blend, rrf


class MetricTests(unittest.TestCase):
    def test_hit_at_k(self) -> None:
        self.assertEqual(1.0, hit_at(["a", "b", "c"], {"c": 1}, 3))
        self.assertEqual(0.0, hit_at(["a", "b", "c"], {"c": 1}, 2))

    def test_mrr_is_the_reciprocal_rank_of_the_first_relevant(self) -> None:
        self.assertAlmostEqual(1 / 3, mrr(["a", "b", "c"], {"c": 1, "d": 1}))
        self.assertEqual(0.0, mrr(["a"], {"z": 1}))

    def test_zero_grade_is_not_relevant(self) -> None:
        labels = {"zero": 0, "yes": 2}
        self.assertEqual(0.0, hit_at(["zero", "yes"], labels, 1))
        self.assertEqual(0.5, mrr(["zero", "yes"], labels))

    def test_duplicate_ranked_keys_are_refused(self) -> None:
        for metric in (lambda: hit_at(["a", "a"], {"a": 1}, 2),
                       lambda: mrr(["a", "a"], {"a": 1}),
                       lambda: ndcg_at(["a", "a"], {"a": 1}, 2)):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "ranked keys must be unique"):
                    metric()

    def test_negative_or_boolean_cutoffs_are_refused(self) -> None:
        for k in (-1, True):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    hit_at(["a"], {"a": 1}, k)
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    ndcg_at(["a"], {"a": 1}, k)

    def test_ndcg_by_hand(self) -> None:
        # DCG = 2/log2(2) + 0 + 1/log2(4) = 2.5; ideal = 2 + 1/log2(3).
        got = ndcg_at(["x", "y", "z"], {"x": 2, "z": 1}, 3)
        self.assertAlmostEqual(2.5 / (2 + 1 / math.log2(3)), got)
        self.assertEqual(1.0, ndcg_at(["x", "z"], {"x": 2, "z": 1}, 3))

    def test_evaluate_averages_and_reports_queries_with_no_answer_apart(self) -> None:
        run = {"q1": ["a"], "q2": ["b"], "none": ["c", "d"]}
        qrels = {"q1": {"a": 1}, "q2": {"z": 1}, "none": {}}
        got = evaluate(run, qrels, k=10)
        self.assertAlmostEqual(0.5, got["hit@10"])
        self.assertAlmostEqual(0.5, got["mrr"])
        self.assertEqual(2, got["queries"])
        self.assertEqual(1, got["no_answer_queries"])
        self.assertAlmostEqual(2.0, got["no_answer_returned"])

    def test_a_query_missing_from_the_run_scores_zero(self) -> None:
        self.assertEqual(0.0, evaluate({}, {"q": {"a": 1}}, k=5)["hit@5"])

    def test_a_query_with_only_zero_grades_is_a_no_answer_query(self) -> None:
        got = evaluate({"q": ["not-relevant"]}, {"q": {"not-relevant": 0}}, k=5)
        self.assertEqual(0, got["queries"])
        self.assertEqual(1, got["no_answer_queries"])
        self.assertEqual(1.0, got["no_answer_returned"])


def runset() -> RunSet:
    """Keyword finds the right answer for q1 and q2, vector for q3 and q4."""
    lists = {
        "q1": {"keyword": [("a", 3.0, -3.0), ("b", 1.0, -1.0)], "vector": [("b", 0.9, 0.9), ("a", 0.1, 0.1)]},
        "q2": {"keyword": [("c", 3.0, -3.0), ("d", 1.0, -1.0)], "vector": [("d", 0.9, 0.9), ("c", 0.1, 0.1)]},
        "q3": {"keyword": [("f", 3.0, -3.0), ("e", 1.0, -1.0)], "vector": [("e", 0.9, 0.9), ("f", 0.1, 0.1)]},
        "q4": {"keyword": [("h", 3.0, -3.0), ("g", 1.0, -1.0)], "vector": [("g", 0.9, 0.9), ("h", 0.1, 0.1)]},
    }
    return RunSet(lists, {"space": "test", "generation": 7})


QRELS = {"q1": {"a": 1}, "q2": {"c": 1}, "q3": {"e": 1}, "q4": {"g": 1}}


class RunFileTests(unittest.TestCase):
    def test_a_run_set_survives_a_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            write_runs(path, runset())
            back = read_runs(path)
        self.assertEqual(runset().lists, back.lists)
        self.assertEqual(runset().meta, back.meta)

    def test_one_sources_run_is_a_baseline(self) -> None:
        self.assertEqual({"q1": ["a", "b"], "q2": ["c", "d"], "q3": ["f", "e"],
                          "q4": ["h", "g"]}, source_run(runset(), "keyword"))

    def test_fusing_a_run_set_uses_the_given_fuser(self) -> None:
        fused = runset().fuse(lambda lists: rrf(lists, weights={"vector": 0.0}))
        self.assertEqual(source_run(runset(), "keyword"), fused)

    def test_fusing_tied_scores_breaks_the_tie_by_durable_key(self) -> None:
        runs = RunSet({"q": {"source": [("z", 1.0, 1.0), ("a", 1.0, 1.0)]}},
                      {"space": "test", "generation": 1})
        self.assertEqual(["a", "z"], runs.fuse(rrf)["q"])

    def test_a_run_file_requires_corpus_and_space_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            for meta in ({}, {"generation": 1}, {"space": "test"}):
                with self.subTest(meta=meta):
                    with self.assertRaisesRegex(ValueError, "generation.*space"):
                        write_runs(path, RunSet({}, meta))

    def test_a_run_file_refuses_non_standard_json_numbers(self) -> None:
        runs = RunSet({"q": {"source": [("a", math.nan, 0.0)]}},
                      {"space": "test", "generation": 1})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "JSON"):
                write_runs(Path(tmp) / "runs.jsonl", runs)

    def test_a_run_file_refuses_duplicate_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            path.write_text(
                '{"meta":{"space":"test","generation":1}}\n'
                '{"query":"q","lists":{}}\n'
                '{"query":"q","lists":{}}\n')
            with self.assertRaisesRegex(ValueError, "query.*appears twice"):
                read_runs(path)


class SplitTests(unittest.TestCase):
    def test_a_split_is_disjoint_complete_and_repeatable(self) -> None:
        ids = [f"q{i}" for i in range(40)]
        train, test = split(ids, test_share=0.25, seed="s")
        self.assertEqual(set(ids), set(train) | set(test))
        self.assertFalse(set(train) & set(test))
        self.assertEqual((train, test), split(list(reversed(ids)), test_share=0.25, seed="s"))
        self.assertTrue(5 <= len(test) <= 15)

    def test_a_share_outside_zero_to_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            split(["a"], test_share=1.0)

    def test_duplicate_query_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "query ids must be unique"):
            split(["a", "a"])


class TuneTests(unittest.TestCase):
    def test_picks_on_train_and_reports_on_test(self) -> None:
        options = {"keyword-only": lambda ls: rrf(ls, weights={"vector": 0.0}),
                   "vector-only": lambda ls: rrf(ls, weights={"keyword": 0.0}),
                   "blend": lambda ls: blend(ls)}
        got = tune(runset(), QRELS, options, train=["q1", "q2"], test=["q3", "q4"],
                   metric="mrr", k=10)
        self.assertEqual("keyword-only", got.best)
        self.assertEqual(1.0, got.train[got.best])
        self.assertEqual(0.5, got.test[got.best])
        self.assertEqual(set(options), set(got.train))

    def test_train_and_test_must_not_overlap(self) -> None:
        with self.assertRaises(ValueError):
            tune(runset(), QRELS, {"x": rrf}, train=["q1"], test=["q1"], metric="mrr")

    def test_train_and_test_must_both_contain_queries(self) -> None:
        for train, test in (([], ["q1"]), (["q1"], [])):
            with self.subTest(train=train, test=test):
                with self.assertRaisesRegex(ValueError, "train and test must be non-empty"):
                    tune(runset(), QRELS, {"x": rrf}, train=train, test=test)

    def test_only_higher_is_better_quality_metrics_can_be_tuned(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric"):
            tune(runset(), QRELS, {"x": rrf}, train=["q1"], test=["q2"],
                 metric="no_answer_returned")


class CollectTests(unittest.TestCase):
    def test_collect_records_each_source_by_durable_key(self) -> None:
        import sqlite3

        from semsift.embed import FakeEncoder
        from semsift.evals import collect
        from semsift.search import KeywordSource, VectorSource
        from semsift.store import Field, Item, Store

        conn = sqlite3.connect(":memory:", isolation_level=None)
        store = Store(conn, "s", [Field("key", "text")], encoder=FakeEncoder(dims=8))
        items = [Item(10, "alpha words", {"key": "doc-a"}), Item(20, "beta words", {"key": "doc-b"})]
        vectors = store.embed(items)
        conn.execute("BEGIN")
        store.upsert(items, vectors)
        conn.execute("COMMIT")
        keys = {10: "doc-a", 20: "doc-b"}

        runs = collect([KeywordSource(store), VectorSource(store)], {"q": "alpha"},
                       key_of=keys.__getitem__, k=5, meta={"space": "fake"})
        self.assertEqual(["doc-a"], [key for key, _, _ in runs.lists["q"]["keyword"]])
        self.assertEqual({"doc-a", "doc-b"}, {key for key, _, _ in runs.lists["q"]["vector"]})
        self.assertEqual({"space": "fake"}, runs.meta)

    def test_collect_refuses_duplicate_returned_source_names(self) -> None:
        from semsift.evals import collect
        from semsift.fuse import RankedList

        source = SimpleNamespace(search=lambda *args: RankedList("same", ()))
        with self.assertRaisesRegex(ValueError, "source names must be unique"):
            collect([source, source], {"q": "text"}, key_of=str)

    def test_collect_refuses_ids_that_map_to_the_same_durable_key(self) -> None:
        from semsift.evals import collect
        from semsift.fuse import RankedList, Scored

        source = SimpleNamespace(search=lambda *args: RankedList(
            "source", (Scored(1, 2.0, 2.0), Scored(2, 1.0, 1.0))))
        with self.assertRaisesRegex(ValueError, "durable key appears twice"):
            collect([source], {"q": "text"}, key_of=lambda _: "same")


if __name__ == "__main__":
    unittest.main()
