"""Generic rerankers: recency, metadata rules, MMR."""

from __future__ import annotations

import math
import sqlite3
import unittest

from semsift.rerank import MMR, Candidate, Recency, Rule, Rules
from semsift.store import filters as f

DAY = 86400.0


def cands(*pairs, **columns):
    """Candidates from (id, score) pairs; keyword lists give metadata per id."""
    out = []
    for n, (i, s) in enumerate(pairs):
        meta = {k: v[n] for k, v in columns.items()}
        out.append(Candidate(i, s, metadata=meta))
    return out


class RecencyTests(unittest.TestCase):
    def rerank(self, cs, **kw):
        r = Recency("at", half_life=10 * DAY, clock=lambda: 100 * DAY, **kw)
        return r.rerank("q", cs)

    def test_score_halves_every_half_life(self) -> None:
        got = {c.id: c.score for c in self.rerank(
            cands((1, 1.0), (2, 1.0), at=[100 * DAY, 90 * DAY]))}
        self.assertAlmostEqual(1.0, got[1])
        self.assertAlmostEqual(0.5, got[2])

    def test_newer_can_overtake_older(self) -> None:
        got = self.rerank(cands((1, 1.0), (2, 0.8), at=[70 * DAY, 100 * DAY]))
        self.assertEqual([2, 1], [c.id for c in got])

    def test_a_missing_timestamp_takes_the_configured_factor(self) -> None:
        got = self.rerank(cands((1, 1.0), at=[None]), missing=0.25)
        self.assertAlmostEqual(0.25, got[0].score)

    def test_a_future_timestamp_counts_as_now(self) -> None:
        got = self.rerank(cands((1, 1.0), at=[200 * DAY]))
        self.assertAlmostEqual(1.0, got[0].score)

    def test_needs_metadata(self) -> None:
        self.assertEqual({"metadata"}, Recency("at", half_life=1.0).needs)

    def test_a_half_life_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            Recency("at", half_life=0)


class RulesTests(unittest.TestCase):
    def test_a_matching_rule_multiplies_and_matches_compound(self) -> None:
        rules = Rules([Rule(f.glob("path", "tests/*"), 0.5),
                       Rule(f.eq("status", "superseded"), 0.2)])
        got = {c.id: c.score for c in rules.rerank("q", cands(
            (1, 1.0), (2, 1.0), (3, 1.0),
            path=["src/a", "tests/b", "tests/c"],
            status=["live", "live", "superseded"]))}
        self.assertEqual({1: 1.0, 2: 0.5}, {k: got[k] for k in (1, 2)})
        self.assertAlmostEqual(0.1, got[3])

    def test_results_are_resorted_by_score(self) -> None:
        rules = Rules([Rule(f.eq("kind", "test"), 0.1)])
        got = rules.rerank("q", cands((1, 1.0), (2, 0.5), kind=["test", "code"]))
        self.assertEqual([2, 1], [c.id for c in got])

    def test_negation_matches_missing_values_as_the_store_does(self) -> None:
        rules = Rules([Rule(f.ne("status", "live"), 0.5)])
        got = {c.id: c.score for c in rules.rerank("q", cands(
            (1, 1.0), (2, 1.0), status=["live", None]))}
        self.assertEqual({1: 1.0, 2: 0.5}, got)

    def test_glob_matches_sqlite_instead_of_python_fnmatch(self) -> None:
        conn = sqlite3.connect(":memory:")
        cases = (("a", "[!a]"), ("b", "[!a]"),
                 ("a", "[^a]"), ("b", "[^a]"),
                 ("[", "["), ("]", "[]]"),
                 ("a\0b", "a?b"), ("a", "a\0b"))
        for value, pattern in cases:
            with self.subTest(value=value, pattern=pattern):
                matches = bool(conn.execute(
                    "SELECT ? GLOB ?", (value, pattern)).fetchone()[0])
                got = Rules([Rule(f.glob("path", pattern), 2.0)]).rerank(
                    "q", cands((1, 1.0), path=[value]))[0]
                self.assertEqual(matches, got.score == 2.0)

    def test_empty_compounds_are_rejected_as_store_filters_are(self) -> None:
        for empty in (f.and_(), f.or_()):
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                Rules([Rule(empty, 2.0)]).rerank("q", cands((1, 1.0)))

    def test_factors_must_be_finite_and_positive(self) -> None:
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError, msg=bad):
                Rule(f.eq("a", "b"), bad)

    def test_id_sets_cannot_be_rules(self) -> None:
        with self.assertRaises(ValueError):
            Rules([Rule(f.IdSet("SELECT 1"), 0.5)]).rerank("q", cands((1, 1.0), a=[1]))


class MMRTests(unittest.TestCase):
    def test_a_near_duplicate_gives_way_to_something_different(self) -> None:
        cs = [Candidate(1, 1.0, vector=(1.0, 0.0)),
              Candidate(2, 0.95, vector=(1.0, 0.01)),
              Candidate(3, 0.9, vector=(0.0, 1.0))]
        got = MMR(balance=0.5).rerank("q", cs)
        self.assertEqual([1, 3, 2], [c.id for c in got])

    def test_balance_one_keeps_relevance_order(self) -> None:
        cs = [Candidate(1, 1.0, vector=(1.0, 0.0)),
              Candidate(2, 0.95, vector=(1.0, 0.01)),
              Candidate(3, 0.9, vector=(0.0, 1.0))]
        self.assertEqual([1, 2, 3], [c.id for c in MMR(balance=1.0).rerank("q", cs)])

    def test_relevance_is_normalised_so_its_scale_does_not_matter(self) -> None:
        def ids(scale):
            cs = [Candidate(1, 1.0 * scale, vector=(1.0, 0.0)),
                  Candidate(2, 0.95 * scale, vector=(1.0, 0.01)),
                  Candidate(3, 0.9 * scale, vector=(0.0, 1.0))]
            return [c.id for c in MMR(balance=0.5).rerank("q", cs)]
        self.assertEqual(ids(1.0), ids(1000.0))

    def test_needs_vectors_and_keeps_every_candidate(self) -> None:
        self.assertEqual({"vector"}, MMR().needs)
        cs = [Candidate(i, 1.0 / i, vector=(1.0, float(i))) for i in range(1, 6)]
        self.assertEqual({1, 2, 3, 4, 5}, {c.id for c in MMR().rerank("q", cs)})

    def test_a_candidate_without_a_vector_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MMR().rerank("q", [Candidate(1, 1.0)])


if __name__ == "__main__":
    unittest.main()
