"""Fusing ranked lists from several sources into one."""

from __future__ import annotations

import math
import unittest

from semsift.fuse import Evidence, RankedList, Scored, blend, first, rrf


def ranked(source, *pairs):
    """A list whose raw value is the score, for tests that do not care."""
    return RankedList(source, tuple(Scored(i, s, s) for i, s in pairs))


class RankedListTests(unittest.TestCase):
    def test_a_list_out_of_order_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ranked("kw", (1, 0.2), (2, 0.9))

    def test_an_id_twice_in_one_list_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ranked("kw", (1, 0.9), (1, 0.5))

    def test_non_finite_scores_and_raw_values_are_refused(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(field="score", value=value):
                with self.assertRaises(ValueError):
                    RankedList("kw", (Scored(1, value, 1.0),))
            with self.subTest(field="raw", value=value):
                with self.assertRaises(ValueError):
                    RankedList("kw", (Scored(1, 1.0, value),))

    def test_ties_must_be_in_id_order(self) -> None:
        with self.assertRaises(ValueError):
            ranked("kw", (2, 1.0), (1, 1.0))

    def test_warnings_must_be_an_immutable_sequence_of_strings(self) -> None:
        for warnings in (None, ["drifted"], ("drifted", 3)):
            with self.subTest(warnings=warnings), self.assertRaises(TypeError):
                RankedList("kw", (), warnings)


class RrfTests(unittest.TestCase):
    def test_appearing_in_two_sources_beats_appearing_in_one(self) -> None:
        a = ranked("kw", (1, 2.0), (2, 1.0))
        b = ranked("vec", (1, 0.9), (3, 0.8))
        self.assertEqual(1, rrf([a, b]).items[0][0])

    def test_scores_follow_the_formula(self) -> None:
        a = ranked("kw", (1, 5.0), (2, 4.0))
        got = dict(rrf([a], k=60).items)
        self.assertAlmostEqual(1 / 61, got[1])
        self.assertAlmostEqual(1 / 62, got[2])

    def test_weights_scale_a_source(self) -> None:
        a = ranked("kw", (1, 1.0))
        b = ranked("vec", (2, 1.0))
        got = rrf([a, b], weights={"kw": 1.0, "vec": 0.25}).items
        self.assertEqual([1, 2], [i for i, _ in got])
        self.assertAlmostEqual(0.25 / 61, dict(got)[2])

    def test_a_zero_weight_drops_the_source(self) -> None:
        a = ranked("kw", (1, 1.0))
        b = ranked("vec", (2, 1.0))
        self.assertEqual([1], [i for i, _ in rrf([a, b], weights={"vec": 0.0}).items])

    def test_ties_break_by_id(self) -> None:
        a = ranked("kw", (9, 1.0))
        b = ranked("vec", (4, 1.0))
        self.assertEqual([4, 9], [i for i, _ in rrf([a, b]).items])

    def test_k_cannot_make_a_non_positive_denominator(self) -> None:
        with self.assertRaises(ValueError):
            rrf([ranked("kw", (1, 1.0))], k=-1)

    def test_weights_must_be_finite_and_non_negative(self) -> None:
        source = ranked("kw", (1, 1.0))
        for weight in (math.nan, math.inf, -math.inf, -1.0):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    rrf([source], weights={"kw": weight})


class BlendTests(unittest.TestCase):
    def test_min_max_puts_each_source_on_zero_to_one(self) -> None:
        a = ranked("kw", (1, 30.0), (2, 10.0))
        b = ranked("vec", (2, 0.9), (1, 0.1))
        got = dict(blend([a, b], normalise="min-max").items)
        self.assertAlmostEqual(1.0, got[1])     # 1.0 + 0.0
        self.assertAlmostEqual(1.0, got[2])     # 0.0 + 1.0

    def test_an_item_missing_from_a_source_scores_nothing_there(self) -> None:
        a = ranked("kw", (1, 30.0), (2, 10.0))
        b = ranked("vec", (3, 0.9))
        got = dict(blend([a, b], normalise="min-max").items)
        self.assertAlmostEqual(1.0, got[3])

    def test_weights_scale_normalised_scores(self) -> None:
        a = ranked("kw", (1, 30.0), (2, 10.0))
        b = ranked("vec", (2, 0.9), (1, 0.1))
        got = dict(blend([a, b], weights={"kw": 0.2, "vec": 0.8},
                         normalise="min-max").items)
        self.assertAlmostEqual(0.2, got[1])
        self.assertAlmostEqual(0.8, got[2])

    def test_max_divides_by_the_largest_magnitude(self) -> None:
        got = dict(blend([ranked("kw", (1, 4.0), (2, 1.0))], normalise="max").items)
        self.assertAlmostEqual(0.25, got[2])

    def test_sum_divides_by_the_total_magnitude(self) -> None:
        got = dict(blend([ranked("kw", (1, 3.0), (2, 1.0))], normalise="sum").items)
        self.assertAlmostEqual(0.75, got[1])

    def test_z_score_centres_on_the_mean(self) -> None:
        got = dict(blend([ranked("kw", (1, 3.0), (2, 2.0), (3, 1.0))],
                         normalise="z-score").items)
        self.assertAlmostEqual(0.0, got[2])
        self.assertAlmostEqual(-got[3], got[1])

    def test_equal_scores_do_not_divide_by_zero(self) -> None:
        expected = {"min-max": 1.0, "max": 0.0, "sum": 0.0, "z-score": 0.0}
        for how, score in expected.items():
            got = blend([ranked("kw", (1, 0.0), (2, 0.0))], normalise=how)
            self.assertEqual(((1, score), (2, score)), got.items, how)

    def test_an_unknown_normalisation_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            blend([ranked("kw", (1, 1.0))], normalise="bogus")

    def test_negative_scores_stay_finite_and_best_first(self) -> None:
        source = ranked("kw", (1, -1.0), (2, -3.0))
        for how in ("min-max", "max", "sum", "z-score"):
            with self.subTest(normalise=how):
                got = blend([source], normalise=how).items
                self.assertEqual([1, 2], [item_id for item_id, _ in got])
                self.assertTrue(all(math.isfinite(score) for _, score in got))

    def test_single_item_sources_stay_finite(self) -> None:
        for how in ("min-max", "max", "sum", "z-score"):
            with self.subTest(normalise=how):
                [(item_id, score)] = blend(
                    [ranked("kw", (1, -2.0))], normalise=how).items
                self.assertEqual(1, item_id)
                self.assertTrue(math.isfinite(score))

    def test_large_finite_scores_do_not_overflow_normalisation(self) -> None:
        source = ranked("kw", (1, 1e308), (2, 0.0), (3, -1e308))
        for how in ("min-max", "max", "sum", "z-score"):
            with self.subTest(normalise=how):
                got = blend([source], normalise=how).items
                self.assertEqual([1, 2, 3], [item_id for item_id, _ in got])
                self.assertTrue(all(math.isfinite(score) for _, score in got))

    def test_overflowing_weighted_result_is_refused(self) -> None:
        lists = [ranked("a", (1, 1.0)), ranked("b", (1, 1.0))]
        with self.assertRaises(ValueError):
            blend(lists, weights={"a": 1e308, "b": 1e308})

    def test_a_zero_weight_drops_the_source(self) -> None:
        a = ranked("kw", (1, 1.0))
        b = ranked("vec", (2, 1.0))
        got = blend([a, b], weights={"vec": 0.0})
        self.assertEqual([1], [item_id for item_id, _ in got.items])
        self.assertNotIn(2, got.evidence)

    def test_weights_must_be_finite_and_non_negative(self) -> None:
        source = ranked("kw", (1, 1.0))
        for weight in (math.nan, math.inf, -math.inf, -1.0):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    blend([source], weights={"kw": weight})


class FirstTests(unittest.TestCase):
    def test_takes_the_first_non_empty_source_in_order(self) -> None:
        empty = ranked("exact")
        a = ranked("kw", (5, 3.0), (6, 1.0))
        b = ranked("vec", (7, 0.9))
        self.assertEqual([5, 6], [i for i, _ in first([empty, a, b]).items])


class EvidenceTests(unittest.TestCase):
    def test_evidence_carries_source_rank_score_and_raw_value(self) -> None:
        a = RankedList("kw", (Scored(1, 15.6, -15.6),))
        b = RankedList("vec", (Scored(1, 0.46, 0.46),))
        for fused in (rrf([a, b]), blend([a, b]), first([a, b])):
            self.assertIn(Evidence("kw", 1, 15.6, -15.6), fused.evidence[1])
        self.assertEqual(
            (Evidence("kw", 1, 15.6, -15.6),
             Evidence("vec", 1, 0.46, 0.46)),
            rrf([a, b]).evidence[1],
        )

    def test_duplicate_source_names_are_refused(self) -> None:
        lists = [ranked("kw", (1, 2.0)), ranked("kw", (2, 1.0))]
        for fuse in (rrf, blend, first):
            with self.subTest(fuser=fuse.__name__):
                with self.assertRaises(ValueError):
                    fuse(lists)


if __name__ == "__main__":
    unittest.main()
