"""Vector backends share one contract; approximate ones keep high recall."""

from __future__ import annotations

import importlib.util
import unittest

from semsift.store import index as vindex


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def backends():
    out = [("exhaustive", vindex.Exhaustive)]
    if available("hnswlib"):
        out.append(("hnsw", vindex.HNSW))
    if available("usearch"):
        out.append(("usearch", vindex.USearch))
    return out


class ContractTests(unittest.TestCase):
    def test_nearest_first_with_cosine_scores(self) -> None:
        ids = [10, 11, 12]
        rows = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        for name, backend in backends():
            with self.subTest(backend=name):
                got = backend(ids, rows).query([1.0, 0.0], 3)
                self.assertEqual([11, 12, 10], [i for i, _ in got])
                self.assertAlmostEqual(1.0, got[0][1], places=3)
                self.assertAlmostEqual(0.0, got[2][1], places=3)

    def test_equal_scores_resolve_by_id(self) -> None:
        ids = [9, 4, 7, 2, 5]
        rows = [[1.0, 0.0]] * 5
        for name, backend in backends():
            with self.subTest(backend=name):
                got = backend(ids, rows).query([1.0, 0.0], 5)
                self.assertEqual([2, 4, 5, 7, 9], [i for i, _ in got])

    def test_k_beyond_the_rows_and_empty_input(self) -> None:
        for name, backend in backends():
            with self.subTest(backend=name):
                self.assertEqual(2, len(backend([1, 2], [[1.0, 0.0], [0.0, 1.0]]).query([1.0, 0.0], 10)))
                self.assertEqual([], backend([], []).query([1.0, 0.0], 3))
                self.assertEqual([], backend([1], [[1.0, 0.0]]).query([1.0, 0.0], 0))

    def test_a_query_of_another_width_is_refused(self) -> None:
        for name, backend in backends():
            with self.subTest(backend=name):
                with self.assertRaises(ValueError):
                    backend([1], [[1.0, 0.0]]).query([1.0, 0.0, 0.0], 1)


class RecallTests(unittest.TestCase):
    def test_approximate_backends_find_most_true_neighbours(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        rows = rng.normal(size=(2000, 32)).astype("float32")
        queries = rng.normal(size=(50, 32)).astype("float32")
        ids = list(range(2000))
        exact = vindex.Exhaustive(ids, rows)
        truth = [{i for i, _ in exact.query(q, 10)} for q in queries]
        for name, backend in backends()[1:]:
            with self.subTest(backend=name):
                index = backend(ids, rows)
                found = sum(len(t & {i for i, _ in index.query(q, 10)})
                            for q, t in zip(queries, truth))
                self.assertGreaterEqual(found / (10 * len(queries)), 0.95)


if __name__ == "__main__":
    unittest.main()
