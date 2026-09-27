"""Packing vectors for storage and reading them back."""

from __future__ import annotations

import unittest

from semsift.store.codec import pack, unpack


class PackingTests(unittest.TestCase):
    def test_two_bytes_per_dimension(self) -> None:
        self.assertEqual(512, len(pack([0.1] * 256)))

    def test_packing_normalises(self) -> None:
        got = unpack([pack([3.0, 4.0])])[0]
        self.assertAlmostEqual(0.6, got[0], places=3)
        self.assertAlmostEqual(0.8, got[1], places=3)

    def test_a_zero_vector_does_not_divide_by_zero(self) -> None:
        self.assertEqual([0.0, 0.0], list(unpack([pack([0.0, 0.0])])[0]))

    def test_a_vector_must_have_a_dimension(self) -> None:
        with self.assertRaises(ValueError):
            pack([])

    def test_a_vector_must_be_one_dimensional(self) -> None:
        with self.assertRaises(ValueError):
            pack([[1.0, 0.0], [0.0, 1.0]])

    def test_non_finite_components_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            pack([float("nan"), 1.0])

    def test_half_precision_holds_the_range_we_use(self) -> None:
        import numpy as np

        v = list(np.linspace(-1.0, 1.0, 256))
        back = unpack([pack(v)])[0]
        norm = float(np.linalg.norm(v))
        self.assertTrue(
            np.allclose([x / norm for x in v], back, atol=1e-3))

class UnpackTests(unittest.TestCase):
    def test_rows_become_a_matrix(self) -> None:
        m = unpack([pack([3.0, 4.0]), pack([0.0, 2.0])])
        self.assertEqual((2, 2), m.shape)
        self.assertAlmostEqual(0.6, float(m[0][0]), places=3)

    def test_rows_of_mixed_width_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            unpack([pack([1.0, 0.0]), pack([1.0, 0.0, 0.0])])

    def test_a_row_that_is_not_whole_components_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            unpack([b"\x00\x00\x00"])

    def test_no_rows_is_an_empty_matrix(self) -> None:
        self.assertEqual(0, unpack([]).shape[0])

    def test_a_zero_width_row_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            unpack([b""])


if __name__ == "__main__":
    unittest.main()
