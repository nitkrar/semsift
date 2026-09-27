"""Items written first, vectors attached later."""

from __future__ import annotations

import sqlite3
import unittest

from semsift.embed import FakeEncoder
from semsift.store import Item, Store


class LateVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.plain = Store(self.conn, "s", [])
        self.enc = FakeEncoder(dims=8)
        self.embedding = Store(self.conn, "s", [], encoder=self.enc)
        self.conn.execute("BEGIN")
        self.plain.upsert([Item(1, "alpha words"), Item(2, "beta words")])
        self.conn.execute("COMMIT")

    def test_items_without_vectors_are_listed_and_keyword_searchable(self) -> None:
        self.assertEqual([1, 2], self.embedding.missing_vectors())
        self.assertEqual([1], [s.id for s in self.plain.search_keyword("alpha", 5).items])
        self.assertEqual((), self.embedding.search_vector("alpha", 5).items)

    def test_added_vectors_make_items_vector_searchable(self) -> None:
        ids = self.embedding.missing_vectors()
        records = self.embedding.fetch(ids, {"text"})
        vectors = self.embedding.embed([Item(i, records[i].text) for i in ids])
        self.conn.execute("BEGIN")
        self.embedding.add_vectors(ids, vectors)
        self.conn.execute("COMMIT")
        self.assertEqual([], self.embedding.missing_vectors())
        self.assertEqual({1, 2}, {s.id for s in self.embedding.search_vector("alpha", 5).items})
        self.assertEqual(self.enc.space, self.embedding.space)
        self.assertEqual("ok", self.embedding.health().state)

    def test_vectors_for_unknown_ids_or_outside_a_transaction_are_refused(self) -> None:
        vectors = self.embedding.embed([Item(9, "gamma")])
        with self.assertRaises(RuntimeError):
            self.embedding.add_vectors([9], vectors)
        self.conn.execute("BEGIN")
        with self.assertRaises(ValueError):
            self.embedding.add_vectors([9], vectors)
        self.conn.execute("ROLLBACK")


if __name__ == "__main__":
    unittest.main()
