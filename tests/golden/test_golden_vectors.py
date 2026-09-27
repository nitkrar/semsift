"""Encoders reproduce recorded vectors, so a prefix or pooling change shows.

Skips a model that is not cached locally, or whose cached snapshot is not
the one recorded: vectors from another snapshot are expected to differ.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.golden.models import revision

GOLDEN = json.loads(Path(__file__).with_name("vectors.json").read_text())
#: Cosine similarity each vector must keep with its recorded value.
TOLERANCE = 0.9999


def cosine(a, b) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector widths differ: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


class GoldenVectorTests(unittest.TestCase):
    def test_cosine_refuses_different_vector_widths(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector widths differ"):
            cosine([1.0, 0.0], [1.0, 0.0, 100.0])

    def test_revision_is_the_snapshot_selected_by_main(self) -> None:
        repo = SimpleNamespace(
            repo_id="example/model",
            refs={"main": SimpleNamespace(commit_hash="active")},
            revisions=(
                SimpleNamespace(commit_hash="active", last_modified=1),
                SimpleNamespace(commit_hash="orphan", last_modified=2),
            ),
        )
        with patch("huggingface_hub.scan_cache_dir",
                   return_value=SimpleNamespace(repos=(repo,))):
            self.assertEqual("active", revision("example/model"))

    def test_revision_is_absent_without_a_main_ref(self) -> None:
        repo = SimpleNamespace(
            repo_id="example/model",
            refs={},
            revisions=(SimpleNamespace(commit_hash="orphan", last_modified=1),),
        )
        with patch("huggingface_hub.scan_cache_dir",
                   return_value=SimpleNamespace(repos=(repo,))):
            self.assertIsNone(revision("example/model"))

    def test_each_model_reproduces_its_recorded_vectors(self) -> None:
        for entry in GOLDEN["entries"]:
            with self.subTest(model=entry["model"]):
                if revision(entry["model"]) != entry["revision"]:
                    self.skipTest(f"{entry['model']} snapshot {entry['revision']} not cached")
                enc = self.encoder(entry)
                for side, got in (("documents", enc.encode(GOLDEN["texts"])),
                                  ("queries", enc.encode_query(GOLDEN["texts"]))):
                    for text, want, have in zip(GOLDEN["texts"], entry[side], got):
                        self.assertGreaterEqual(cosine(want, have), TOLERANCE,
                                                f"{side}: {text}")

    @staticmethod
    def encoder(entry):
        from semsift.embed import OnnxEncoder, StaticEncoder

        if entry["backend"] == "static":
            return StaticEncoder(entry["model"])
        return OnnxEncoder(entry["model"], local_only=True, providers="cpu")


if __name__ == "__main__":
    unittest.main()
