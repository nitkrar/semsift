"""The cross-encoder reranker: pair scores, the top-N cut, blending, the cache."""

from __future__ import annotations

import math
import unittest

from semsift.rerank import Candidate, CrossEncoder


def cands(*texts, scores=None):
    scores = scores or [1.0 / (n + 1) for n in range(len(texts))]
    return [Candidate(n + 1, s, text=t) for n, (t, s) in enumerate(zip(texts, scores))]


class ScoredBy:
    """A pair scorer from a table of text -> logit, counting what it scores."""

    def __init__(self, table):
        self.table, self.pairs = table, []

    def __call__(self, pairs):
        self.pairs.extend(pairs)
        return [self.table[text] for _, text in pairs]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class RerankingTests(unittest.TestCase):
    def test_candidates_are_reordered_by_pair_score(self) -> None:
        scorer = ScoredBy({"weak": -2.0, "strong": 5.0, "middling": 1.0})
        got = CrossEncoder(scorer=scorer).rerank("q", cands("weak", "strong", "middling"))
        self.assertEqual(["strong", "middling", "weak"], [c.text for c in got])
        self.assertAlmostEqual(sigmoid(5.0), got[0].score)

    def test_only_the_top_are_scored_and_the_rest_are_not_returned(self) -> None:
        scorer = ScoredBy({"a": 0.0, "b": 1.0, "c": 9.0})
        got = CrossEncoder(scorer=scorer, top=2).rerank("q", cands("a", "b", "c"))
        self.assertEqual(["b", "a"], [c.text for c in got])
        self.assertEqual([("q", "a"), ("q", "b")], scorer.pairs)

    def test_weight_blends_with_the_normalised_incoming_score(self) -> None:
        # Incoming scores normalise to 1.0 and 0.0; pair scores favour the second.
        scorer = ScoredBy({"first": 0.0, "second": 1.0})
        got = CrossEncoder(scorer=scorer, weight=0.25).rerank(
            "q", cands("first", "second", scores=[0.8, 0.2]))
        want = {"first": 0.25 * sigmoid(0.0) + 0.75 * 1.0,
                "second": 0.25 * sigmoid(1.0) + 0.75 * 0.0}
        self.assertEqual(["first", "second"], [c.text for c in got])
        for c in got:
            self.assertAlmostEqual(want[c.text], c.score)

    def test_equal_incoming_scores_normalise_to_one(self) -> None:
        scorer = ScoredBy({"x": 0.0, "y": 0.0})
        got = CrossEncoder(scorer=scorer, weight=0.5).rerank("q", cands("x", "y", scores=[0.3, 0.3]))
        self.assertEqual([0.5 * 0.5 + 0.5] * 2, [c.score for c in got])

    def test_equal_pair_scores_keep_the_incoming_order(self) -> None:
        scorer = ScoredBy({"p": 2.0, "q": 2.0, "r": 2.0})
        got = CrossEncoder(scorer=scorer).rerank("query", cands("p", "q", "r"))
        self.assertEqual(["p", "q", "r"], [c.text for c in got])

    def test_a_pair_is_scored_once_across_searches(self) -> None:
        scorer = ScoredBy({"a": 1.0, "b": 2.0})
        reranker = CrossEncoder(scorer=scorer)
        reranker.rerank("q", cands("a", "b"))
        reranker.rerank("q", cands("b", "a"))
        reranker.rerank("other", cands("a"))
        self.assertEqual([("q", "a"), ("q", "b"), ("other", "a")], scorer.pairs)

    def test_the_cache_is_bounded(self) -> None:
        scorer = ScoredBy({"a": 1.0, "b": 2.0})
        reranker = CrossEncoder(scorer=scorer, cache_size=1)
        reranker.rerank("q", cands("a"))
        reranker.rerank("q", cands("b"))
        reranker.rerank("q", cands("a"))
        self.assertEqual([("q", "a"), ("q", "b"), ("q", "a")], scorer.pairs)

    def test_a_candidate_without_text_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "text"):
            CrossEncoder(scorer=ScoredBy({})).rerank("q", [Candidate(1, 1.0)])

    def test_settings_are_checked(self) -> None:
        for kwargs in ({"top": 0}, {"top": True}, {"weight": 1.5}, {"weight": -0.1},
                       {"cache_size": -1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    CrossEncoder(scorer=ScoredBy({}), **kwargs)


class ModelTests(unittest.TestCase):
    """Guarded: needs onnxruntime and the cached model. Skips, never fails,
    so the suite stays offline-safe."""

    MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def setUp(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            self.skipTest("onnx extra not installed")
        try:
            for name in ("tokenizer.json", "onnx/model.onnx"):
                hf_hub_download(self.MODEL, name, local_files_only=True)
        except LocalEntryNotFoundError:
            self.skipTest("model unavailable")
        self.reranker = CrossEncoder(self.MODEL, local_only=True, providers="cpu")

    def test_raw_scores_match_the_model_card(self) -> None:
        """The card's sentence-transformers output for these two pairs."""
        query = "How many people live in Berlin?"
        got = self.reranker.logits([
            (query, "Berlin had a population of 3,520,031 registered inhabitants"
                    " in an area of 891.82 square kilometers."),
            (query, "Berlin is well known for its museums."),
        ])
        self.assertAlmostEqual(8.607138, got[0], places=3)
        self.assertAlmostEqual(-4.320078, got[1], places=3)

    def test_the_passage_that_answers_ranks_first(self) -> None:
        query = "Which planet is known as the Red Planet?"
        got = self.reranker.rerank(query, cands(
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        ))
        self.assertTrue(got[0].text.startswith("Mars"))


if __name__ == "__main__":
    unittest.main()
