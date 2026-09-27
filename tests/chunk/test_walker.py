"""LanguagePackChunker: syntax-aligned windows from our own walk of the tree."""

from __future__ import annotations

import unittest

from semsift.chunk import TextChunker, LanguagePackChunker


def grammar(language: str) -> bool:
    try:
        import tree_sitter_language_pack as pack
    except ImportError:
        return False
    return language in pack.downloaded_languages()


def spans_ok(test, text, chunks) -> None:
    for c in chunks:
        test.assertEqual(text[c.start:c.end], c.text)
        test.assertEqual(text.count("\n", 0, c.start) + 1, c.start_line)
        test.assertEqual(text.count("\n", 0, max(c.start, c.end - 1)) + 1, c.end_line)


@unittest.skipUnless(grammar("python"), "tree-sitter extra or python grammar not installed")
class WalkerTests(unittest.TestCase):
    def test_small_siblings_merge_up_to_the_target(self) -> None:
        text = "".join(f"value_{i} = compute({i})\n" for i in range(200))
        chunks = LanguagePackChunker(target_bytes=400, min_chars=0).chunk(text, "python")
        spans_ok(self, text, chunks)
        sizes = [len(c.text.encode()) for c in chunks]
        # Merged, not one chunk per statement: every chunk but the last is
        # near the target. The target counts node bytes, so the newlines
        # between merged nodes can carry a chunk a little past it.
        self.assertTrue(all(300 < s <= 440 for s in sizes[:-1]), sizes)

    def test_an_oversized_definition_is_split_along_its_children(self) -> None:
        text = "class Big:\n" + "".join(
            f"    def m{i}(self):\n        return {i} * compute_value_{i}()\n\n" for i in range(40))
        chunks = LanguagePackChunker(target_bytes=300, min_chars=0).chunk(text, "python")
        spans_ok(self, text, chunks)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            for i in range(40):
                if f"def m{i}(" in c.text:
                    self.assertIn(f"compute_value_{i}()", c.text)

    def test_a_node_too_big_to_split_is_cut_at_the_ceiling(self) -> None:
        text = "blob = '" + "x" * 5000 + "'\n"
        chunks = LanguagePackChunker(target_bytes=300, max_bytes=1000, min_chars=0).chunk(text, "python")
        self.assertTrue(all(len(c.text.encode()) <= 1000 for c in chunks))
        self.assertEqual(text.rstrip("\n"), "".join(c.text for c in chunks).rstrip("\n"))

    def test_the_target_is_an_aim_and_the_ceiling_a_bound(self) -> None:
        # The string's 600-byte content node cannot be split, so a 300-byte
        # target leaves it whole under a 1000-byte ceiling.
        text = "s = '" + "y" * 600 + "'\n"
        chunks = LanguagePackChunker(target_bytes=300, max_bytes=1000, min_chars=0).chunk(text, "python")
        self.assertIn("y" * 600, [c.text for c in chunks])

    def test_chunks_below_min_chars_are_dropped(self) -> None:
        text = "x = 1\n\n\n" + "def long_function_name():\n    return 'enough text to keep'\n"
        chunks = LanguagePackChunker(target_bytes=40, min_chars=30).chunk(text, "python")
        self.assertTrue(all(len(c.text.strip()) >= 30 for c in chunks))

    def test_offsets_are_characters(self) -> None:
        text = "s = 'héllo wörld ünïcode'\n" * 60
        chunks = LanguagePackChunker(target_bytes=200, min_chars=0).chunk(text, "python")
        spans_ok(self, text, chunks)

    def test_unknown_language_or_oversized_source_falls_back_to_text(self) -> None:
        text = "# Title\n\nsome prose here\n" * 10
        ts = LanguagePackChunker(target_bytes=100, min_chars=0)
        self.assertFalse(ts.fallback.markdown)
        self.assertEqual(ts.fallback.chunk(text), ts.chunk(text, "no-such-language"))
        small = LanguagePackChunker(target_bytes=100, min_chars=0, max_source_bytes=50)
        self.assertEqual(small.fallback.chunk(text), small.chunk(text, "python"))

    def test_default_fallback_preserves_min_chars_above_its_normal_size(self) -> None:
        chunks = LanguagePackChunker(min_chars=1000).chunk("x" * 900, "no-such-language")
        self.assertEqual([], chunks)

    def test_default_fallback_uses_the_target_size(self) -> None:
        chunks = LanguagePackChunker(target_bytes=100, max_bytes=500, min_chars=0).chunk(
            "x" * 250, "no-such-language")
        self.assertEqual([100, 100, 50], [len(chunk.text) for chunk in chunks])

    def test_bounds_are_checked(self) -> None:
        for kw in ({"target_bytes": 0}, {"max_bytes": 3, "target_bytes": 3},
                   {"max_bytes": 10, "target_bytes": 20},
                   {"max_bytes": 20, "target_bytes": 10, "min_chars": 21},
                   {"min_chars": -1}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                LanguagePackChunker(**kw)


if __name__ == "__main__":
    unittest.main()
