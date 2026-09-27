"""TextChunker and TreeSitterChunker: coverage, bounds, spans, fallback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from semsift.chunk import Chunk, TextChunker, TreeSitterChunker

PROSE = ("# Lease\n\nThe landlord renewed the lease for flat 4B.\nRent is unchanged.\n\n"
         "## Deposit\n\nThe deposit is protected with the scheme.\n\n"
         "## Repairs\n\n" + "The boiler needs a service. " * 30 + "\n")

CODE = ('"""Module doc."""\nimport os\n\ndef small():\n    return 1\n\n\nclass Big:\n'
        + "".join(f"    def m{i}(self):\n        return {i} * compute_value_{i}()\n\n"
                  for i in range(25))
        + "\nx = small()\n")


def assert_spans(test: unittest.TestCase, text: str, chunks: list[Chunk]) -> None:
    """Each chunk's text is exactly its span, and its lines are right."""
    for c in chunks:
        test.assertEqual(text[c.start:c.end], c.text)
        test.assertEqual(text.count("\n", 0, c.start) + 1, c.start_line)
        test.assertEqual(text.count("\n", 0, max(c.start, c.end - 1)) + 1, c.end_line)


def covered(chunks: list[Chunk]) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in chunks]


class TextChunkerTests(unittest.TestCase):
    def test_spans_tile_the_text_and_stay_within_the_bound(self) -> None:
        chunks = TextChunker(max_chars=200, min_chars=0).chunk(PROSE)
        assert_spans(self, PROSE, chunks)
        self.assertEqual(0, chunks[0].start)
        self.assertEqual(len(PROSE), chunks[-1].end)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.end, b.start)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))

    def test_headings_start_chunks(self) -> None:
        chunks = TextChunker(max_chars=200, min_chars=0).chunk(PROSE)
        starts = {c.text.lstrip("\n").split("\n", 1)[0] for c in chunks}
        self.assertIn("## Deposit", starts)
        self.assertIn("## Repairs", starts)

    def test_headings_inside_fenced_code_do_not_start_chunks(self) -> None:
        text = ("introduction\n```python\n# not a heading\nvalue = 1\n```\n"
                "# Real heading\nbody\n")
        chunks = TextChunker(max_chars=200, min_chars=0).chunk(text)
        self.assertEqual(2, len(chunks))
        self.assertIn("# not a heading", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("# Real heading"))

    def test_only_valid_atx_headings_start_chunks(self) -> None:
        text = "first\n#comment\n####### not-a-heading\n# Heading\nbody\n"
        chunks = TextChunker(max_chars=200, min_chars=0).chunk(text)
        self.assertEqual(2, len(chunks))
        self.assertIn("#comment", chunks[0].text)
        self.assertIn("####### not-a-heading", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("# Heading"))

    def test_a_paragraph_break_is_preferred_over_a_mid_paragraph_cut(self) -> None:
        text = ("one two three four five six seven.\n" * 3 + "\n"
                + "eight nine ten eleven twelve.\n" * 3)
        chunks = TextChunker(max_chars=120, min_chars=0).chunk(text)
        self.assertTrue(chunks[0].text.endswith("\n\n") or chunks[0].text.endswith(".\n"))
        self.assertTrue(chunks[1].text.lstrip("\n").startswith("eight"))

    def test_a_line_longer_than_the_bound_is_cut(self) -> None:
        text = "word " * 500
        chunks = TextChunker(max_chars=300, min_chars=0).chunk(text)
        self.assertTrue(all(len(c.text) <= 300 for c in chunks))
        self.assertEqual(text, "".join(c.text for c in chunks))

    def test_chunks_shorter_than_min_chars_are_dropped(self) -> None:
        chunks = TextChunker(max_chars=40, min_chars=30).chunk("tiny\n\n\n" + "x" * 35 + "\n")
        self.assertEqual(1, len(chunks))
        self.assertIn("x" * 35, chunks[0].text)

    def test_empty_and_blank_text_give_no_chunks(self) -> None:
        self.assertEqual([], TextChunker().chunk(""))
        self.assertEqual([], TextChunker().chunk("\n\n   \n"))

    def test_line_numbers_handle_cr_line_endings(self) -> None:
        text = "one\rtwo\rthree"
        chunks = TextChunker(max_chars=5, min_chars=0).chunk(text)
        self.assertEqual([(1, 1), (2, 2), (3, 3)],
                         [(c.start_line, c.end_line) for c in chunks])

    def test_line_numbers_handle_crlf_without_a_trailing_newline(self) -> None:
        text = "one\r\ntwo\r\nthree"
        chunks = TextChunker(max_chars=6, min_chars=0).chunk(text)
        self.assertEqual([(1, 1), (2, 2), (3, 3)],
                         [(c.start_line, c.end_line) for c in chunks])

    def test_bounds_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            TextChunker(max_chars=0)
        with self.assertRaises(ValueError):
            TextChunker(max_chars=10, min_chars=11)


def grammar(language: str) -> bool:
    try:
        import tree_sitter_language_pack as pack
    except ImportError:
        return False
    return language in pack.downloaded_languages()


@unittest.skipUnless(grammar("python"), "tree-sitter extra or python grammar not installed")
class TreeSitterChunkerTests(unittest.TestCase):
    def test_blank_text_tiles_when_min_chars_is_zero(self) -> None:
        text = " \n \n"
        chunks = TreeSitterChunker(max_chars=3, min_chars=0).chunk(text, "python")
        self.assertEqual(text, "".join(c.text for c in chunks))

    def test_spans_tile_the_source_and_stay_within_the_bound(self) -> None:
        chunks = TreeSitterChunker(max_chars=400, min_chars=0).chunk(CODE, "python")
        assert_spans(self, CODE, chunks)
        self.assertEqual(0, chunks[0].start)
        self.assertEqual(len(CODE), chunks[-1].end)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.end, b.start)
        self.assertTrue(all(len(c.text) <= 400 for c in chunks))
        self.assertTrue(all(len(c.text.encode()) <= 400 for c in chunks))

    def test_no_method_is_cut_through_the_middle(self) -> None:
        chunks = TreeSitterChunker(max_chars=400, min_chars=0).chunk(CODE, "python")
        for c in chunks:
            for i in range(25):
                if f"def m{i}(" in c.text:
                    self.assertIn(f"compute_value_{i}()", c.text)

    def test_chunks_carry_their_enclosing_context_and_symbols(self) -> None:
        chunks = TreeSitterChunker(max_chars=400, min_chars=0).chunk(CODE, "python")
        methods = [c for c in chunks if "def m3(" in c.text][0]
        self.assertEqual(("Big",), methods.context)
        self.assertIn("m3", methods.symbols)

    def test_offsets_are_characters_not_bytes(self) -> None:
        text = "s = 'héllo wörld ünïcode'\n" * 40
        chunks = TreeSitterChunker(max_chars=200, min_chars=0).chunk(text, "python")
        assert_spans(self, text, chunks)
        self.assertEqual(len(text), chunks[-1].end)

    def test_metadata_follows_unicode_byte_offset_conversion(self) -> None:
        text = ("label = 'éééééééééé'\n\nclass Café:\n"
                + "".join(f"    def méthode_{i}(self):\n        return {i}\n\n"
                          for i in range(12)))
        chunks = TreeSitterChunker(max_chars=160, min_chars=0).chunk(text, "python")
        method = next(c for c in chunks if "def méthode_6(" in c.text)
        self.assertEqual(("Café",), method.context)
        self.assertIn("méthode_6", method.symbols)

    def test_an_unknown_language_falls_back_to_text_chunking(self) -> None:
        ts = TreeSitterChunker(max_chars=200, min_chars=0)
        self.assertFalse(ts.supports("no-such-language"))
        self.assertEqual(TextChunker(max_chars=200, min_chars=0, markdown=False).chunk(PROSE),
                         ts.chunk(PROSE, "no-such-language"))

    def test_a_syntax_error_still_chunks(self) -> None:
        text = "def f(:\n  return (\n\nclass X\n"
        chunks = TreeSitterChunker(max_chars=200, min_chars=0).chunk(text, "python")
        self.assertEqual(text, "".join(c.text for c in chunks))

    def test_pack_failures_fall_back_to_text_chunking(self) -> None:
        import tree_sitter_language_pack as pack

        ts = TreeSitterChunker(max_chars=200, min_chars=0)
        with patch.object(pack, "process", side_effect=pack.ParseFailedError("failed")):
            self.assertEqual(TextChunker(max_chars=200, min_chars=0).chunk(CODE),
                             ts.chunk(CODE, "python"))

    def test_unexpected_pack_errors_are_not_hidden(self) -> None:
        import tree_sitter_language_pack as pack

        with patch.object(pack, "process", side_effect=RuntimeError("bug")):
            with self.assertRaisesRegex(RuntimeError, "bug"):
                TreeSitterChunker(max_chars=200, min_chars=0).chunk(CODE, "python")

    def test_incomplete_pack_output_falls_back_to_bounded_chunks(self) -> None:
        import tree_sitter_language_pack as pack

        result = SimpleNamespace(chunks=[SimpleNamespace(start_byte=0, metadata=None)])
        text = "x" * 500
        with patch.object(pack, "process", return_value=result):
            chunks = TreeSitterChunker(max_chars=100, min_chars=0).chunk(text, "python")
        self.assertEqual(text, "".join(c.text for c in chunks))
        self.assertTrue(all(len(c.text) <= 100 for c in chunks))



class MarkdownSwitchTests(unittest.TestCase):
    CODE = "x = 1\n# set up the cache\ny = 2\n# and the index\nz = 3\n"

    def test_without_markdown_a_hash_line_is_not_a_heading(self) -> None:
        chunks = TextChunker(max_chars=200, min_chars=0, markdown=False).chunk(self.CODE)
        self.assertEqual([self.CODE], [c.text for c in chunks])

    def test_with_markdown_it_is(self) -> None:
        self.assertGreater(len(TextChunker(max_chars=200, min_chars=0).chunk(self.CODE)), 1)

    def test_the_tree_sitter_fallback_does_not_read_markdown(self) -> None:
        self.assertFalse(TreeSitterChunker().fallback.markdown)


@unittest.skipUnless(grammar("python"), "tree-sitter extra or python grammar not installed")
class LimitTests(unittest.TestCase):
    def test_a_source_over_the_byte_limit_is_chunked_as_text(self) -> None:
        ts = TreeSitterChunker(max_chars=200, min_chars=0, max_source_bytes=100)
        self.assertEqual(ts.fallback.chunk(CODE), ts.chunk(CODE, "python"))

    def test_the_source_limit_counts_utf8_bytes_not_characters(self) -> None:
        import tree_sitter_language_pack as pack

        text = "é" * 60
        ts = TreeSitterChunker(max_chars=200, min_chars=0, max_source_bytes=100)
        with patch.object(pack, "process", side_effect=AssertionError("must not parse")):
            self.assertEqual(ts.fallback.chunk(text), ts.chunk(text, "python"))

    def test_the_limits_reach_the_parser(self) -> None:
        from unittest import mock

        import tree_sitter_language_pack as pack

        seen = []
        real = pack.process

        def spy(text, config):
            seen.append((config.max_source_bytes, config.parse_timeout_ms))
            return real(text, config)

        with mock.patch.object(pack, "process", spy):
            TreeSitterChunker(max_source_bytes=10_000, parse_timeout_ms=1234).chunk(CODE, "python")
        self.assertEqual([(10_000, 1234)], seen)

    def test_defaults_are_set(self) -> None:
        ts = TreeSitterChunker()
        self.assertEqual((5_000_000, 5_000), (ts.max_source_bytes, ts.parse_timeout_ms))

    def test_limits_must_be_positive_integers(self) -> None:
        for name in ("max_source_bytes", "parse_timeout_ms"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TreeSitterChunker(**{name: value})

    def test_a_timeout_or_failed_download_falls_back_to_text(self) -> None:
        from unittest import mock

        import tree_sitter_language_pack as pack

        for error in (pack.ParseTimeoutError("slow"), pack.DownloadError("offline")):
            ts = TreeSitterChunker(max_chars=200, min_chars=0)
            with mock.patch.object(pack, "process", side_effect=error):
                self.assertEqual(ts.fallback.chunk(CODE), ts.chunk(CODE, "python"), repr(error))

if __name__ == "__main__":
    unittest.main()
