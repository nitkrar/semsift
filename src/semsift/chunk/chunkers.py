"""Splitting text into chunks whose spans double as citation spans."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """`text` is exactly `source[start:end]`; lines are 1-based and inclusive.

    `context` is the enclosing definitions (e.g. a class name) and
    `symbols` the names defined inside, when the chunker knows them.
    """

    text: str
    start: int
    end: int
    start_line: int
    end_line: int
    context: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()


def _build(text: str, spans: list[tuple[int, int]], min_chars: int,
           extras: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None = None) -> list[Chunk]:
    newlines = [i for i, ch in enumerate(text)
                if ch == "\n" or (ch == "\r" and (i + 1 == len(text) or text[i + 1] != "\n"))]
    out = []
    for start, end in spans:
        body = text[start:end]
        if len(body.strip()) < min_chars:
            continue
        context, symbols = (extras or {}).get(start, ((), ()))
        out.append(Chunk(body, start, end, bisect_right(newlines, start - 1) + 1,
                         bisect_right(newlines, max(start, end - 1) - 1) + 1,
                         context, symbols))
    return out


def _check(max_chars: int, min_chars: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= min_chars <= max_chars:
        raise ValueError("min_chars must be between 0 and max_chars")


class TextChunker:
    """Line-based windows of at most `max_chars`, split at natural breaks.

    With `markdown`, an ATX heading outside a fenced block always starts a
    chunk; turn it off for text where `#` begins a comment. After a blank
    line, a chunk at least half full ends before the next paragraph. A
    line longer than `max_chars` is cut into pieces. Chunks whose stripped
    text is shorter than `min_chars` are dropped, so the spans tile the
    text only when `min_chars` is 0.
    """

    def __init__(self, max_chars: int = 750, min_chars: int = 1, *,
                 markdown: bool = True) -> None:
        _check(max_chars, min_chars)
        self.max_chars, self.min_chars, self.markdown = max_chars, min_chars, markdown

    def spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = pos = 0
        after_blank = False
        fence: tuple[str, int] | None = None
        for line in text.splitlines(keepends=True):
            size = pos - start
            blank = not line.strip()
            marker = _markdown_fence(line)
            heading = self.markdown and fence is None and _markdown_heading(line)
            if size and (heading
                         or (after_blank and not blank and size >= self.max_chars // 2)
                         or size + len(line) > self.max_chars):
                spans.append((start, pos))
                start = pos
            if len(line) > self.max_chars:
                for cut in range(pos, pos + len(line), self.max_chars):
                    spans.append((cut, min(cut + self.max_chars, pos + len(line))))
                start = pos + len(line)
            pos += len(line)
            after_blank = blank
            if marker is not None:
                char, width, rest = marker
                if fence is None:
                    fence = (char, width)
                elif char == fence[0] and width >= fence[1] and not rest.strip():
                    fence = None
        if pos > start:
            spans.append((start, pos))
        return spans

    def chunk(self, text: str) -> list[Chunk]:
        return _build(text, self.spans(text), self.min_chars)


class TreeSitterChunker:
    """Syntax-aligned chunks from tree-sitter-language-pack's chunker.

    Needs the `tree-sitter` extra. The language pack downloads a
    grammar the first time a language is parsed. A language it does not
    know, a grammar it cannot download, or a failure to parse falls back
    to `fallback`: a TextChunker of the same bounds that does not read
    markdown, by default. So does a source over `max_source_bytes`, or a
    parse that runs past `parse_timeout_ms`, so one generated or minified
    file cannot stall indexing.
    """

    def __init__(self, max_chars: int = 750, min_chars: int = 1,
                 fallback: TextChunker | None = None, *,
                 max_source_bytes: int = 5_000_000, parse_timeout_ms: int = 5_000) -> None:
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError as exc:
            raise ImportError("TreeSitterChunker needs the tree-sitter extra:"
                              " pip install 'semsift[tree-sitter]'") from exc
        _check(max_chars, min_chars)
        if (isinstance(max_source_bytes, bool)
                or not isinstance(max_source_bytes, int) or max_source_bytes <= 0
                or isinstance(parse_timeout_ms, bool)
                or not isinstance(parse_timeout_ms, int) or parse_timeout_ms <= 0):
            raise ValueError("max_source_bytes and parse_timeout_ms must be positive integers")
        self.max_chars, self.min_chars = max_chars, min_chars
        self.max_source_bytes, self.parse_timeout_ms = max_source_bytes, parse_timeout_ms
        self.fallback = fallback or TextChunker(max_chars, min_chars, markdown=False)

    def supports(self, language: str) -> bool:
        """Whether the pack knows `language`; its grammar may still need a download."""
        import tree_sitter_language_pack as pack

        return language in pack.manifest_languages()

    def chunk(self, text: str, language: str) -> list[Chunk]:
        import tree_sitter_language_pack as pack

        if not text.strip():
            return self.fallback.chunk(text)
        if not self.supports(language) or len(text.encode()) > self.max_source_bytes:
            return self.fallback.chunk(text)
        try:
            result = pack.process(text, pack.ProcessConfig(
                language=language, structure=False, imports=False, exports=False,
                chunk_max_size=self.max_chars, max_source_bytes=self.max_source_bytes,
                parse_timeout_ms=self.parse_timeout_ms))
        except pack.Error:
            return self.fallback.chunk(text)
        pieces = sorted(result.chunks, key=lambda c: c.start_byte)
        if not pieces:
            return self.fallback.chunk(text)
        data = text.encode()
        # Chunks are cut at their start bytes, so they tile the text even
        # if the pack leaves a gap; each start is moved back to a
        # character boundary before converting to a character offset.
        starts = sorted({0} | {_char_start(data, c.start_byte) for c in pieces})
        chars = _char_offsets(data, starts)
        extras = {}
        for c in pieces:
            meta = c.metadata
            if meta is not None:
                key = chars[_char_start(data, c.start_byte)]
                extras.setdefault(key, (tuple(meta.context_path), tuple(meta.symbols_defined)))
        bounds = [chars[b] for b in starts] + [len(text)]
        spans = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
        if any(end - start > self.max_chars for start, end in spans):
            return self.fallback.chunk(text)
        return _build(text, spans, self.min_chars, extras)


def _char_start(data: bytes, offset: int) -> int:
    """`offset` moved back to the first byte of the UTF-8 character it is in."""
    offset = min(max(offset, 0), len(data))
    while 0 < offset < len(data) and data[offset] & 0xC0 == 0x80:
        offset -= 1
    return offset


def _char_offsets(data: bytes, byte_offsets: list[int]) -> dict[int, int]:
    """Character offset of each sorted byte offset, which must be on a boundary."""
    out, chars, prev = {}, 0, 0
    for b in byte_offsets:
        chars += len(data[prev:b].decode())
        out[b] = chars
        prev = b
    return out


def _markdown_heading(line: str) -> bool:
    body = line.rstrip("\r\n")
    stripped = body.lstrip(" ")
    if len(body) - len(stripped) > 3:
        return False
    width = len(stripped) - len(stripped.lstrip("#"))
    return 1 <= width <= 6 and (width == len(stripped) or stripped[width] in " \t")


def _markdown_fence(line: str) -> tuple[str, int, str] | None:
    body = line.rstrip("\r\n")
    stripped = body.lstrip(" ")
    if len(body) - len(stripped) > 3 or not stripped or stripped[0] not in "`~":
        return None
    char = stripped[0]
    width = len(stripped) - len(stripped.lstrip(char))
    if width < 3:
        return None
    rest = stripped[width:]
    if char == "`" and "`" in rest:
        return None
    return char, width, rest
