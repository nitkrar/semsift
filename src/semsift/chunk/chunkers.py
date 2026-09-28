"""Splitting text into chunks whose spans double as citation spans."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import cache


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


#: A line break; its last character is where `_build` places the line end.
_LINE_BREAK = re.compile(r"\r\n?|\n")


@cache
def _pack_languages() -> frozenset[str]:
    import tree_sitter_language_pack as pack

    return frozenset(pack.manifest_languages())


def _build(text: str, spans: list[tuple[int, int]], min_chars: int,
           extras: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None = None) -> list[Chunk]:
    newlines = [m.end() - 1 for m in _LINE_BREAK.finditer(text)]
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


def _check(size: int, min_chars: int, name: str = "max_chars") -> None:
    if size <= 0:
        raise ValueError(f"{name} must be positive")
    if not 0 <= min_chars <= size:
        raise ValueError(f"min_chars must be between 0 and {name}")


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


#: A node smaller than this is emitted whole rather than descended into.
_MIN_NODE_BYTES = 50
#: Recursion bound for pathological nesting.
_MAX_DEPTH = 500


class LanguagePackChunker:
    """Syntax-aligned windows from our own walk of a tree-sitter parse.

    Groups adjacent sibling nodes until the next would pass
    `target_bytes`, descends into any node bigger than that, then merges
    neighbouring groups back up towards the target. `target_bytes` is an
    aim; a node that cannot be split can exceed it, up to `max_bytes`,
    where it is cut. Sizes are UTF-8 bytes, as the tree reports them.
    Chunks cover syntax nodes, so whitespace between groups belongs to no
    chunk; chunks whose stripped text is shorter than `min_chars` are
    dropped.

    Needs the `tree-sitter` extra. A language the pack does not list, a
    grammar it cannot download, a parse failure, or a source over
    `max_source_bytes` falls back to `fallback`: a TextChunker that does
    not read markdown, by default.
    """

    def __init__(self, target_bytes: int = 750, max_bytes: int = 20_000,
                 min_chars: int = 1, fallback: TextChunker | None = None, *,
                 max_source_bytes: int = 5_000_000) -> None:
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError as exc:
            raise ImportError("LanguagePackChunker needs the tree-sitter extra:"
                              " pip install 'semsift[tree-sitter]'") from exc
        for name, value in (("target_bytes", target_bytes), ("max_bytes", max_bytes),
                            ("max_source_bytes", max_source_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_bytes < 4:
            raise ValueError("max_bytes must fit one UTF-8 character")
        if max_bytes < target_bytes:
            raise ValueError("max_bytes must be at least target_bytes")
        if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 0:
            raise ValueError("min_chars must be a non-negative integer")
        if min_chars > max_bytes:
            raise ValueError("min_chars must not exceed max_bytes")
        self.target_bytes, self.max_bytes, self.min_chars = target_bytes, max_bytes, min_chars
        self.max_source_bytes = max_source_bytes
        fallback_size = max(target_bytes, min_chars)
        self.fallback = fallback or TextChunker(fallback_size, min_chars, markdown=False)

    def supports(self, language: str) -> bool:
        """Whether the pack knows `language`; its grammar may still need a download."""
        return language in _pack_languages()

    def chunk(self, text: str, language: str) -> list[Chunk]:
        import tree_sitter_language_pack as pack

        data = text.encode()
        if not text.strip() or not self.supports(language) or len(data) > self.max_source_bytes:
            return self.fallback.chunk(text)
        try:
            tree = pack.get_parser(language).parse(data)
        except pack.Error:
            return self.fallback.chunk(text)
        spans = _merge_adjacent(_split_node(tree.root_node, self.target_bytes, 0),
                                self.target_bytes)
        spans = [(_char_start(data, a), _char_start(data, b))
                 for a, b in _bounded(spans, self.max_bytes, data)]
        chars = _char_offsets(data, sorted({o for span in spans for o in span}))
        return _build(text, [(chars[a], chars[b]) for a, b in spans if b > a],
                      self.min_chars)


def _split_node(node, target: int, depth: int) -> list[tuple[int, int]]:
    """Group a node's children into byte spans aiming at `target`."""
    if (not node.children or depth > _MAX_DEPTH
            or node.end_byte - node.start_byte < _MIN_NODE_BYTES):
        return [(node.start_byte, node.end_byte)]
    groups: list[tuple[int, int]] = []
    children = node.children
    i = 0
    while i < len(children):
        start, end = children[i].start_byte, children[i].end_byte
        size = end - start
        i += 1
        if size > target:
            groups.extend(_split_node(children[i - 1], target, depth + 1))
            continue
        while i < len(children):
            nxt = children[i]
            if size + (nxt.end_byte - nxt.start_byte) > target:
                break
            end = nxt.end_byte
            size += nxt.end_byte - nxt.start_byte
            i += 1
        groups.append((start, end))
    return groups


def _merge_adjacent(spans: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Coalesce neighbouring spans back up towards `target`.

    Splitting alone leaves many small spans, because a node's children
    are often individually tiny; merging is what makes sizes uniform.
    """
    if not spans:
        return []
    out: list[tuple[int, int]] = []
    start, end = spans[0]
    for nxt_start, nxt_end in spans[1:]:
        if (end - start) + (nxt_end - nxt_start) > target:
            out.append((start, end))
            start, end = nxt_start, nxt_end
            continue
        end = nxt_end
    out.append((start, end))
    return out


def _bounded(spans: list[tuple[int, int]], limit: int, data: bytes) -> list[tuple[int, int]]:
    """Cut any span over `limit` bytes into pieces that fit, at character boundaries."""
    out: list[tuple[int, int]] = []
    for start, end in spans:
        while end - start > limit:
            cut = _char_start(data, start + limit)
            out.append((start, cut))
            start = cut
        out.append((start, end))
    return out


class TreeSitterPackChunker:
    """Syntax-aligned chunks from tree-sitter-language-pack's chunker.

    The pack cuts chunks of at most `target_chars`; adjacent small ones
    are then merged back up towards it. A merged chunk keeps the first
    piece's context and the symbols of all of them. A pack piece over the
    target means incomplete output and falls back. Needs the `tree-sitter`
    extra. The language pack downloads a
    grammar the first time a language is parsed. A language it does not
    know, a grammar it cannot download, or a failure to parse falls back
    to `fallback`: a TextChunker of the same bounds that does not read
    markdown, by default. So does a source over `max_source_bytes`, or a
    parse that runs past `parse_timeout_ms`, so one generated or minified
    file cannot stall indexing.
    """

    def __init__(self, target_chars: int = 750, min_chars: int = 1,
                 fallback: TextChunker | None = None, *,
                 max_source_bytes: int = 5_000_000, parse_timeout_ms: int = 5_000) -> None:
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError as exc:
            raise ImportError("TreeSitterPackChunker needs the tree-sitter extra:"
                              " pip install 'semsift[tree-sitter]'") from exc
        _check(target_chars, min_chars, "target_chars")
        if (isinstance(max_source_bytes, bool)
                or not isinstance(max_source_bytes, int) or max_source_bytes <= 0
                or isinstance(parse_timeout_ms, bool)
                or not isinstance(parse_timeout_ms, int) or parse_timeout_ms <= 0):
            raise ValueError("max_source_bytes and parse_timeout_ms must be positive integers")
        self.target_chars, self.min_chars = target_chars, min_chars
        self.max_source_bytes, self.parse_timeout_ms = max_source_bytes, parse_timeout_ms
        self.fallback = fallback or TextChunker(target_chars, min(min_chars, target_chars),
                                                markdown=False)

    def supports(self, language: str) -> bool:
        """Whether the pack knows `language`; its grammar may still need a download."""
        return language in _pack_languages()

    def chunk(self, text: str, language: str) -> list[Chunk]:
        import tree_sitter_language_pack as pack

        if not text.strip():
            return self.fallback.chunk(text)
        if not self.supports(language) or len(text.encode()) > self.max_source_bytes:
            return self.fallback.chunk(text)
        try:
            result = pack.process(text, pack.ProcessConfig(
                language=language, structure=False, imports=False, exports=False,
                chunk_max_size=self.target_chars, max_source_bytes=self.max_source_bytes,
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
        # The pack was asked for pieces of at most target_chars; a larger
        # one means its output is incomplete.
        if any(end - start > self.target_chars for start, end in spans):
            return self.fallback.chunk(text)
        spans, extras = _merge_up(spans, extras, self.target_chars)
        return _build(text, spans, self.min_chars, extras)


def _merge_up(spans, extras, target: int):
    """Join adjacent spans while the joined span stays within `target`."""
    merged: list[tuple[int, int]] = []
    meta: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for start, end in spans:
        context, symbols = extras.get(start, ((), ()))
        if merged and end - merged[-1][0] <= target:
            first = merged[-1][0]
            merged[-1] = (first, end)
            old_context, old_symbols = meta[first]
            meta[first] = (old_context, old_symbols + tuple(
                s for s in symbols if s not in old_symbols))
            continue
        merged.append((start, end))
        meta[start] = (context, symbols)
    return merged, meta


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
