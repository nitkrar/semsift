"""Choosing which files under a root are candidates for chunking."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence


@dataclass(frozen=True)
class Found:
    """A file `discover` yields; `path` is relative to the root, `/`-separated."""

    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _Spec:
    base: Path
    spec: object


def discover(root: Path | str, *, ignore_files: Sequence[str] = (".gitignore",),
             include: Callable[[str, bool], bool] | None = None) -> Iterator[Found]:
    """Yield the files under `root` that no ignore file excludes and `include` accepts.

    `ignore_files` are read in every directory, in gitignore syntax, and
    apply to that directory and below. Within one directory a later file
    settles disagreements with an earlier one, and a `!` pattern in a
    nearer directory re-admits what an outer one excluded. `include` is
    called with each relative path and whether it is a directory; False
    prunes a directory or skips a file.

    `.git` is never walked. Symlinked directories are not followed, which
    bounds the walk; a symlinked file is followed, and a file reached by
    two paths is yielded once. Unreadable entries are skipped.
    """
    from pathspec import GitIgnoreSpec

    root = Path(root)
    stack: list[tuple[Path, tuple[_Spec, ...]]] = [(root, ())]
    seen: set[str] = set()
    while stack:
        directory, inherited = stack.pop()
        lines: list[str] = []
        for name in ignore_files:
            lines += _read_lines(directory / name)
        if lines:
            inherited = (*inherited, _Spec(directory, GitIgnoreSpec.from_lines(lines)))
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            try:
                path = Path(entry.path)
                rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if (entry.name != ".git" and not _ignored(path, inherited, is_dir=True)
                        and (include is None or include(rel, True))):
                    stack.append((path, inherited))
                continue
            if _ignored(path, inherited, is_dir=False):
                continue
            if include is not None and not include(rel, False):
                continue
            try:
                real = os.path.realpath(entry.path)
                if real in seen:
                    continue
                st = entry.stat(follow_symlinks=True)
            except OSError:
                continue
            # A symlink to a directory is not a directory to
            # is_dir(follow_symlinks=False), so it arrives here.
            if not stat.S_ISREG(st.st_mode):
                continue
            seen.add(real)
            yield Found(rel, st.st_size, st.st_mtime_ns)


def _ignored(path: Path, specs: Sequence[_Spec], is_dir: bool) -> bool:
    """The last verdict of every pattern of every inherited spec, in order.

    Keeping the last verdict across specs, rather than asking each spec
    alone, is what lets a nearer `!` pattern override an outer exclusion.
    """
    ignored = False
    for entry in specs:
        try:
            relative = path.relative_to(entry.base)
        except ValueError:
            continue
        text = relative.as_posix() + ("/" if is_dir else "")
        for pattern in entry.spec.patterns:
            if pattern.include is not None and pattern.match_file(text) is not None:
                ignored = pattern.include
    return ignored


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except OSError:
        return []
