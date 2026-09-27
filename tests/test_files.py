"""discover(): which files under a root are candidates for chunking."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from semsift.files import discover


class Tree(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, rel: str, body: str = "x\n") -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def found(self, **kw) -> set[str]:
        return {f.path for f in discover(self.root, **kw)}


class IgnoreFileTests(Tree):
    def test_gitignore_is_honoured_by_default(self) -> None:
        self.write("a.py"); self.write("build/out.py"); self.write(".gitignore", "build/\n")
        self.assertEqual({"a.py", ".gitignore"}, self.found())

    def test_no_ignore_files_means_nothing_is_ignored(self) -> None:
        self.write("a.py"); self.write("build/out.py"); self.write(".gitignore", "build/\n")
        self.assertIn("build/out.py", self.found(ignore_files=()))

    def test_a_nested_ignore_file_applies_only_below_it(self) -> None:
        self.write("pkg/.gitignore", "*.log\n"); self.write("pkg/x.log"); self.write("top.log")
        self.assertEqual({"pkg/.gitignore", "top.log"}, self.found())

    def test_a_nearer_negation_readmits_what_an_outer_file_excluded(self) -> None:
        self.write(".gitignore", "*.log\n"); self.write("keep/.gitignore", "!important.log\n")
        self.write("keep/important.log"); self.write("keep/other.log")
        self.assertIn("keep/important.log", self.found())
        self.assertNotIn("keep/other.log", self.found())

    def test_a_later_ignore_file_settles_disagreements(self) -> None:
        self.write(".gitignore", "*.md\n"); self.write(".myignore", "!README.md\n")
        self.write("README.md"); self.write("NOTES.md")
        got = self.found(ignore_files=(".gitignore", ".myignore"))
        self.assertIn("README.md", got)
        self.assertNotIn("NOTES.md", got)

    def test_utf8_bom_is_not_part_of_the_first_pattern(self) -> None:
        self.write(".gitignore", "\ufeffignored.txt\n")
        self.write("ignored.txt")
        self.assertNotIn("ignored.txt", self.found())


class FilterTests(Tree):
    def test_include_can_prune_directories_and_skip_files(self) -> None:
        self.write("src/a.py"); self.write("src/b.txt"); self.write("node_modules/dep.py")

        def include(path: str, is_dir: bool) -> bool:
            if is_dir:
                return path != "node_modules"
            return path.endswith(".py")

        self.assertEqual({"src/a.py"}, self.found(include=include))

    def test_include_errors_are_not_treated_as_unreadable_entries(self) -> None:
        self.write("a.py")

        def include(path: str, is_dir: bool) -> bool:
            raise OSError("callback failed")

        with self.assertRaisesRegex(OSError, "callback failed"):
            list(discover(self.root, include=include))

    def test_git_is_never_walked(self) -> None:
        self.write(".git/config"); self.write("a.py")
        self.assertEqual({"a.py"}, self.found())

    def test_found_files_carry_size_and_mtime(self) -> None:
        self.write("a.py", "hello\n")
        (f,) = list(discover(self.root))
        self.assertEqual(6, f.size)
        self.assertEqual(os.stat(self.root / "a.py").st_mtime_ns, f.mtime_ns)


class LinkTests(Tree):
    def test_symlinked_directories_are_not_followed(self) -> None:
        self.write("real/a.py")
        os.symlink(self.root / "real", self.root / "loop")
        self.assertEqual({"real/a.py"}, self.found())

    def test_a_broken_symlink_is_skipped(self) -> None:
        self.write("a.py")
        os.symlink(self.root / "missing.py", self.root / "dangling.py")
        self.assertEqual({"a.py"}, self.found())

    def test_a_file_and_its_symlink_are_found_once(self) -> None:
        self.write("a.py")
        os.symlink(self.root / "a.py", self.root / "b.py")
        self.assertEqual(1, len(self.found()))


if __name__ == "__main__":
    unittest.main()
