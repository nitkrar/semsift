"""Import rules: embed stands alone; third-party imports are declared."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "semsift"
PYPROJECT = SRC.parents[1] / "pyproject.toml"


def _package_names(requirements: list[str]) -> set[str]:
    return {
        re.match(r"[A-Za-z0-9_.-]+", requirement).group().replace("-", "_")
        for requirement in requirements
    }


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def _imports(path: Path) -> list[tuple[int, int, str]]:
    """(line, relative level, top-level module) for every import."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            out.append((node.lineno, node.level, (node.module or "").split(".")[0]))
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, 0, a.name.split(".")[0]) for a in node.names)
    return out


class BoundaryTests(unittest.TestCase):
    def test_embed_imports_nothing_else_from_semsift(self) -> None:
        files = sorted((SRC / "embed").rglob("*.py"))
        self.assertTrue(files)
        found = [f"{f.name}:{line}" for f in files
                 for line, level, top in _imports(f)
                 if level >= 2 or (level == 0 and top == "semsift")]
        self.assertEqual([], found)

    def test_third_party_imports_are_declared(self) -> None:
        project = _project()
        requirements = list(project.get("dependencies", ()))
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        third_party = _package_names(requirements)
        allowed = set(sys.stdlib_module_names) | third_party | {"__future__", "semsift"}
        found = [f"{f.relative_to(SRC)}:{line} {top}"
                 for f in sorted(SRC.rglob("*.py"))
                 for line, level, top in _imports(f)
                 if level == 0 and top not in allowed]
        self.assertEqual([], found)

    def test_webgpu_extra_also_installs_the_onnx_backend(self) -> None:
        extras = _project()["optional-dependencies"]
        self.assertLessEqual(
            _package_names(extras["onnx"]) - {"onnxruntime"},
            _package_names(extras["webgpu"]),
        )


if __name__ == "__main__":
    unittest.main()
