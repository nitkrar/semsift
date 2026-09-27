"""Regenerate tests/golden/vectors.json from locally cached models.

    .venv/bin/python -m tests.golden.make_vectors

Run it only when a change to semsift is meant to move vectors, and say so
in the commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from semsift.embed import OnnxEncoder, StaticEncoder

from tests.golden.models import MODELS, TEXTS, revision

OUT = Path(__file__).with_name("vectors.json")


def main() -> None:
    entries = []
    for model, backend in MODELS:
        enc = (StaticEncoder(model) if backend == "static"
               else OnnxEncoder(model, local_only=True, providers="cpu"))
        entries.append({
            "model": model, "backend": backend, "revision": revision(model),
            "documents": [[round(x, 6) for x in v] for v in enc.encode(TEXTS)],
            "queries": [[round(x, 6) for x in v] for v in enc.encode_query(TEXTS)],
        })
    OUT.write_text(json.dumps({"texts": TEXTS, "entries": entries}, indent=1) + "\n")


if __name__ == "__main__":
    main()
