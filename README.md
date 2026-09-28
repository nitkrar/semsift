# semsift

Building blocks for hybrid retrieval: find files, chunk text, embed it, store vectors
and a keyword index together in SQLite or memory, search both, fuse the
ranked results and rerank them. It retrieves and ranks; it never
generates text.

**Status:** Implemented, not yet released. See
[docs/design.md](docs/design.md) for the blocks and their contracts, and
[docs/features.md](docs/features.md) for what was taken from other
projects.

```python
import sqlite3

from semsift.embed import StaticEncoder
from semsift.search import KeywordSource, Search, VectorSource
from semsift.store import Field, Item, Store

conn = sqlite3.connect(":memory:", isolation_level=None)
store = Store(conn, "notes", [Field("source", "text")],
              encoder=StaticEncoder("minishlab/potion-base-8M"))

items = [Item(1, "Rent is due on the first of the month.", {"source": "lease.md"}),
         Item(2, "The boiler service is booked for Tuesday.", {"source": "home.md"})]
vectors = store.embed(items)          # encode before opening the transaction
conn.execute("BEGIN")
store.upsert(items, vectors)          # the store never commits; you do
conn.execute("COMMIT")

search = Search(store, sources=[VectorSource(store), KeywordSource(store)],
                citation=("source",))
for hit in search.run("when is rent due", k=1).hits:
    print(hit.citation["source"], hit.text)
```

`pip install semsift` includes the static (model2vec) encoders. Extras
add the rest: `onnx`, `webgpu`, and `tree-sitter` for syntax-aware
chunking.

## Development

```
uv venv .venv && uv pip install --python .venv/bin/python -e ".[onnx,tree-sitter]"
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python benchmarks/run.py
```

To use an unreleased semsift from another project, depend on the
checkout (`uv add --editable ../semsift`, or `pip install -e ../semsift`),
or build it (`uv build`) and install the wheel from `dist/`.

## Releasing

The version lives only in `semsift.__version__`. Bump it, commit, and push
a matching tag (`v0.0.4`). The release workflow runs the tests, builds,
refuses to publish unless the tag, `__version__`, the sdist and the wheel
agree, publishes to PyPI through trusted publishing, and creates the
GitHub release. PyPI must list this repository's `release.yml` as a
trusted publisher for the `semsift` project.
