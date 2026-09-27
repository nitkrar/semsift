# Features taken from other projects

**Status:** Proposed. Each row names the project it comes from and where
it lands. Rows marked *design* are part of [design.md](design.md); the
rest are not yet designed.

Sources: retriv (https://github.com/AmenRa/retriv), retrivo on PyPI
(https://pypi.org/project/retrivo/), Retrivo on GitHub
(https://github.com/matorverse/Retrivo), semshift
(https://github.com/VeerajSai/SemShift).

## In the library

| Feature | From | Where | Effort |
|---|---|---|---|
| Linear blend with selectable normalisation (`min-max`, `max`, `sum`, `z-score`) | retriv `retriv/merger/normalization.py` | design: fuse | S |
| Typed metadata filters compiled to SQL, applied before keyword and vector search | retriv `retriv/experimental/advanced_retriever.py` (which filters the keyword side only) | design: store | M |
| Hits carry a source key, chunk ordinal and snippet for citation | Retrivo `lib/llm.ts` | design: search | S |
| `warnings` on the result | semshift `semantic_diff.py` (`SemanticDiffResult.warnings`) | design: search | S |
| MMR with relevance normalised before mixing | retrivo `retrivo/features/mmr.py` (which divides by the max only) | design: rerank | S |
| Embedding cache keyed on full text and `VectorSpace` | both Retrivos key on a text prefix or omit the model | design: embed | S |
| FTS5 tokenizer as a store option (`porter` for prose) | retriv stemming and stopwords (`docs/text_preprocessing.md`) | design: store | S |
| Accept precomputed vectors, still recording their `VectorSpace` | retriv `embeddings_path` (`dense_retriever.py`) | store `upsert` | S |
| Allowlist of embedder model ids from the environment | semshift `SEMSHIFT_ALLOWED_MODELS` (`embeddings.py`) | embed | S |

## Evaluation

The harness belongs to semsift (see [design.md](design.md#evals)); each
consumer brings its own labelled data.

| Feature | From | Effort |
|---|---|---|
| Tune fusion weights, RRF `k`, candidate depth and normalisation from labelled queries, on a held-out split | retriv `Merger.autotune` with ranx `optimize_fusion`; retriv tunes on its evaluation queries, which overfits | M |
| Query slices: should-return-nothing, held-out; keyword-only and vector-only baselines | semshift `docs/benchmarks.md`, `run_baselines.py` | S |
| Save each source's ranked list per run, so fusion and reranking can be swept offline | retriv `bsearch(path=...)` | S |
| ranx as a dev-only metrics dependency | retriv | S |
| Metric helpers (Hit@K, MRR, NDCG) over labelled queries keyed by stable ids | Retrivo `eval/metrics.ts`, `eval/golden-set.schema.json` | S |

## Later

| Feature | From | Effort |
|---|---|---|
| Cross-encoder reranker with a pair-score cache keyed on query, document and model | retrivo `retrivo/search/reranker.py` | M |
| Semantic-boundary chunking for prose (RAG pipeline, not semsift) | retrivo `retrivo/chunking/semantic.py` | M |

## Declined

The reasons are in [design.md](design.md#rejected-alternatives) where
they shape the design. The rest:

| Item | From | Reason |
|---|---|---|
| Faiss, torch, transformers | retriv | Heavy for exhaustive-scale corpora; vicinity covers approximate search if needed |
| Full-rebuild-only indexing | retriv | Unusable for refresh and for an append-heavy memory store |
| Intent routing, suffix query expansion | retrivo | Resembles repoglass's measured and rejected `tier_routing`; FTS5 stemming covers expansion |
| TF-IDF refit per call as an embedder | semshift | Vectors depend on the batch, which breaks `VectorSpace` |
| Regex claim, risk and taint rules | semshift, semsift | Domain heuristics, not retrieval |
| Sentence-per-chunk splitting | semshift | Chunks too small to retrieve on |
