# semsift — design

| Block | Status |
|---|---|
| `embed` | Implemented |
| `fuse` | Implemented |
| `store` | Implemented |
| `rerank` | Implemented |
| `search` | Implemented |
| `evals` | Implemented |
| `chunk` | Implemented |

## What it is

A library of building blocks for hybrid retrieval: find the most relevant
items in a corpus by combining keyword search and vector search, then
rank them. Each block is a small interface with shipped implementations;
a consumer assembles the ones it needs and adds its own.

Two consumers shape it:

- **repoglass**: code search over one SQLite index per repository.
- **The knowledge and memory store in toolshed**: retrieval over captured
  documents and agent memories, in its own SQLite file. An agent using it
  may generate answers from the hits; the store and semsift do not.

Both run the same flow:

```
write:  content → chunk → embed (outside the transaction) → store
query:  query → store vector search  ┐
                store keyword search ┼→ fuse → hydrate top N → rerank → hits
                consumer sources ────┘
```

## Non-goals

- Generating text. A model that returns a number (an encoder, a
  cross-encoder) is in scope; one that returns text is not.
- Content-specific chunking: one chunk per definition linked to a symbol
  table, one per mail message. Consumers build those on `chunk`'s spans.
- Owning files, connections or transactions. The consumer opens its SQLite
  database, sets WAL and busy timeouts, hands semsift the connection and
  commits.
- Mapping identities. semsift stores integer ids; each consumer keeps the
  mapping from its durable keys (a file path and span, a provider message
  id) to those integers.
- A server, CLI or MCP surface.

## Blocks

| Block | Interface | Shipped | Left to consumers |
|---|---|---|---|
| `embed` | `Encoder` | static (model2vec), ONNX, HTTP; model policy; `VectorSpace` | task instructions in query prefixes |
| `store` | `Store` | one SQLite store, on disk or in memory | the id mapping; their own tables |
| `fuse` | `RankedList` in, `Fused` out | RRF, linear blend, first list | weights by query shape |
| `rerank` | `Reranker` | recency, MMR, metadata rules | domain rerankers (code boosts, file coherence) |
| `search` | composer and `Source` | one composer | extra sources, including candidate expansion |
| `evals` | harness | metrics, run files, fusion tuning, golden vectors, a small labelled corpus | their own labelled data |
| `chunk` | `chunk(text)` → `Chunk`s | `TextChunker`, `TreeSitterChunker` | content-specific chunking |

`embed` imports nothing else from semsift, so it can become its own
package if something needs embeddings without retrieval.

### embed

An `Encoder` has `encode(texts)`, `encode_query(texts)` and `space`.
Anything with those three works; the shipped backends are conveniences.
The store calls an encoder from several threads at once (see Canaries),
so an encoder must be safe to call concurrently; the shipped ones are.

- `encode` adds the document prefix and `encode_query` the query prefix.
  Constructors take `query_prefix` and `doc_prefix` as `str | None`:
  `None` takes the model family's default, `""` means none.
- The default table holds only markers a model needs whatever is searched
  (e5, nomic, bge). An instruction that names a task (Qwen3-Embedding,
  CodeRankEmbed) has no default; the consumer passes it.
- `VectorSpace(model, backend, variant, dims, doc_prefix, pooling)` names
  what decides whether two stored vectors are comparable. `variant` is the
  ONNX graph file or the HTTP endpoint. `VectorSpace.of(...)` computes it
  without loading a model, except `dims`.
- An embedding cache, when used, keys document vectors on
  `(space, "document", text)` and query vectors on
  `(space, "query", query_prefix, text)`. The query prefix is outside the
  space because it never touches stored vectors, but it changes query
  vectors.

### store

`Store(conn, prefix, fields, encoder=None, tokenizer="unicode61")`.
Without an encoder the store is keyword-only and vector search raises.

- `prefix` names the store's tables. Several stores can share one
  database. It must match `[a-z][a-z0-9_]*`; anything else is refused
  before it reaches SQL.
- `fields` declares the metadata the store can filter and hydrate: a name
  and a type (`int`, `float`, `text`, `bool`). Each becomes a real, typed
  column; declaring one `indexed` adds an index. Undeclared metadata goes
  in a JSON `extra` column that is returned on hydration and never
  filtered.
- `tokenizer` is one of `unicode61`, `porter`, `trigram`.

**Tables.** Per store: `items` (id, text, keyword text, declared fields,
`extra`), `vectors` (id, little-endian float16 blob), an FTS5 table with
its own content, and a `meta` row holding the `VectorSpace`, the declared
fields and a generation counter. They are created with individual
statements, never `executescript`, which would commit the caller's
transaction.

**Items.** An item has an integer `id` chosen by the consumer, `text`
(what is embedded and returned), an optional `keyword_text` (what the
keyword index reads; defaults to `text`), and metadata. repoglass puts
path words in `keyword_text` without them reaching its embeddings; the
knowledge store puts title, people and notes there.

**Writes.**

| Operation | Meaning |
|---|---|
| `embed(items)` | encodes items with the store's encoder; touches no table |
| `upsert(items)` | writes items with their vectors; replaces rows with the same id |
| `remove(ids)` | deletes items, vectors and keyword rows |

Encoding is slow, so it happens in `embed`, before the consumer opens its
write transaction. `upsert` accepts only vectors produced by an encoder
whose space matches the stored one, or precomputed `Vectors(space, rows)`
checked the same way. The store never commits: `upsert` and `remove`
require the caller to be inside a transaction, so an item, its vector,
its keyword row and the consumer's own rows commit or roll back together.
The FTS5 table is updated by explicit statements in the same transaction,
so a rollback leaves it consistent. Every write bumps the generation
counter.

**Vector space.** An empty store adopts the encoder's space on first
write. A populated store whose space differs from the encoder's refuses
every read and write, naming both spaces.

**Canaries.** The space is the encoder's identity; it does not show a
model whose weights changed under the same name, or an HTTP server that
swapped its model. So the first write also records the vectors of a few
fixed canary texts, each encoded alone with `encode`, and when the
vectors were made. Every vector search re-encodes the canaries,
concurrently with the query, and takes the lowest cosine with the stored
ones; `health()` does the same on demand. `encode` and `encode_query` run
the same model, so one side shows any change to it. Cost: each vector
search makes three extra encoder calls, running alongside the query's, so
little wall time but more CPU, and three more requests for an HTTP
encoder.

| Similarity | State | Vector search |
|---|---|---|
| at least `drift_warn` (0.9999) | `ok` | normal |
| at least `drift_stale` (0.99) | `drifted` | normal, with a warning to re-embed |
| below `drift_stale` | `stale` | raises `StaleVectors`; the composer skips it with a warning, leaving keyword results |

Vectors written without canaries (precomputed `Vectors`) are `unchecked`.

**Re-embedding.** The store keeps each item's `text`, so it can re-embed
without the consumer. `prepare_reembed()` encodes every stored text with
the store's encoder and touches no table; `apply_reembed(plan)`, inside
the caller's transaction, replaces every vector and records the new
space, canaries and time. It refuses a plan prepared before another
write. It also moves a store to a new vector space.

**Searches** return a `RankedList`, which may carry warnings (a drifted
encoder).

| Operation | Meaning |
|---|---|
| `search_vector(query, k, filter)` | encodes `query` with `encode_query`, ranks by cosine |
| `search_keyword(query, k, filter)` | FTS5 BM25, negated so higher is better; `raw` keeps the original |
| `fetch(ids, fields)` | text, metadata and vectors for hydration |

**Vector search** delegates the nearest-neighbour math to MinishLab
vicinity's basic backend. The adapter around it owns:

- loading the matching vectors, widening them to float32 and building the
  index;
- converting vicinity's cosine distance to a higher-is-better score;
- deterministic ties: vicinity picks among equal scores arbitrarily, so
  the adapter fetches past `k` and keeps widening while the score at the
  cut equals the score after it, up to all rows, then orders by
  `(-score, id)` and cuts to `k`;
- refusing a query vector whose width differs from the stored one.

**The vector cache** holds built indexes, keyed by filter, bounded by
entry count and bytes and evicted least recently used. A filter with a
range or an id subquery is not cached. An entry is valid for one value of
the store's generation counter, which every search reads: a write through
any connection bumps it, and writes to other tables leave it alone. While
the connection is inside a transaction nothing is cached, because a
rollback would leave the cache holding rows that no longer exist.

**Filters** are typed conditions over declared fields: `eq`, `ne`, `in`,
`gt`, `gte`, `lt`, `lte`, `between`, `glob`, `is_null`, combined with
`and`, `or`, `not`. Values are checked against the field's type and bound
as parameters. `ne` and `not` treat a missing value as not matching the
inner condition, so `ne(field, x)` includes rows where the field is null.
A filter compiles to a `WHERE` clause applied before both searches.

A filter may carry an `IdSet(sql, params)`: a trusted subquery selecting
ids from the consumer's own tables, applied as `id IN (subquery)`. It is
for code, never for end-user input. repoglass uses it to filter by
language and path in its `file` table.

A filter that cannot be compiled or applied raises. It never degrades to
an unfiltered search.

### fuse

A `RankedList` is a source name and `Scored(id, score, raw)` items:

- items are ordered best first, and each id appears once;
- `score` is finite and higher-is-better, whatever the source;
- `raw` is the source's own value, kept for display (`bm25=-15.6`), never
  read by a fuser.

A source that produces ascending-best scores, such as SQLite `bm25()`,
negates them when building its list.

| Fuser | Uses | Parameters |
|---|---|---|
| RRF | ranks only | `k` (default 60), per-source weights |
| linear blend | scores | per-source weights; normalisation `min-max`, `max`, `sum`, `z-score` |
| first list | the first non-empty source | — |

A fuser returns `Fused`: ids and fused scores, best first with ties by
id, and evidence. Evidence for an id lists, for every input list that
contains it, the source, its rank there, its score and its raw value.

### rerank

A `Reranker` takes the query and candidates and returns candidates. A
candidate is `(id, score, text, metadata, vector)`. Each reranker
declares which fields it needs, and the composer fetches only those.

A reranker may re-score, reorder and drop; the order it returns is the
ranking. Recency and metadata rules re-score and sort by score; MMR
reorders without re-scoring, so it runs last. It may not add ids; a consumer
that needs to add candidates does it with a `Source`. Output ids are
unique. When reranking leaves fewer than `k`, the result has fewer than
`k` and a warning says so.

| Reranker | Needs | Behaviour |
|---|---|---|
| recency | a declared timestamp field (UTC epoch seconds) | multiplies score by `0.5 ** (age / half_life)`; the clock is injectable; a missing timestamp takes a configured factor; a future timestamp counts as age 0 |
| MMR | vectors | trades relevance against similarity to hits already chosen; relevance is min-max normalised before mixing |
| metadata rules | declared fields | multiplies score by each matching rule's factor; matches compound; factors are finite and positive |

Recency ranks; it does not decide what is current. A consumer with
superseded records (the knowledge store's memories) filters inactive
revisions out before retrieval.

### search

A `Source` takes the query, a depth and the filter and returns a
`RankedList`. The
store's vector and keyword searches are sources; a consumer adds its own
(repoglass's exact-symbol match, or candidate expansion that adds items
no other source found).

The composer is given sources, a fuser, rerankers, `k`, a candidate depth
(how many results each source returns and how many fused ids are
hydrated; four times `k` by default) and a **base filter**. The base
filter is ANDed with every request's filter, passed to every source, and
applied again when candidates are hydrated, so a consumer source that
ignores it cannot widen the result; the knowledge store puts its account
scope there. A filter or database error raises; any other source failure
skips that source with a warning, and raises if every source failed. The composer runs the
sources, fuses, fetches the fields the rerankers need for the top
candidates in one read, runs the rerankers in order and cuts to `k`.

A result carries:

- hits: `id`, `score`, `text`, and the declared fields the consumer named
  as its citation fields (a durable key and a span);
- evidence per hit;
- `warnings`: facts about a degraded result, such as a source that failed
  and was skipped, a drifted encoder, or fewer than `k` hits after
  reranking. A filter or
  scope failure is never a warning; it raises.

### chunk

A `Chunk` is `text`, its character span (`start`, `end`), 1-based
`start_line` and `end_line`, and, when the chunker knows them, `context`
(enclosing definitions) and `symbols` (names defined inside). The span is
what a hit cites. Both chunkers bound each chunk by `max_chars` and drop
chunks whose stripped text is shorter than `min_chars`; with `min_chars`
at 0, the spans tile the text in order.

- `TextChunker` groups lines. With `markdown` on (the default), an ATX
  heading outside a fenced block always starts a chunk; after a blank
  line, a chunk at least half full ends before the next paragraph; a line
  longer than the bound is cut.
- `TreeSitterChunker` delegates to tree-sitter-language-pack's chunker,
  which splits along the syntax tree and reports each chunk's context and
  symbols. It needs the `tree-sitter` extra. A language the pack does not
  list, a grammar it cannot download, a source over `max_source_bytes`
  (5 MB), a parse past `parse_timeout_ms` (5 s) or a parse that fails
  falls back to a `TextChunker` with `markdown` off, since `#` often
  begins a comment in code. `supports()` says whether the pack lists a
  language; the grammar may still need downloading on first use.

### evals

- Metrics: Hit@K, MRR and NDCG over labelled queries. Labels name items by
  the consumer's durable keys, never by store ids, which can be reissued.
- Run files: each source's ranked list per query, with the corpus
  generation and `VectorSpace` they came from. Fusion settings can be
  swept from run files alone; reranker settings cannot, because rerankers
  read vectors and metadata the run file does not hold.
- Fusion tuning: weights, RRF `k` and normalisation are chosen on one part
  of the queries and reported on the rest.
- Baselines and slices: keyword-only and vector-only runs beside the fused
  one; queries that should return nothing.
- Golden vectors: fixed texts with expected vectors per supported model,
  compared within a tolerance, which fail when a prefix, pooling or model
  change moves them. They skip when the model is not available locally.
- A small labelled corpus written for semsift, to judge changes to fusion
  and reranking.

Consumers keep their own data: repoglass its code corpus and navigation
benchmark in `benchmarks/`, the knowledge store a personal labelled set
outside any repository.

## Packaging

- Published to PyPI only. semsift has no command of its own, so it has no
  Homebrew formula; a consumer's formula lists it as a resource.
- Core dependencies: numpy, vicinity, and model2vec with huggingface-hub
  and tokenizers, so a plain install can embed. `sqlite3` is in the
  standard library; FTS5 must be compiled in.
- Extras: `onnx` (onnxruntime), `webgpu` (onnxruntime and its webgpu
  plugin), `tree-sitter` (tree-sitter-language-pack). Each class that
  needs one names it in the ImportError it raises when it is missing.
- Python 3.11 or newer.

## Migration from repoglass

`repoglass.semantic_core` moved here as `embed` and `fuse`. repoglass
adopts `embed`, `fuse`, `store` and `chunk` first; `TreeSitterChunker`
replaces its window chunker, and it keeps its definition chunks and
symbol table. It keeps its own orchestration too:
its definition boost adds candidates retrieval never returned, and its
file coherence reads the whole fused pool, neither of which fits a
reranker. It moves onto the composer when those become `Source`s.

## Open

- **Language detection.** Callers name the language; the pack's
  detection by path or content is not exposed.
- **Canary thresholds.** 0.9999 and 0.99 have not been checked against a
  real model revision change.
- **Unknown ONNX models** pool with `cls` unless the caller says
  otherwise.
- **BM25 parameters.** FTS5's `bm25()` takes column weights, not `k1` and
  `b`, so BM25 itself cannot be tuned.
- **Batch-dependent static vectors.** The same text can get slightly
  different vectors depending on which texts share its batch.
- **Public corpus.** Whether semsift also ships a public labelled set
  (a BEIR subset) depends on size and licence, not yet checked.

## Rejected alternatives

**Our own nearest-neighbour math.** vicinity provides it, with
approximate backends for when a corpus outgrows exhaustive scoring.

**A SQLite backend inside vicinity.** A vicinity backend holds vectors in
memory; SQLite would only be where they load from.

**A second, in-memory store implementation.** SQLite runs in memory with
the same tables, FTS5 and filters.

**Filtering on JSON metadata.** `json_extract` comparisons follow SQLite's
loose typing and cannot be indexed per field; declared typed columns can.

**External-content or contentless FTS5.** External content goes stale
unless triggers or rebuilds keep it in step; contentless-delete needs a
recent SQLite. Contentful FTS updated in the same transaction stays
consistent through rollback.

**Encoding inside `upsert`.** A model call inside the write transaction
holds the database lock for as long as the model takes.

**A warning for a filter that could not be applied.** An unfiltered
result can cross an account boundary.

**Candidate expansion as a reranker.** Adding items is retrieval; a
reranker that adds ids hides a source.

**A separate retriever component.** Search depends on how vectors are
stored, so it lives on the store; fusion needs only a common output shape.

**Carrying each score's meaning on the item.** One rule at the source
spares every fuser from branching on it.

**Chroma, LanceDB, pgvector, a file-backed store like semble's.** No
consumer needs a server or millions of vectors.

**A separate embeddings package now.** Both consumers use encoders with
search, and the store checks the encoder's space.

**Generating inside semsift (HyDE, answer synthesis).** A consumer that
wants a hypothetical document generates it and passes it as the query.

**A custom BM25 index.** It duplicates FTS5.

**Derived confidence fields.** A rescaled score read as a probability
misleads; evidence says which sources matched.

**Fallbacks that fabricate results.** Stub vectors, fixed scores and mock
hits mix with real results and rank wrongly without an error.

**Fusing by content hash.** Identical text from two items is two items.

**Pickle for persisted state.** Loading it can run code.
