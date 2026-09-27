"""Score keyword, vector and fused retrieval on semsift's labelled corpus.

    .venv/bin/python benchmarks/run.py
    .venv/bin/python benchmarks/run.py --model minishlab/potion-code-16M-v2 --runs out.jsonl

Needs the model locally or a network connection to fetch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from semsift.embed import StaticEncoder
from semsift.evals import collect, evaluate, source_run, split, tune, write_runs
from semsift.fuse import blend, rrf
from semsift.search import KeywordSource, VectorSource
from semsift.store import Field, Item, Store

CORPUS = Path(__file__).with_name("corpus.json")
K = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="minishlab/potion-base-8M")
    parser.add_argument("--runs", type=Path, help="also write the run set here")
    args = parser.parse_args()

    corpus = json.loads(CORPUS.read_text())
    keys = sorted(corpus["documents"])
    items = [Item(i, corpus["documents"][key], {"key": key}) for i, key in enumerate(keys)]
    conn = sqlite3.connect(":memory:", isolation_level=None)
    store = Store(conn, "bench", [Field("key", "text")], encoder=StaticEncoder(args.model),
                  tokenizer="porter")
    vectors = store.embed(items)
    conn.execute("BEGIN")
    store.upsert(items, vectors)
    conn.execute("COMMIT")

    queries = {q: spec["text"] for q, spec in corpus["queries"].items()}
    qrels = {q: spec["relevant"] for q, spec in corpus["queries"].items()}
    space = store.space
    assert space is not None
    runs = collect([KeywordSource(store), VectorSource(store)], queries,
                   key_of=keys.__getitem__, k=K,
                   meta={"model": args.model,
                         "generation": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                         "space": asdict(space)})
    if args.runs:
        write_runs(args.runs, runs)

    rows = {"keyword": source_run(runs, "keyword"), "vector": source_run(runs, "vector"),
            "rrf": runs.fuse(rrf), "blend min-max": runs.fuse(blend),
            "blend z-score": runs.fuse(lambda ls: blend(ls, normalise="z-score"))}
    print(f"{'run':16} hit@{K}  mrr    ndcg@{K}  no-answer returned")
    for name, run in rows.items():
        m = evaluate(run, qrels, k=K)
        print(f"{name:16} {m[f'hit@{K}']:.3f}  {m['mrr']:.3f}  "
              f"{m[f'ndcg@{K}']:.3f}  {m['no_answer_returned']:.1f}")

    train, test = split(list(qrels), test_share=0.3)
    options = {f"rrf vector={w}": (lambda w: lambda ls: rrf(ls, weights={"vector": w}))(w)
               for w in (0.25, 0.5, 1.0, 2.0, 4.0)}
    tuned = tune(runs, qrels, options, train=train, test=test, metric=f"ndcg@{K}", k=K)
    train_answered = sum(any(grade > 0 for grade in qrels[q].values()) for q in train)
    test_answered = sum(any(grade > 0 for grade in qrels[q].values()) for q in test)
    print(f"\ntuned on {train_answered} answered queries, reported on "
          f"{test_answered}: {tuned.best}"
          f" train {tuned.train[tuned.best]:.3f} test {tuned.test[tuned.best]:.3f}")


if __name__ == "__main__":
    main()
