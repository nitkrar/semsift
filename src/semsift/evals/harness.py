"""Metrics, run files, query splits and fusion tuning.

Everything here names items by the consumer's durable keys, never by
store ids, which can be reissued.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..fuse import Fused, RankedList, Scored

Run = Mapping[str, Sequence[str]]
Qrels = Mapping[str, Mapping[str, int]]


def _validate_ranking(ranked: Sequence[str]) -> None:
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked keys must be unique")


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")


def hit_at(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    _validate_ranking(ranked)
    _validate_k(k)
    return 1.0 if any(relevant.get(key, 0) > 0 for key in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: Mapping[str, int]) -> float:
    _validate_ranking(ranked)
    for rank, key in enumerate(ranked, start=1):
        if relevant.get(key, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalised DCG with gain equal to the grade."""
    _validate_ranking(ranked)
    _validate_k(k)
    dcg = sum(max(relevant.get(key, 0), 0) / math.log2(rank + 1)
              for rank, key in enumerate(ranked[:k], start=1))
    ideal = sorted((grade for grade in relevant.values() if grade > 0), reverse=True)[:k]
    best = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / best if best else 0.0


def evaluate(run: Run, qrels: Qrels, *, k: int = 10) -> dict[str, float]:
    """Mean Hit@k, MRR and NDCG@k over queries that have an answer.

    Queries labelled with no relevant item are counted apart:
    `no_answer_returned` is how many items the run returned for them on
    average, which a good system keeps low when it can abstain.
    """
    _validate_k(k)
    for query in qrels:
        _validate_ranking(run.get(query, ()))
    answered = [q for q, rel in qrels.items() if any(grade > 0 for grade in rel.values())]
    unanswered = [q for q, rel in qrels.items() if not any(
        grade > 0 for grade in rel.values())]
    out = {f"hit@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0,
           "queries": len(answered), "no_answer_queries": len(unanswered),
           "no_answer_returned": 0.0}
    for q in answered:
        ranked = list(run.get(q, ()))
        out[f"hit@{k}"] += hit_at(ranked, qrels[q], k)
        out["mrr"] += mrr(ranked, qrels[q])
        out[f"ndcg@{k}"] += ndcg_at(ranked, qrels[q], k)
    for key in (f"hit@{k}", "mrr", f"ndcg@{k}"):
        out[key] = out[key] / len(answered) if answered else 0.0
    if unanswered:
        out["no_answer_returned"] = sum(len(run.get(q, ())) for q in unanswered) / len(unanswered)
    return out


@dataclass(frozen=True)
class RunSet:
    """Each source's ranked list per query, and what produced them.

    `lists[query][source]` is `[(key, score, raw), ...]`, best first.
    `meta` records the corpus generation and vector space, so a run set is
    not compared against a different corpus by mistake. Fusion can be
    replayed from a run set; reranking cannot, since rerankers read
    vectors and metadata a run set does not hold.
    """

    lists: Mapping[str, Mapping[str, Sequence[tuple[str, float, float]]]]
    meta: Mapping[str, object] = field(default_factory=dict)

    def fuse(self, fuser: Callable[[Sequence[RankedList]], Fused]) -> dict[str, list[str]]:
        out = {}
        for query, by_source in self.lists.items():
            keys = sorted({key for items in by_source.values() for key, _, _ in items})
            # Keys become ids in sorted order, so a fuser's id tie-break is
            # a key tie-break.
            ids = {key: i for i, key in enumerate(keys)}
            lists = []
            for source, items in by_source.items():
                if any(a[1] < b[1] for a, b in zip(items, items[1:])):
                    raise ValueError(f"{source}: items are not best first")
                ordered = sorted(items, key=lambda item: (-item[1], item[0]))
                lists.append(RankedList(source, tuple(
                    Scored(ids[key], score, raw) for key, score, raw in ordered)))
            out[query] = [keys[i] for i, _ in fuser(lists).items]
        return out


def collect(sources: Sequence, queries: Mapping[str, str], *,
            key_of: Callable[[int], str], k: int = 50, filter=None,
            meta: Mapping[str, object] | None = None) -> RunSet:
    """Run each source for each query and record its list by durable key.

    `key_of` maps a store id to the consumer's durable key. `meta` should
    name the corpus generation and vector space the lists came from.
    """
    lists = {}
    for qid, text in queries.items():
        lists[qid] = {}
        for source in sources:
            ranked = source.search(text, k, filter)
            if ranked.source in lists[qid]:
                raise ValueError("source names must be unique")
            items = [(key_of(s.id), s.score, s.raw) for s in ranked.items]
            keys = [key for key, _, _ in items]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{ranked.source}: a durable key appears twice")
            lists[qid][ranked.source] = sorted(
                items, key=lambda item: (-item[1], item[0]))
    return RunSet(lists, dict(meta or {}))


def source_run(runs: RunSet, source: str) -> dict[str, list[str]]:
    """One source's ranking per query: the baseline for that source."""
    return {q: [key for key, _, _ in by_source.get(source, ())]
            for q, by_source in runs.lists.items()}


def write_runs(path: Path, runs: RunSet) -> None:
    """JSON lines: a meta line, then one line per query."""
    _validate_meta(runs.meta)
    lines = [json.dumps({"meta": dict(runs.meta)}, allow_nan=False)]
    for query, by_source in runs.lists.items():
        lines.append(json.dumps({"query": query, "lists": {
            s: [list(t) for t in items] for s, items in by_source.items()}},
            allow_nan=False))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_runs(path: Path) -> RunSet:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("run file is empty")
    meta = _load_json(lines[0])["meta"]
    _validate_meta(meta)
    lists = {}
    for line in lines[1:]:
        row = _load_json(line)
        if row["query"] in lists:
            raise ValueError(f"query {row['query']!r} appears twice")
        lists[row["query"]] = {s: [tuple(t) for t in items] for s, items in row["lists"].items()}
    return RunSet(lists, meta)


def _validate_meta(meta: Mapping[str, object]) -> None:
    if "generation" not in meta or "space" not in meta:
        raise ValueError("run metadata must include generation and space")


def _load_json(line: str):
    def reject(value: str):
        raise ValueError(f"non-standard JSON number {value}")

    return json.loads(line, parse_constant=reject)


def split(query_ids: Sequence[str], *, test_share: float = 0.3,
          seed: str = "semsift") -> tuple[list[str], list[str]]:
    """Train and test query ids, decided per id by a hash of `seed` and the id.

    The same ids and seed give the same split in any order.
    """
    if not 0.0 < test_share < 1.0:
        raise ValueError("test_share must be between 0 and 1")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query ids must be unique")
    train, test = [], []
    for q in sorted(query_ids):
        digest = hashlib.blake2b(f"{seed}:{q}".encode(), digest_size=8).digest()
        (test if int.from_bytes(digest, "big") / 2**64 < test_share else train).append(q)
    return train, test


@dataclass(frozen=True)
class Tuned:
    best: str
    #: metric per option on the train queries, and on the test queries.
    train: Mapping[str, float]
    test: Mapping[str, float]


def tune(runs: RunSet, qrels: Qrels,
         options: Mapping[str, Callable[[Sequence[RankedList]], Fused]], *,
         train: Sequence[str], test: Sequence[str], metric: str = "mrr",
         k: int = 10) -> Tuned:
    """Choose the fuser with the best `metric` on `train`; report both splits.

    Ties go to the option listed first. The test figures are the honest
    estimate; the train figures only rank the options.
    """
    if set(train) & set(test):
        raise ValueError("train and test queries overlap")
    if not train or not test:
        raise ValueError("train and test must be non-empty")
    _validate_k(k)
    quality_metrics = {"mrr", f"hit@{k}", f"ndcg@{k}"}
    if metric not in quality_metrics:
        raise ValueError(f"metric must be one of {sorted(quality_metrics)}")
    train_scores, test_scores = {}, {}
    for name, fuser in options.items():
        fused = runs.fuse(fuser)
        train_scores[name] = evaluate(fused, {q: qrels[q] for q in train}, k=k)[metric]
        test_scores[name] = evaluate(fused, {q: qrels[q] for q in test}, k=k)[metric]
    best = max(options, key=lambda name: train_scores[name])
    return Tuned(best, train_scores, test_scores)
