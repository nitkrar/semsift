"""Model-family defaults for prefixes and pooling.

Only what a model needs whatever is being searched. An instruction that
names a task (Qwen3-Embedding, CodeRankEmbed) has no default here; the
caller supplies it, through `extra` or a prefix argument.
"""

from __future__ import annotations

from typing import Literal, Mapping

#: Models that were trained with an instruction on the query side only.
#: Encoding a query the same way as a document costs recall silently:
#: nothing errors, results are just worse.
_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
    "nomic-embed": "search_query: ",
}

#: Prefixes for the DOCUMENT side. Most models want nothing here, but
#: e5 and nomic were trained with a marker on both sides and omitting
#: the document half is not neutral -- it puts queries and documents in
#: different regions of the space. Nothing errors; recall just drops.
_DOC_PREFIXES = {
    "e5": "passage: ",
    "nomic-embed": "search_document: ",
}

#: How a model turns token vectors into one sentence vector. Getting this
#: wrong produces plausible-looking vectors that rank badly, and nothing
#: errors.
_POOLING = {
    "bge": "cls",
    "e5": "mean",
    "gte": "mean",
    "qwen3-embedding": "last",     # last non-pad token, not CLS
    "coderankembed": "mean",       # community ONNX card; base model is CLS
    "all-minilm": "mean",
    "nomic-embed": "mean",
}


def _family(model: str, extra: Mapping[str, str] | None = None) -> str:
    stem = model.rsplit("/", 1)[-1].lower()
    known = (set(_POOLING) | set(_QUERY_PREFIXES) | set(_DOC_PREFIXES)
             | set(extra or ()))
    for family in sorted(known, key=len, reverse=True):
        if family in stem:
            return family
    return ""


def default_query_prefix(model: str,
                         extra: Mapping[str, str] | None = None) -> str:
    """The prefix a model family expects on queries, or '' if none.

    `extra` maps family names to the caller's own prefixes and wins over
    this module's table.
    """
    table = {**_QUERY_PREFIXES, **(extra or {})}
    return table.get(_family(model, extra), "")


def default_doc_prefix(model: str) -> str:
    """The prefix a model family expects on documents, or '' if none."""
    return _DOC_PREFIXES.get(_family(model), "")


def default_pooling(model: str) -> str:
    """The pooling a model was trained with, or 'cls' when unknown."""
    return _POOLING.get(_family(model), "cls")


def resolve_prefix(override: str | None, model: str,
                   side: Literal["query", "doc"],
                   extra: Mapping[str, str] | None = None) -> str:
    """An explicit prefix wins over the recommendation; `''` is a choice.

    `None` means "whatever the model wants"; `""` means "none". Collapsing
    the two would make the recommendation impossible to turn off. `extra`
    is as for `default_query_prefix`, and applies to the query side.
    """
    if override is not None:
        return override
    if side == "query":
        return default_query_prefix(model, extra)
    return default_doc_prefix(model)
