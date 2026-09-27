"""Typed filters over a store's declared fields, compiled to SQL.

Every value is checked against its field's type and bound as a
parameter. A filter that cannot be compiled raises; it never widens to an
unfiltered search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union


class FilterError(ValueError):
    """A filter that cannot be compiled: unknown field, wrong type, empty set."""


@dataclass(frozen=True)
class Cmp:
    op: str
    field: str
    value: Any


@dataclass(frozen=True)
class In:
    field: str
    values: tuple


@dataclass(frozen=True)
class Between:
    field: str
    low: Any
    high: Any


@dataclass(frozen=True)
class IsNull:
    field: str


@dataclass(frozen=True)
class All:
    parts: tuple


@dataclass(frozen=True)
class Any_:
    parts: tuple


@dataclass(frozen=True)
class Not:
    inner: Any


@dataclass(frozen=True)
class IdSet:
    """A trusted subquery selecting ids from the consumer's own tables.

    Applied as `id IN (sql)`. For code, never for end-user input.
    """

    sql: str
    params: tuple = ()


Filter = Union[Cmp, In, Between, IsNull, All, Any_, Not, IdSet]

_SQL_OPS = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
            "glob": "GLOB"}
#: Filters whose matching set is small and repeated enough to cache an
#: index for. Ranges and id subqueries tend to be one-off.
_UNCACHEABLE = {"gt", "gte", "lt", "lte"}
_MIN_SQLITE_INT = -(1 << 63)
_MAX_SQLITE_INT = (1 << 63) - 1


def eq(field: str, value: Any) -> Filter:
    return Cmp("eq", field, value)


def ne(field: str, value: Any) -> Filter:
    """Rows where `field` differs from `value`, including rows where it is null."""
    return Not(Cmp("eq", field, value))


def in_(field: str, values: Sequence[Any]) -> Filter:
    return In(field, tuple(values))


def gt(field: str, value: Any) -> Filter:
    return Cmp("gt", field, value)


def gte(field: str, value: Any) -> Filter:
    return Cmp("gte", field, value)


def lt(field: str, value: Any) -> Filter:
    return Cmp("lt", field, value)


def lte(field: str, value: Any) -> Filter:
    return Cmp("lte", field, value)


def between(field: str, low: Any, high: Any) -> Filter:
    """Inclusive at both ends."""
    return Between(field, low, high)


def glob(field: str, pattern: str) -> Filter:
    """SQLite GLOB: case-sensitive, `*` and `?` wildcards."""
    return Cmp("glob", field, pattern)


def is_null(field: str) -> Filter:
    return IsNull(field)


def and_(*parts: Filter) -> Filter:
    return All(tuple(parts))


def or_(*parts: Filter) -> Filter:
    return Any_(tuple(parts))


def not_(inner: Filter) -> Filter:
    """Rows the inner filter does not match, including rows where it is null."""
    return Not(inner)


@dataclass(frozen=True)
class Compiled:
    sql: str
    params: tuple
    cacheable: bool


def _value(kind: str, field: str, value: Any) -> Any:
    """`value` as a valid bound parameter for a column of `kind`."""
    if kind == "bool":
        if type(value) is not bool:
            raise TypeError(f"{field} is bool; got {value!r}")
        return int(value)
    if kind == "int":
        if type(value) is not int:
            raise TypeError(f"{field} is int; got {value!r}")
        if not _MIN_SQLITE_INT <= value <= _MAX_SQLITE_INT:
            raise ValueError(f"{field} is outside SQLite's integer range")
        return value
    if kind == "float":
        if type(value) not in (int, float):
            raise TypeError(f"{field} is float; got {value!r}")
        try:
            converted = float(value)
        except OverflowError as exc:
            raise TypeError(f"{field} is float; value is out of range") from exc
        if not math.isfinite(converted):
            raise TypeError(f"{field} is float; got {value!r}")
        return converted
    if type(value) is not str:
        raise TypeError(f"{field} is text; got {value!r}")
    return value


def compile_filter(flt: Filter | None, kinds: Mapping[str, str],
                   alias: str = "i") -> Compiled:
    """SQL over `alias` for `flt`; `kinds` maps declared field to type."""
    if flt is None:
        return Compiled("1", (), True)
    params: list = []
    cacheable = True

    def col(field: str) -> str:
        if field not in kinds:
            raise ValueError(f"unknown field {field!r}; declared: {sorted(kinds)}")
        return f'{alias}."{field}"'

    def walk(node) -> str:
        nonlocal cacheable
        if isinstance(node, Cmp):
            if node.op not in _SQL_OPS:
                raise ValueError(f"unknown comparison {node.op!r}")
            if node.op == "glob" and kinds.get(node.field) != "text":
                raise TypeError(f"glob needs a text field; {node.field!r} is not")
            if node.op in _UNCACHEABLE:
                cacheable = False
            c = col(node.field)
            params.append(_value(kinds[node.field], node.field, node.value))
            return f"{c} {_SQL_OPS[node.op]} ?"
        if isinstance(node, In):
            c = col(node.field)
            if not node.values:
                raise ValueError(f"in_({node.field!r}) needs at least one value")
            params.extend(_value(kinds[node.field], node.field, v) for v in node.values)
            return f"{c} IN ({', '.join('?' * len(node.values))})"
        if isinstance(node, Between):
            c = col(node.field)
            cacheable = False
            params.append(_value(kinds[node.field], node.field, node.low))
            params.append(_value(kinds[node.field], node.field, node.high))
            return f"{c} BETWEEN ? AND ?"
        if isinstance(node, IsNull):
            return f"{col(node.field)} IS NULL"
        if isinstance(node, (All, Any_)):
            if not node.parts:
                raise ValueError("and_/or_ need at least one filter")
            joiner = " AND " if isinstance(node, All) else " OR "
            return "(" + joiner.join(walk(p) for p in node.parts) + ")"
        if isinstance(node, Not):
            # NOT over NULL is NULL, which would drop rows missing the
            # field; coalescing first counts them as not matching.
            return f"NOT coalesce(({walk(node.inner)}), 0)"
        if isinstance(node, IdSet):
            cacheable = False
            params.extend(node.params)
            return f"{alias}.id IN ({node.sql})"
        raise TypeError(f"not a filter: {node!r}")

    try:
        sql = walk(flt)
    except FilterError:
        raise
    except (TypeError, ValueError) as exc:
        raise FilterError(str(exc)) from exc
    return Compiled(sql, tuple(params), cacheable)
