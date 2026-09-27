"""What decides whether two stored vectors are comparable."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import default_pooling, resolve_prefix


@dataclass(frozen=True)
class VectorSpace:
    """Vectors from two different spaces rank wrongly and never error.

    `variant` is what a backend serves the model from, when that can
    change under one model name: the ONNX graph file, or the HTTP
    endpoint. The query prefix is not a field: it changes query vectors,
    not stored ones. `pooling` is '' for backends that do not pool.
    """

    model: str
    backend: str
    variant: str
    dims: int
    doc_prefix: str
    pooling: str

    @classmethod
    def of(cls, model: str, backend: str, *, variant: str = "",
           doc_prefix: str | None = None, pooling: str | None = None,
           dims: int = 0) -> "VectorSpace":
        """The space an encoder built from these arguments writes into.

        Resolves defaults the way the encoder constructors do, without
        loading a model. `dims` is the one field that needs the model.
        """
        return cls(
            model=model,
            backend=backend,
            variant=variant.rstrip("/") if backend == "http" else variant,
            dims=dims,
            doc_prefix=resolve_prefix(doc_prefix, model, "doc"),
            pooling=(pooling or default_pooling(model)) if backend == "onnx" else "",
        )
