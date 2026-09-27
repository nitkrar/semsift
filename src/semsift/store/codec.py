"""How a vector is stored: unit length, little-endian float16, one row each."""

from __future__ import annotations

from typing import Sequence

#: Two-byte encoding of one stored vector component. Changing it makes
#: every stored vector unreadable.
VECTOR_DTYPE = "<f2"


def pack(vector: Sequence[float], dtype: str = VECTOR_DTYPE) -> bytes:
    """Normalise to unit length, so scoring is a bare dot product."""
    import numpy as np

    v = np.asarray(vector, dtype="float32")
    if v.ndim != 1:
        raise ValueError(f"a vector must be one-dimensional, got shape {v.shape}")
    if not v.size:
        raise ValueError("a vector must have at least one component")
    if not np.isfinite(v).all():
        raise ValueError("a vector must contain only finite components")
    norm = float(np.linalg.norm(v))
    if norm:
        v = v / norm
    return v.astype(dtype).tobytes()


def unpack(rows: Sequence[bytes], dtype: str = VECTOR_DTYPE):
    """`pack` outputs, one per row, as an (n, dims) matrix of `dtype`.

    Rows are taken separately so their widths can be checked: joined, a
    set of 2-d rows is indistinguishable from fewer 3-d ones. Raises
    ValueError on rows of mixed width.
    """
    import numpy as np

    itemsize = np.dtype(dtype).itemsize
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(f"stored vectors have mixed byte widths {sorted(widths)}")
    width = widths.pop() if widths else 0
    if rows and not width:
        raise ValueError("stored vectors have zero byte width")
    if width % itemsize:
        raise ValueError(f"a {width}-byte row is not whole {dtype} components")
    return np.frombuffer(b"".join(rows), dtype=dtype).reshape(
        len(rows), width // itemsize)
