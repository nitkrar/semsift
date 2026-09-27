"""Text to vectors.

Imports nothing else from semsift, so it can become its own package.
"""

from .backends import (Encoder, FakeEncoder, HttpEncoder, OnnxEncoder,
                       StaticEncoder)
from .policy import resolve_prefix
from .space import VectorSpace

__all__ = ["Encoder", "FakeEncoder", "HttpEncoder", "OnnxEncoder",
           "StaticEncoder", "VectorSpace", "resolve_prefix"]
