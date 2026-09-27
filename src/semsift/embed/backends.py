"""Text to vectors: the encoder protocol and its backends.

The same encoder must encode documents and queries; `space` names what
has to match for stored vectors to stay comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

from .policy import default_pooling, resolve_prefix
from .space import VectorSpace


class Encoder(Protocol):
    """Backends implement `_encode`; callers use `encode`/`encode_query`.

    Only the public pair applies the side-specific prefixes, so a backend
    that encoded through `_encode` directly would skip the model's
    query/document markers.
    """

    name: str
    backend: str
    dims: int
    query_prefix: str
    doc_prefix: str

    @property
    def space(self) -> VectorSpace: ...

    def _encode(self, texts: Sequence[str]) -> list[Sequence[float]]: ...

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]: ...

    def encode_query(self, texts: Sequence[str]) -> list[Sequence[float]]: ...


class _PrefixMixin:
    """Side-specific prefixes and the vector space, in one place.

    Retrieval models are trained with markers that differ by side. bge
    marks only the query; e5 wants `query: ` and `passage: `; nomic wants
    `search_query: ` and `search_document: `. Applying the wrong one, or
    only half the pair, puts queries and documents in different regions
    of the space and nothing errors -- recall just drops. Backends define
    `_encode` and inherit both public methods so no backend can get this
    subtly different.
    """

    name: str
    backend: str
    dims: int
    query_prefix: str = ""
    doc_prefix: str = ""
    pooling: str = ""
    variant: str = ""

    def _prefixes(self, query_prefix: str | None,
                  doc_prefix: str | None) -> None:
        """`None` takes the model family's default; `''` means none."""
        self.query_prefix = resolve_prefix(query_prefix, self.name, "query")
        self.doc_prefix = resolve_prefix(doc_prefix, self.name, "doc")

    @property
    def space(self) -> VectorSpace:
        return VectorSpace(model=self.name, backend=self.backend,
                           variant=self.variant, dims=self.dims,
                           doc_prefix=self.doc_prefix, pooling=self.pooling)

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Documents, with the model's document-side marker if it has one."""
        if not self.doc_prefix:
            return self._encode(texts)         # type: ignore[attr-defined]
        return self._encode(                   # type: ignore[attr-defined]
            [self.doc_prefix + t for t in texts]
        )

    def encode_query(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return self._encode(                   # type: ignore[attr-defined]
            [self.query_prefix + t for t in texts] if self.query_prefix
            else list(texts)
        )


def resolve_model_source(model: str) -> str:
    """A hub id -> a local snapshot path, without the network if cached.

    `StaticModel.from_pretrained` defaults `force_download=True`, so a
    bare model id re-resolves through huggingface_hub on every
    construction, dominating startup. One process per query is exactly
    the CLI's shape, so that cost is paid on every invocation.

    Falls back to a networked resolve when nothing is cached, and to
    the id itself when that fails too -- the model loader's own error
    is more useful than one invented here.
    """
    if Path(model).exists():
        return model
    from huggingface_hub import snapshot_download

    for local_only in (True, False):
        try:
            return snapshot_download(model, local_files_only=local_only)
        except Exception:
            continue
    return model


class StaticEncoder(_PrefixMixin):
    """model2vec. Runs in process, no server."""

    backend = "static"

    def __init__(self, model: str, *, query_prefix: str | None = None,
                 doc_prefix: str | None = None) -> None:
        from model2vec import StaticModel

        self.name = model
        self._prefixes(query_prefix, doc_prefix)
        self._model = StaticModel.from_pretrained(resolve_model_source(model))
        self.dims = int(self._model.dim)

    def _encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts))]


class HttpEncoder(_PrefixMixin):
    """OpenAI-compatible /v1/embeddings, e.g. a local llama-server.

    urllib only; no client library.
    """

    backend = "http"

    def __init__(self, endpoint: str, model: str, *, api_key: str = "",
                 probe: bool = True, query_prefix: str | None = None,
                 doc_prefix: str | None = None) -> None:
        self.name = model
        self._prefixes(query_prefix, doc_prefix)
        self.endpoint = endpoint.rstrip("/")
        self.variant = self.endpoint
        self._api_key = api_key
        self.dims = 0
        if probe:
            # Learned eagerly, with one request, because `dims` is part of
            # the vector space. Left at 0 until the first real encode, a
            # consumer comparing a fresh encoder's space against a stored
            # one would see a change on every run.
            self.dims = len(self._encode(["probe"])[0])

    #: Requests carry at most this many texts. A large single POST can
    #: exceed the server's batch limit or time out, so the transport
    #: splits it without requiring callers to do so.
    BATCH = 64

    def _encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        out: list[Sequence[float]] = []
        for i in range(0, len(texts), self.BATCH):
            out.extend(self._post(list(texts[i : i + self.BATCH])))
        return out

    def _post(self, texts: list[str]) -> list[Sequence[float]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = json.dumps({"input": texts, "model": self.name}).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/v1/embeddings", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
        # Order is not guaranteed by the spec. Every input index must be
        # present exactly once; sorting an incomplete response would attach
        # later vectors to the wrong texts.
        items = payload["data"]
        indices = [item.get("index") for item in items]
        expected = list(range(len(texts)))
        if (any(type(index) is not int for index in indices)
                or sorted(indices) != expected):
            raise ValueError(
                f"embedding response indices {indices!r}; expected {expected!r}")
        items = sorted(items, key=lambda d: d["index"])
        vectors = [item["embedding"] for item in items]
        widths = {len(vector) for vector in vectors}
        if len(widths) > 1:
            raise ValueError(
                f"embedding response has mixed vector widths {sorted(widths)}")
        if widths == {0}:
            raise ValueError("embedding response contains empty vectors")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for vector in vectors for value in vector):
            raise ValueError("embedding response contains a non-finite value")
        if widths and self.dims and widths != {self.dims}:
            raise ValueError(
                f"embedding response width {next(iter(widths))}; expected {self.dims}")
        if vectors and not self.dims:
            self.dims = len(vectors[0])
        return vectors


class FakeEncoder(_PrefixMixin):
    """Deterministic vectors derived from the text hash. Tests only."""

    backend = "fake"

    def __init__(self, dims: int = 8, *, query_prefix: str | None = None,
                 doc_prefix: str | None = None) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self.name = "fake"
        self.dims = dims
        self._prefixes(query_prefix, doc_prefix)

    def _encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        out: list[Sequence[float]] = []
        for text in texts:
            encoded = text.encode()
            digest = (hashlib.blake2b(encoded, digest_size=self.dims).digest()
                      if self.dims <= 64
                      else hashlib.shake_256(encoded).digest(self.dims))
            out.append([(b / 255.0) - 0.5 for b in digest])
        return out


#: Sequences per inference call. Narrow rather than wide: peak memory
#: scales with the batch, while throughput does not once the batch is
#: grouped by length.
_BATCH = 16


class OnnxEncoder(_PrefixMixin):
    """A sentence transformer through onnxruntime. No torch, no server.

    It costs an optional dependency and is slower than static embedding.

    `dims` is learned by encoding once at construction rather than read
    from config.json, because the pooling choice -- not the hidden size
    alone -- decides the output width.
    """

    backend = "onnx"

    def __init__(self, repo_id: str, *, local_only: bool = False,
                 pooling: str | None = None, filename: str = "onnx/model.onnx",
                 providers: str = "auto", query_prefix: str | None = None,
                 doc_prefix: str | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:      # pragma: no cover - environment
            raise ImportError(
                "OnnxEncoder needs the onnx extra: pip install 'semsift[onnx]'"
            ) from exc
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.name = repo_id
        self._prefixes(query_prefix, doc_prefix)
        self.variant = filename
        self.pooling = pooling or default_pooling(repo_id)
        if self.pooling not in ("cls", "mean", "last"):
            raise ValueError(f"unknown pooling {self.pooling!r}")
        get = lambda f: hf_hub_download(  # noqa: E731
            repo_id, f, local_files_only=local_only
        )
        self._tok = Tokenizer.from_file(get("tokenizer.json"))
        self._tok.enable_truncation(512)
        self._tok.enable_padding()
        path = get(filename)
        # Models over 2 GB store weights beside the graph; fetch the
        # sidecar or the session loads a graph with no tensors.
        if filename.endswith(".onnx"):
            try:
                get(filename + "_data")
            except Exception:       # most models have no sidecar
                pass
        self._sess = _session(ort, path, providers)
        self.providers = self._sess.get_providers()
        self._inputs = {i.name for i in self._sess.get_inputs()}
        # Decoder-style exports (Qwen3-Embedding and friends) declare
        # position_ids and a full empty KV cache as required inputs, two
        # entries per layer. Encoder exports (bge, MiniLM) have none of
        # this, so it stays empty for them.
        self._kv = [
            (i.name, i.shape, i.type) for i in self._sess.get_inputs()
            if i.name.startswith("past_key_values.")
        ]
        self._wants_positions = "position_ids" in self._inputs
        self.dims = len(self._encode(["probe"])[0])

    def _decoder_inputs(self, batch: int, length: int) -> dict:
        """position_ids and a zero-length KV cache, sized from the graph."""
        import numpy as np

        extra: dict = {}
        if self._wants_positions:
            extra["position_ids"] = np.tile(
                np.arange(length, dtype=np.int64), (batch, 1)
            )
        for name, shape, dtype in self._kv:
            # shape is [batch, heads, past_sequence_length, head_dim];
            # past length is 0 because we never reuse a cache.
            dims = [batch, int(shape[1]), 0, int(shape[3])]
            extra[name] = np.zeros(
                dims, dtype=np.float16 if "float16" in dtype else np.float32
            )
        return extra

    def _encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        import numpy as np

        # A batch is padded to its longest member. Grouping by length first
        # limits padding, then restoring the caller's order preserves the
        # input-to-output mapping.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[Sequence[float]] = [()] * len(texts)
        for i in range(0, len(order), _BATCH):
            window = order[i : i + _BATCH]
            enc = self._tok.encode_batch([texts[j] for j in window])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array(
                    [e.attention_mask for e in enc], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [e.type_ids for e in enc], dtype=np.int64
                ),
            }
            run = {k: v for k, v in feed.items() if k in self._inputs}
            if self._kv or self._wants_positions:
                run.update(
                    self._decoder_inputs(*feed["input_ids"].shape)
                )
            raw = self._sess.run(None, run)[0]
            vec = (raw if raw.ndim == 2
                   else _pool(raw, feed["attention_mask"], self.pooling))
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
            for j, v in zip(window, vec):
                out[j] = v.tolist()
        return out


def _pool(hidden, mask, how: str):
    """Token vectors -> one sentence vector.

    Padding must be excluded for `mean` and `last`, which is why the
    attention mask is threaded through rather than taking `hidden[:, -1]`:
    with right padding the final row is a pad token for every sequence
    shorter than the batch maximum, so the naive version would embed
    padding and the error would be invisible.
    """
    import numpy as np

    if how == "cls":
        return hidden[:, 0]
    m = mask.astype("float32")
    if how == "mean":
        return (hidden * m[:, :, None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1e-9)
    idx = np.maximum(m.sum(1).astype("int64") - 1, 0)
    return hidden[np.arange(hidden.shape[0]), idx]


#: Set once `register_execution_provider_library` has run for webgpu.
#: Registration is process-wide and a second call raises.
_WEBGPU_NAME: str | None = None


def _webgpu(ort) -> str:
    """Register the webgpu plugin library and return its provider name."""
    global _WEBGPU_NAME

    if _WEBGPU_NAME is None:
        try:
            import onnxruntime_ep_webgpu as plugin
        except ImportError as exc:
            raise ImportError(
                "providers='webgpu' needs the webgpu extra: pip install"
                " 'semsift[webgpu]', or use providers='cpu'"
            ) from exc
        name = plugin.get_ep_name()
        ort.register_execution_provider_library(name, plugin.get_library_path())
        _WEBGPU_NAME = name
    return _WEBGPU_NAME


def _session(ort, path: str, choice: str):
    """An inference session bound to the chosen execution provider.

    webgpu ships as a plugin provider, which `providers=` cannot reach:
    a name onnxruntime does not recognise there is dropped and the
    session runs on CPU reporting success. Plugins attach by device
    instead, so they take a separate path rather than a longer list.
    """
    if choice != "webgpu":
        return ort.InferenceSession(path, providers=_providers(ort, choice))
    name = _webgpu(ort)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devices:
        raise ValueError(
            f"{name} registered but exposes no device; "
            "use providers='cpu'"
        )
    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})
    return ort.InferenceSession(path, opts)


def _providers(ort, choice: str) -> list[str]:
    """Execution providers for onnxruntime, most preferred first.

    'auto' takes CoreML or CUDA when the build offers it. CPU is always
    appended as the fallback, because CoreML silently declines operators
    it cannot compile and a session with no usable provider would fail to
    construct.
    """
    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "auto":
        wanted = [p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider")
                  if p in available]
        return wanted + ["CPUExecutionProvider"]
    if choice not in available:
        raise ValueError(
            f"execution provider {choice!r} not available; "
            f"onnxruntime offers {available}"
        )
    return [choice, "CPUExecutionProvider"]
