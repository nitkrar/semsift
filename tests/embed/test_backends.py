"""Embedding backends, prefixes, pooling and the vector space.

Queries can need a different encoding shape from documents, and a
backend change moves vectors into a different space, so both behaviors
must be pinned. No test may reach the network: FakeEncoder is the only
backend constructed unguarded.
"""

from __future__ import annotations

import json
import inspect
import math
import unittest
import unittest.mock

from semsift.embed import backends
from semsift.embed import backends as embeddings
from semsift.embed import policy
from semsift.embed.space import VectorSpace


class FakeEncoderTests(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        a = embeddings.FakeEncoder(dims=8)
        b = embeddings.FakeEncoder(dims=8)
        self.assertEqual(a.encode(["hello"]), b.encode(["hello"]))

    def test_different_text_gives_different_vectors(self) -> None:
        e = embeddings.FakeEncoder(dims=8)
        self.assertNotEqual(e.encode(["a"])[0], e.encode(["b"])[0])

    def test_respects_requested_dims(self) -> None:
        for dims in (16, 384):
            with self.subTest(dims=dims):
                e = embeddings.FakeEncoder(dims=dims)
                self.assertEqual(dims, len(e.encode(["x"])[0]))
                self.assertEqual(dims, e.dims)

    def test_dims_must_be_positive(self) -> None:
        for dims in (0, -1):
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    embeddings.FakeEncoder(dims=dims)

    def test_encodes_a_batch(self) -> None:
        e = embeddings.FakeEncoder(dims=4)
        self.assertEqual(3, len(e.encode(["a", "b", "c"])))


class QueryAsymmetryTests(unittest.TestCase):
    """bge-family models want a prefix on the query side and nothing on
    the document side. Encoding both the same way silently costs recall,
    and nothing would fail loudly, so it is pinned."""

    def test_default_encoder_treats_query_and_document_alike(self) -> None:
        emb = embeddings.FakeEncoder(dims=8)
        self.assertEqual(emb.encode(["abc"]), emb.encode_query(["abc"]))

    def test_a_prefix_changes_only_the_query_side(self) -> None:
        emb = embeddings.FakeEncoder(dims=8, query_prefix="PREFIX: ")
        plain = embeddings.FakeEncoder(dims=8)
        self.assertEqual(plain.encode(["abc"]), emb.encode(["abc"]))
        self.assertNotEqual(emb.encode(["abc"]), emb.encode_query(["abc"]))
        # The prefix is literally prepended, not hashed in some other way.
        self.assertEqual(emb.encode(["PREFIX: abc"]), emb.encode_query(["abc"]))

    def test_a_document_prefix_changes_only_the_document_side(self) -> None:
        emb = embeddings.FakeEncoder(dims=8, doc_prefix="DOC: ")
        plain = embeddings.FakeEncoder(dims=8)
        self.assertEqual(plain.encode(["DOC: abc"]), emb.encode(["abc"]))
        self.assertEqual(plain.encode_query(["abc"]), emb.encode_query(["abc"]))

    def test_known_models_carry_a_recommended_prefix(self) -> None:
        self.assertTrue(policy.default_query_prefix("BAAI/bge-small-en-v1.5"))
        self.assertEqual("", policy.default_query_prefix("minishlab/potion-base-8M"))

    def test_a_task_instruction_is_never_a_default(self) -> None:
        """Models whose instruction names a task get none from the core:
        the task depends on what the caller searches."""
        for model in ("onnx-community/Qwen3-Embedding-0.6B-ONNX",
                      "nomic-ai/CodeRankEmbed"):
            self.assertEqual("", policy.default_query_prefix(model), model)

    def test_a_caller_table_extends_the_defaults(self) -> None:
        own = {"qwen3-embedding": "Instruct: find mail\nQuery: "}
        self.assertEqual(own["qwen3-embedding"], policy.default_query_prefix(
            "onnx-community/Qwen3-Embedding-0.6B-ONNX", own))
        self.assertTrue(policy.default_query_prefix(
            "BAAI/bge-small-en-v1.5", own))

    def test_none_takes_the_model_default(self) -> None:
        emb = embeddings.FakeEncoder(dims=8)
        self.assertEqual("", emb.query_prefix)
        self.assertEqual(
            policy.default_query_prefix("BAAI/bge-small-en-v1.5"),
            policy.resolve_prefix(None, "BAAI/bge-small-en-v1.5", "query"))

    def test_empty_string_is_an_override_not_an_absence(self) -> None:
        """None means 'use the recommendation'; '' means 'no prefix'."""
        self.assertEqual(
            "", policy.resolve_prefix("", "BAAI/bge-small-en-v1.5", "query"))
        self.assertEqual(
            "mine: ",
            policy.resolve_prefix("mine: ", "BAAI/bge-small-en-v1.5", "query"))


class PoolingTests(unittest.TestCase):
    """Pooling is per-model and getting it wrong fails silently.

    bge uses CLS, e5 and MiniLM use mean, Qwen3-Embedding uses the last
    non-pad token. All three produce a plausible unit vector, so a wrong
    choice shows up only as bad ranking.
    """

    def setUp(self) -> None:
        import numpy as np

        self.np = np
        # batch of 2, 4 tokens, 3 dims. Second sequence is 2 tokens + pad.
        self.h = np.array([
            [[1., 0, 0], [2., 0, 0], [3., 0, 0], [4., 0, 0]],
            [[10., 0, 0], [20., 0, 0], [99., 0, 0], [99., 0, 0]],
        ], dtype="float32")
        self.mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype="int64")

    def test_cls_takes_the_first_token(self) -> None:
        got = embeddings._pool(self.h, self.mask, "cls")
        self.assertEqual([1.0, 10.0], [got[0][0], got[1][0]])

    def test_mean_ignores_padding(self) -> None:
        got = embeddings._pool(self.h, self.mask, "mean")
        self.assertAlmostEqual(2.5, float(got[0][0]))    # (1+2+3+4)/4
        self.assertAlmostEqual(15.0, float(got[1][0]))   # (10+20)/2, not /4

    def test_last_takes_the_last_real_token_not_the_last_row(self) -> None:
        """With right padding, hidden[:, -1] is a pad vector for every
        sequence shorter than the batch maximum."""
        got = embeddings._pool(self.h, self.mask, "last")
        self.assertAlmostEqual(4.0, float(got[0][0]))
        self.assertAlmostEqual(20.0, float(got[1][0]))   # not 99
        self.assertNotAlmostEqual(99.0, float(got[1][0]))

    def test_known_models_map_to_their_trained_pooling(self) -> None:
        self.assertEqual("cls", policy.default_pooling("BAAI/bge-small-en-v1.5"))
        self.assertEqual("last", policy.default_pooling(
            "onnx-community/Qwen3-Embedding-0.6B-ONNX"))
        self.assertEqual("mean", policy.default_pooling("nomic-ai/CodeRankEmbed"))


class StaticEncoderTests(unittest.TestCase):
    """Guarded: needs a cached model. Skips, never fails, so the suite
    stays offline-safe."""

    MODEL = "minishlab/potion-code-16M-v2"

    def setUp(self) -> None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            for name in ("tokenizer.json", "model.safetensors", "config.json"):
                hf_hub_download(self.MODEL, name, local_files_only=True)
        except LocalEntryNotFoundError:
            self.skipTest("model unavailable")
        self.emb = embeddings.StaticEncoder(self.MODEL)

    def test_a_vector_does_not_depend_on_its_batch(self) -> None:
        """The model's tokenizer pads a batch to its longest text, and
        model2vec averages the padding in, so a short text beside a long
        one came out different from the same text alone."""
        short = "def add(a, b): return a + b"
        alone = self.emb.encode([short])[0]
        batched = self.emb.encode([short, "word " * 400])[0]
        self.assertEqual(list(alone), list(batched))


class OnnxEncoderTests(unittest.TestCase):
    """Guarded: needs onnxruntime and a cached model. Skips, never fails,
    so the suite stays offline-safe."""

    MODEL = "BAAI/bge-small-en-v1.5"
    #: Pin the provider because these tests cover pooling and prefixes;
    #: provider selection is asserted against a fake onnxruntime below.
    PROVIDERS = "cpu"

    def setUp(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            self.skipTest("onnx extra not installed")
        try:
            hf_hub_download(self.MODEL, "tokenizer.json", local_files_only=True)
            hf_hub_download(
                self.MODEL, "onnx/model.onnx", local_files_only=True)
        except LocalEntryNotFoundError:
            self.skipTest("model unavailable")
        self.emb = embeddings.OnnxEncoder(
            self.MODEL, local_only=True, providers=self.PROVIDERS
        )

    def test_dims_are_learned_not_declared(self) -> None:
        self.assertEqual(384, self.emb.dims)

    def test_vectors_are_unit_length(self) -> None:
        import math

        v = self.emb.encode(["a chunk of text about payment handling"])[0]
        self.assertAlmostEqual(1.0, math.sqrt(sum(x * x for x in v)), places=4)

    def test_it_separates_two_unrelated_texts(self) -> None:
        """The reason this backend exists, and not a tautology: a wrong
        pooling or prefix choice still yields a plausible unit vector,
        and shows up only as the near text scoring below the far
        one."""
        docs = self.emb.encode([
            "records a delivery so the same notification is not sent twice",
            "locate an executable on the PATH and return its absolute path",
        ])
        q = self.emb.encode_query(["stop duplicate notifications to an agent"])[0]
        near = sum(a * b for a, b in zip(docs[0], q))
        far = sum(a * b for a, b in zip(docs[1], q))
        self.assertGreater(near, far)

    def test_the_default_prefix_is_applied(self) -> None:
        self.assertTrue(self.emb.query_prefix)
        self.assertNotEqual(self.emb.encode(["x"]), self.emb.encode_query(["x"]))


class VectorSpaceTests(unittest.TestCase):
    """Everything that decides whether two stored vectors compare."""

    def test_an_encoder_reports_the_space_it_writes(self) -> None:
        emb = embeddings.FakeEncoder(dims=8, doc_prefix="d: ")
        self.assertEqual(
            VectorSpace.of("fake", "fake", doc_prefix="d: ", dims=8), emb.space)

    def test_the_query_prefix_is_not_part_of_the_space(self) -> None:
        a = embeddings.FakeEncoder(dims=8, query_prefix="q: ")
        b = embeddings.FakeEncoder(dims=8)
        self.assertEqual(a.space, b.space)
        self.assertNotEqual(a.encode_query(["same"]), b.encode_query(["same"]))

    def test_each_field_separates_spaces(self) -> None:
        base = VectorSpace.of("m", "onnx", dims=8)
        for other in (VectorSpace.of("n", "onnx", dims=8),
                      VectorSpace.of("m", "http", dims=8),
                      VectorSpace.of("m", "onnx", dims=16),
                      VectorSpace.of("m", "onnx", doc_prefix="p: ", dims=8),
                      VectorSpace.of("m", "onnx", pooling="last", dims=8),
                      VectorSpace.of("m", "onnx", variant="q4.onnx", dims=8)):
            self.assertNotEqual(base, other)

    def test_defaults_resolve_without_loading_a_model(self) -> None:
        space = VectorSpace.of("intfloat/e5-small-v2", "onnx")
        self.assertEqual(policy.default_doc_prefix("intfloat/e5-small-v2"),
                         space.doc_prefix)
        self.assertEqual("mean", space.pooling)
        self.assertEqual(0, space.dims)

    def test_an_http_space_names_its_endpoint(self) -> None:
        """One model name served by two servers is not one model."""
        emb = embeddings.HttpEncoder("http://a:8080", "m", probe=False)
        self.assertEqual(VectorSpace.of("m", "http", variant="http://a:8080"),
                         emb.space)

    def test_equivalent_http_endpoints_name_the_same_space(self) -> None:
        a = embeddings.HttpEncoder("http://a:8080", "m", probe=False)
        b = embeddings.HttpEncoder("http://a:8080/", "m", probe=False)
        self.assertEqual(a.space, b.space)
        self.assertEqual(
            VectorSpace.of("m", "http", variant="http://a:8080/"), b.space)

    def test_only_onnx_pools(self) -> None:
        for backend in ("static", "http", "fake"):
            self.assertEqual(
                "", VectorSpace.of("BAAI/bge-small-en-v1.5", backend).pooling)


class HttpEncoderTests(unittest.TestCase):
    def response(self, data):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"data": data}).encode()
        return response

    def test_response_indices_restore_input_order(self) -> None:
        emb = embeddings.HttpEncoder("http://host", "model", probe=False)
        data = [{"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]}]
        with unittest.mock.patch.object(backends.urllib.request, "urlopen",
                                        return_value=self.response(data)):
            self.assertEqual([[1.0, 0.0], [0.0, 1.0]],
                             emb._post(["first", "second"]))

    def test_missing_response_index_is_refused(self) -> None:
        emb = embeddings.HttpEncoder("http://host", "model", probe=False)
        data = [{"index": 0, "embedding": [1.0, 0.0]},
                {"index": 2, "embedding": [0.0, 1.0]}]
        with unittest.mock.patch.object(backends.urllib.request, "urlopen",
                                        return_value=self.response(data)):
            with self.assertRaises(ValueError):
                emb._post(["first", "second", "third"])

    def test_non_finite_vector_values_are_refused(self) -> None:
        emb = embeddings.HttpEncoder("http://host", "model", probe=False)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                data = [{"index": 0, "embedding": [value]}]
                with unittest.mock.patch.object(
                        backends.urllib.request, "urlopen",
                        return_value=self.response(data)):
                    with self.assertRaises(ValueError):
                        emb._post(["text"])

    def test_empty_vectors_are_refused(self) -> None:
        emb = embeddings.HttpEncoder("http://host", "model", probe=False)
        data = [{"index": 0, "embedding": []}]
        with unittest.mock.patch.object(backends.urllib.request, "urlopen",
                                        return_value=self.response(data)):
            with self.assertRaises(ValueError):
                emb._post(["text"])


class _FakeSession:
    def __init__(self, path, arg=None, providers=None):
        self.path, self.arg, self.providers = path, arg, providers


class _FakeOptions:
    def __init__(self) -> None:
        self.devices = None

    def add_provider_for_devices(self, devices, options) -> None:
        self.devices = devices


class _FakeDevice:
    def __init__(self, ep_name: str) -> None:
        self.ep_name = ep_name


class _FakeOrt:
    """Enough onnxruntime to see which attachment path was taken."""

    InferenceSession = _FakeSession
    SessionOptions = _FakeOptions

    def __init__(self, devices=()) -> None:
        self._devices = devices

    def get_available_providers(self):
        return ["CPUExecutionProvider"]

    def get_ep_devices(self):
        return list(self._devices)


class ProviderAttachmentTests(unittest.TestCase):
    """A plugin provider named in `providers=` is dropped rather than
    refused: the session builds, reports success, and runs on CPU. So
    the two attachment paths are pinned by which one each choice takes,
    not by the session merely constructing."""

    def test_a_named_provider_goes_through_the_providers_argument(self) -> None:
        sess = backends._session(_FakeOrt(), "m.onnx", "cpu")
        self.assertEqual(["CPUExecutionProvider"], sess.providers)
        self.assertIsNone(sess.arg.devices)

    def test_sessions_log_errors_only(self) -> None:
        name = "WebGpuExecutionProvider"
        with unittest.mock.patch.object(backends, "_webgpu", return_value=name):
            sessions = [backends._session(_FakeOrt(), "m.onnx", "cpu"),
                        backends._session(_FakeOrt(devices=[_FakeDevice(name)]),
                                          "m.onnx", "webgpu")]
        self.assertEqual([3, 3], [sess.arg.log_severity_level for sess in sessions])

    def test_default_does_not_require_the_webgpu_extra(self) -> None:
        default = inspect.signature(
            embeddings.OnnxEncoder).parameters["providers"].default
        self.assertEqual("best", default)
        with unittest.mock.patch.object(backends, "_webgpu", side_effect=ImportError):
            self.assertIn(backends._best(_FakeOrt()), ("cpu", "CUDAExecutionProvider"))

    def test_webgpu_attaches_by_device_and_never_by_name(self) -> None:
        name = "WebGpuExecutionProvider"
        ort = _FakeOrt(devices=[_FakeDevice("CPUExecutionProvider"),
                                _FakeDevice(name)])
        with unittest.mock.patch.object(backends, "_webgpu",
                                        return_value=name):
            sess = backends._session(ort, "m.onnx", "webgpu")
        # `providers=` unused: a plugin name there would silently run on CPU.
        self.assertIsNone(sess.providers)
        self.assertEqual([name], [d.ep_name for d in sess.arg.devices])

    def test_webgpu_without_a_device_fails_rather_than_falling_back(self) -> None:
        with unittest.mock.patch.object(backends, "_webgpu",
                                        return_value="WebGpuExecutionProvider"):
            with self.assertRaises(ValueError):
                backends._session(_FakeOrt(), "m.onnx", "webgpu")


if __name__ == "__main__":
    unittest.main()
