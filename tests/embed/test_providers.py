"""providers="best": chosen from what the machine is, without running the model."""

from __future__ import annotations

import inspect
import unittest
import unittest.mock

from semsift.embed import OnnxEncoder
from semsift.embed import backends
from semsift.rerank import CrossEncoder


class FakeOrt:
    def __init__(self, available=("CPUExecutionProvider",)) -> None:
        self.available = list(available)

    def get_available_providers(self):
        return list(self.available)


class BestProviderTests(unittest.TestCase):
    def best(self, *, system, available=("CPUExecutionProvider",), webgpu_device=True):
        with unittest.mock.patch("platform.system", return_value=system), \
             unittest.mock.patch.object(backends, "_webgpu_device", return_value=webgpu_device):
            return backends._best(FakeOrt(available))

    def test_a_mac_with_a_webgpu_device_uses_webgpu(self) -> None:
        self.assertEqual("webgpu", self.best(system="Darwin"))

    def test_a_mac_without_the_plugin_or_a_device_uses_the_cpu(self) -> None:
        # Not CoreML: it runs these graphs slower than the CPU.
        self.assertEqual("cpu", self.best(system="Darwin", webgpu_device=False,
                                          available=("CoreMLExecutionProvider",
                                                     "CPUExecutionProvider")))

    def test_a_machine_offering_cuda_uses_cuda(self) -> None:
        self.assertEqual("CUDAExecutionProvider",
                         self.best(system="Linux", available=("CUDAExecutionProvider",
                                                              "CPUExecutionProvider")))

    def test_linux_does_not_try_webgpu(self) -> None:
        self.assertEqual("cpu", self.best(system="Linux", webgpu_device=True))

    def test_best_opens_the_session_with_the_chosen_provider(self) -> None:
        opened = []
        with unittest.mock.patch.object(backends, "_best", return_value="cpu"), \
             unittest.mock.patch.object(backends, "_open", side_effect=lambda ort, path, choice: opened.append(choice)):
            backends._session(FakeOrt(), "m.onnx", "best")
        self.assertEqual(["cpu"], opened)


class DefaultTests(unittest.TestCase):
    def test_both_onnx_models_default_to_best(self) -> None:
        for cls in (OnnxEncoder, CrossEncoder):
            self.assertEqual("best", inspect.signature(cls).parameters["providers"].default)


if __name__ == "__main__":
    unittest.main()
