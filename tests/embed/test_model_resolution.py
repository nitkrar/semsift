"""Static model ids resolve from cache before falling back to network."""

from __future__ import annotations

import unittest
from unittest import mock

from semsift.embed.backends import resolve_model_source


class ResolutionOrderTests(unittest.TestCase):
    def test_cached_model_never_touches_the_network(self) -> None:
        calls = []

        def fake_download(model_id, **kw):
            calls.append(kw.get("local_files_only", False))
            return "/cache/snapshot"

        with mock.patch("huggingface_hub.snapshot_download", fake_download):
            self.assertEqual("/cache/snapshot",
                             resolve_model_source("some/model"))
        self.assertEqual([True], calls)

    def test_uncached_model_falls_back_to_the_network(self) -> None:
        """A first run on a machine with no cache must still work."""
        calls = []

        def fake_download(model_id, **kw):
            local_only = kw.get("local_files_only", False)
            calls.append(local_only)
            if local_only:
                raise OSError("not in cache")
            return "/downloaded/snapshot"

        with mock.patch("huggingface_hub.snapshot_download", fake_download):
            self.assertEqual("/downloaded/snapshot",
                             resolve_model_source("some/model"))
        self.assertEqual([True, False], calls)

    def test_a_local_path_is_used_as_is(self) -> None:
        """An explicit path is not a hub id and must not be resolved."""
        with mock.patch("huggingface_hub.snapshot_download") as download:
            source = resolve_model_source(__file__)
        self.assertEqual(__file__, source)
        download.assert_not_called()

    def test_resolution_failure_returns_the_id_unchanged(self) -> None:
        """If the hub is unreachable and nothing is cached, hand the id
        back and let the model loader raise its own error rather than
        masking it with ours."""
        with mock.patch("huggingface_hub.snapshot_download",
                        side_effect=OSError("offline")):
            self.assertEqual("some/model", resolve_model_source("some/model"))


if __name__ == "__main__":
    unittest.main()
