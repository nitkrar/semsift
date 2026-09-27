"""The models and texts the golden vectors cover."""

from __future__ import annotations

MODELS = [("minishlab/potion-base-8M", "static"),
          ("minishlab/potion-code-16M-v2", "static"),
          ("BAAI/bge-small-en-v1.5", "onnx")]

TEXTS = ["settle the ledger balance at month end",
         "def parse_config(path): return toml.load(path)",
         "Meeting moved to Thursday; bring the lease papers"]


def revision(model: str) -> str | None:
    """The snapshot selected by the local `main` ref, or None."""
    from huggingface_hub import scan_cache_dir

    try:
        repos = {r.repo_id: r for r in scan_cache_dir().repos}
    except Exception:
        return None
    repo = repos.get(model)
    if repo is None:
        return None
    main = repo.refs.get("main")
    return None if main is None else main.commit_hash
