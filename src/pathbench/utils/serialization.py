from __future__ import annotations
from typing import Any

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except Exception:  # no HF installed
    HfApi = None  # type: ignore


def push_to_hub(local_dir: str, repo_id: str, private: bool = True) -> None:
    if HfApi is None:
        raise RuntimeError("Install optional extra: pip install 'pathbench[hf]'")
    create_repo(repo_id, exist_ok=True, private=private)
    upload_folder(repo_id=repo_id, folder_path=local_dir)