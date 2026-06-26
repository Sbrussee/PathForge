from __future__ import annotations

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except Exception:  # no HF installed
    HfApi = None  # type: ignore


def push_to_hub(local_dir: str, repo_id: str, private: bool = True) -> None:
    """Create a Hub repository when needed and upload one local artifact folder."""
    if HfApi is None:
        raise RuntimeError("Install optional extra: pip install 'pathforge[hf]'")
    create_repo(repo_id, exist_ok=True, private=private)
    upload_folder(repo_id=repo_id, folder_path=local_dir)
