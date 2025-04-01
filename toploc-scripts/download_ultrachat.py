import os
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


def download_ultrachat(repo_id="stingning/ultrachat", target_dir=None):
    """
    Download the UltraChat dataset from Hugging Face.

    Args:
        repo_id (str): The Hugging Face repository ID
        target_dir (str, optional): Target directory to download files to. Defaults to current directory.
    """
    script_dir = Path(__file__).parent.absolute()
    target_dir = script_dir / target_dir
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading UltraChat dataset from {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id, local_dir=target_dir, repo_type="dataset"
    )

    print(f"UltraChat dataset downloaded to {local_dir}")
    return local_dir


if __name__ == "__main__":
    download_ultrachat(target_dir="ultrachat")
