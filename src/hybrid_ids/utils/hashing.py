"""
Cryptographic hashing utilities for dataset and artifact integrity verification.
"""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file efficiently using block-wise reads."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute the SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
