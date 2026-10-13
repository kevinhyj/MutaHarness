"""Small, shared helpers for the repository's frozen artifacts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
