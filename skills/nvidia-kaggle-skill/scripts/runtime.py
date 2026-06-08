#!/usr/bin/env python3
"""Runtime helpers shared by the unified Kaggle skill scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path


def competition_slug(value: str) -> str:
    """Extract a Kaggle competition slug from a slug or competition URL."""
    match = re.search(r"kaggle\.com/competitions/([^/?#]+)", value)
    return match.group(1) if match else value.strip().strip("/")


def kernel_ref(value: str) -> str:
    """Extract owner/kernel-slug from a Kaggle code URL or return a ref as-is."""
    match = re.search(r"kaggle\.com/code/([^/?#]+)/([^/?#]+)", value)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return value.strip()


def require_kaggle_token() -> str:
    """Return KAGGLE_API_TOKEN or raise a clear runtime error."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("Kaggle credentials not found. Set KAGGLE_API_TOKEN in the environment.")
    return token


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing pyproject.toml, or fall back to cwd."""
    explicit = os.environ.get("PROJECT_ROOT")
    if explicit:
        return Path(explicit).resolve()

    candidates = [Path.cwd().resolve()]
    if start:
        candidates.insert(0, start.resolve())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        probe = candidate if candidate.is_dir() else candidate.parent
        for parent in (probe, *probe.parents):
            if (parent / "pyproject.toml").exists():
                return parent

    return Path.cwd().resolve()


def load_project_env(root: Path | None = None) -> Path:
    """Load .env from the project root and return that root."""
    project_root = root or find_project_root()
    env_path = project_root / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
    return project_root
