"""Download, cache, and parse Kaggle notebook source code."""

import json
import re
from pathlib import Path
from typing import Optional

import nbformat

from .kaggle_client import KaggleKernelClient
from .models import NotebookCell, NotebookContent


def _sanitize_ref(ref: str) -> str:
    return re.sub(r"[^\w\-]", "_", ref)


class NotebookReader:
    """Fetch, cache, and parse Kaggle kernel notebooks."""

    def __init__(
        self,
        client: Optional[KaggleKernelClient] = None,
        cache_dir: str | Path = "data/notebooks",
    ):
        self._client = client
        self._cache_dir = Path(cache_dir)

    @property
    def client(self) -> KaggleKernelClient:
        if self._client is None:
            self._client = KaggleKernelClient()
        return self._client

    def _cache_path(self, competition_id: str, kernel_ref: str) -> Path:
        return self._cache_dir / competition_id / _sanitize_ref(kernel_ref)

    def _find_ipynb(self, directory: Path) -> Optional[Path]:
        """Find the .ipynb file in the cache directory."""
        candidates = sorted(directory.glob("*.ipynb"))
        if not candidates:
            return None
        if len(candidates) > 1:
            raise RuntimeError(
                f"Found multiple notebooks in cache directory '{directory}'; "
                "remove stale files or force a fresh download."
            )
        return candidates[0]

    def read_kernel(
        self,
        kernel_ref: str,
        competition_id: str = "__unscoped__",
        *,
        force_download: bool = False,
    ) -> NotebookContent:
        """Download (or load from cache) and parse a kernel's notebook."""
        cache = self._cache_path(competition_id, kernel_ref)

        if force_download or not cache.exists() or not any(cache.iterdir()):
            self.client.pull_kernel(kernel_ref, cache)

        ipynb = self._find_ipynb(cache)
        if ipynb is None:
            return NotebookContent(
                kernel_ref=kernel_ref,
                cells=[NotebookCell(cell_type="raw", source="(no notebook found)")],
            )

        try:
            nb = nbformat.read(str(ipynb), as_version=4)
        except Exception:
            raw = ipynb.read_text(errors="replace")
            return NotebookContent(
                kernel_ref=kernel_ref,
                cells=[NotebookCell(cell_type="code", source=raw)],
            )

        cells = [
            NotebookCell(
                cell_type=cell.cell_type,
                source=cell.source,
                execution_count=getattr(cell, "execution_count", None),
            )
            for cell in nb.cells
        ]
        return NotebookContent(
            kernel_ref=kernel_ref,
            cells=cells,
            metadata=dict(nb.metadata) if nb.metadata else {},
        )

    def get_metadata(self, kernel_ref: str, competition_id: str = "__unscoped__") -> dict:
        """Return the kernel-metadata.json dict, or empty dict if not cached."""
        meta_file = self._cache_path(competition_id, kernel_ref) / "kernel-metadata.json"
        if not meta_file.exists():
            return {}
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_raw_path(self, kernel_ref: str, competition_id: str = "__unscoped__") -> Optional[Path]:
        """Return the path to the cached .ipynb file, or None if not cached."""
        cache = self._cache_path(competition_id, kernel_ref)
        if not cache.exists():
            return None
        return self._find_ipynb(cache)
