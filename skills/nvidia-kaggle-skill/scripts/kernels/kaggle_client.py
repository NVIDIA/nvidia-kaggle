# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Kaggle REST API client with KGAT bearer-token auth, pagination, and notebook downloading."""

import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from runtime import kernel_ref as normalize_kernel_ref, require_kaggle_token

from .models import CompetitionInfo, KernelMetadata

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


class KaggleKernelClient:
    """Calls the Kaggle REST API directly with a KGAT bearer token."""

    def __init__(self, max_retries: int = 6, retry_delay: float = 5.0):
        self._token = require_kaggle_token()
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Authenticated GET with retry logic and 429 backoff. Returns parsed JSON."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(
                    headers={"Authorization": f"Bearer {self._token}"},
                    follow_redirects=True,
                    timeout=30.0,
                ) as client:
                    resp = client.get(f"{KAGGLE_API_BASE}{path}", params=params)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 429 and attempt < self._max_retries - 1:
                    wait = max(30, self._retry_delay * (2 ** attempt))
                    time.sleep(wait)
                elif attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    break
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(
            f"Kaggle API call failed after {self._max_retries} retries: {last_exc}"
        ) from last_exc

    def _get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        """Authenticated GET returning response body as text, with 429 backoff."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(
                    headers={"Authorization": f"Bearer {self._token}"},
                    follow_redirects=True,
                    timeout=60.0,
                ) as client:
                    resp = client.get(f"{KAGGLE_API_BASE}{path}", params=params)
                    resp.raise_for_status()
                    return resp.text
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 429 and attempt < self._max_retries - 1:
                    wait = max(30, self._retry_delay * (2 ** attempt))
                    time.sleep(wait)
                elif attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    break
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(
            f"Kaggle API call failed after {self._max_retries} retries: {last_exc}"
        ) from last_exc

    @staticmethod
    def _split_kernel_ref(kernel_ref: str) -> tuple[str, str]:
        normalized_ref = normalize_kernel_ref(kernel_ref)
        if "/" not in normalized_ref:
            raise RuntimeError(
                f"Invalid kernel ref '{kernel_ref}'. Expected 'owner/kernel-slug' or a Kaggle code URL."
            )
        owner, slug = normalized_ref.split("/", 1)
        return owner, slug

    def list_kernels(
        self,
        competition: Optional[str] = None,
        *,
        search: Optional[str] = None,
        kernel_type: Optional[str] = None,
        sort_by: Optional[str] = None,
        user: Optional[str] = None,
        dataset: Optional[str] = None,
        page_size: int = 20,
        max_pages: Optional[int] = None,
    ) -> list[KernelMetadata]:
        """Fetch kernels with full pagination and competition scoping."""
        all_kernels: list[KernelMetadata] = []
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            params: dict[str, Any] = {"page": page, "pageSize": page_size}
            if competition:
                params["competition"] = competition
            if search:
                params["search"] = search
            if kernel_type:
                params["kernelType"] = kernel_type
            if sort_by:
                params["sortBy"] = sort_by
            if user:
                params["user"] = user
            if dataset:
                params["dataset"] = dataset

            results = self._get("/kernels/list", params)

            if not results:
                break

            comp_id = competition or "__unscoped__"
            for k in results:
                all_kernels.append(
                    KernelMetadata(
                        competition_id=comp_id,
                        ref=k.get("ref", ""),
                        title=k.get("title", ""),
                        author=k.get("author", ""),
                        total_votes=k.get("totalVotes", 0) or 0,
                        last_run_time=k.get("lastRunTime"),
                        is_private=k.get("isPrivate", False) or False,
                    )
                )

            if len(results) < page_size:
                break
            page += 1

        return all_kernels

    def get_kernel_metadata(self, kernel_ref: str) -> dict[str, Any]:
        """Fetch kernel metadata from /kernels/pull (includes kernelDataSources, modelDataSources, etc.)."""
        user_name, kernel_slug = self._split_kernel_ref(kernel_ref)
        data = self._get(
            "/kernels/pull",
            params={"userName": user_name, "kernelSlug": kernel_slug},
        )
        metadata: dict[str, Any] = {}
        blob = data.get("blob", {})
        metadata["id"] = blob.get("id")
        metadata["id_no"] = blob.get("idNo")
        metadata["slug"] = kernel_slug
        metadata["kernel_type"] = blob.get("kernelType")
        metadata["language"] = blob.get("language")
        meta = data.get("metadata", {})
        if meta:
            for key in ("kernelDataSources", "modelDataSources", "datasetDataSources", "competitionDataSources"):
                if key in meta:
                    metadata[key] = meta[key]
        return metadata

    def get_kernel_output_log(self, kernel_ref: str) -> str:
        """Fetch kernel output log (stdout/stderr) from /kernels/output."""
        user_name, kernel_slug = self._split_kernel_ref(kernel_ref)
        text = self._get_text(
            "/kernels/output",
            params={"userName": user_name, "kernelSlug": kernel_slug},
        )
        try:
            data = json.loads(text)
            for key in ("log", "output", "stdout", "stderr"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            return text
        except json.JSONDecodeError:
            return text

    def get_competition_info(self, competition_slug: str) -> CompetitionInfo:
        """Fetch competition metadata from the Kaggle API."""
        results = self._get("/competitions/list", params={"search": competition_slug})

        match = {}
        if isinstance(results, list):
            for item in results:
                ref = item.get("ref", "") or ""
                if ref.rstrip("/").endswith(f"/{competition_slug}"):
                    match = item
                    break

        return CompetitionInfo(
            competition_id=competition_slug,
            title=match.get("title") or competition_slug,
            description=(match.get("description") or "")[:2000],
            evaluation_metric=match.get("evaluationMetric") or "",
            url=match.get("url") or f"https://www.kaggle.com/competitions/{competition_slug}",
            deadline=match.get("deadline"),
        )

    def pull_kernel(self, kernel_ref: str, dest_dir: str | Path) -> Path:
        """Download kernel source as .ipynb and metadata to a local directory."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        user_name, kernel_slug = self._split_kernel_ref(kernel_ref)
        data = self._get(
            "/kernels/pull",
            params={"userName": user_name, "kernelSlug": kernel_slug},
        )

        blob = data.get("blob", {})
        source = blob.get("source") or ""

        ipynb_file = dest / f"{kernel_slug}.ipynb"
        ipynb_file.write_text(source, encoding="utf-8")

        metadata: dict[str, Any] = {
            "id": blob.get("id"),
            "id_no": blob.get("idNo"),
            "slug": kernel_slug,
            "kernel_type": blob.get("kernelType"),
            "language": blob.get("language"),
        }
        meta = data.get("metadata", {})
        for key in ("kernelDataSources", "modelDataSources", "datasetDataSources", "competitionDataSources"):
            if key in meta:
                metadata[key] = meta[key]

        meta_file = dest / "kernel-metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return dest
