# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Project path helpers for Kaggle kernel storage."""

from pathlib import Path

from runtime import find_project_root


def default_db_path(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "data" / "kernels.db"


def default_notebook_cache_dir(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "data" / "notebooks"
