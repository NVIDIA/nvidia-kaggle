# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Domain models for Kaggle kernel metadata and notebook content."""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class CompetitionInfo(BaseModel):
    """Cached metadata about a Kaggle competition."""

    competition_id: str
    title: str = ""
    description: str = ""
    evaluation_metric: str = ""
    url: str = ""
    deadline: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KernelMetadata(BaseModel):
    """Metadata for a single Kaggle kernel/notebook."""

    competition_id: str
    ref: str  # e.g. "username/kernel-slug"
    title: str
    author: str
    total_votes: int = 0
    last_run_time: Optional[datetime] = None
    is_private: bool = False
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def slug(self) -> str:
        return self.ref.split("/")[-1] if "/" in self.ref else self.ref


class NotebookCell(BaseModel):
    """A single cell from a parsed notebook."""

    cell_type: str  # "code", "markdown", "raw"
    source: str
    execution_count: Optional[int] = None


class NotebookContent(BaseModel):
    """Parsed notebook with structured cell data."""

    kernel_ref: str
    cells: list[NotebookCell] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_readable(self) -> str:
        """Render notebook cells as human-readable markdown with code blocks."""
        parts: list[str] = []
        for cell in self.cells:
            if cell.cell_type == "markdown":
                parts.append(cell.source)
            elif cell.cell_type == "code":
                exec_label = f" [{cell.execution_count}]" if cell.execution_count else ""
                parts.append(f"```python{exec_label}\n{cell.source}\n```")
            else:
                parts.append(f"```\n{cell.source}\n```")
        return "\n\n---\n\n".join(parts)


class CompetitionSummary(BaseModel):
    """Aggregate statistics for a competition's kernels in the database."""

    competition_id: str
    kernel_count: int = 0
    top_authors: list[tuple[str, int]] = Field(default_factory=list)
    vote_stats: dict[str, float] = Field(default_factory=dict)
    date_range: Optional[tuple[str, str]] = None
    competition_info: Optional[CompetitionInfo] = None
