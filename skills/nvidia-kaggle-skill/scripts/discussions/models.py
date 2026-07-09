# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Domain models for Kaggle discussion metadata."""

from datetime import datetime, timezone
from typing import Optional

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


class DiscussionRecord(BaseModel):
    """Metadata for a single Kaggle discussion thread."""

    competition_id: str
    discussion_id: int
    title: str
    author: str
    author_username: str = ""
    author_tier: str = ""
    votes: int = 0
    comment_count: int = 0
    body_markdown: str = ""
    url: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_fetched_at: Optional[str] = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DiscussionComment(BaseModel):
    """A single comment on a discussion thread."""

    id: Optional[int] = None
    discussion_id: int
    competition_id: str
    author: str = ""
    author_username: str = ""
    author_tier: str = ""
    votes: int = 0
    body_markdown: str = ""
    created_at: Optional[str] = None


class CompetitionSummary(BaseModel):
    """Aggregate statistics for a competition's discussions in the database."""

    competition_id: str
    discussion_count: int = 0
    top_authors: list[tuple[str, int]] = Field(default_factory=list)
    vote_stats: dict[str, float] = Field(default_factory=dict)
    date_range: Optional[tuple[str, str]] = None
    competition_info: Optional[CompetitionInfo] = None
