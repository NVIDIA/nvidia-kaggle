#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared rendering helpers for Kaggle database info scripts."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from rich.console import Console
from rich.table import Table

from constants import DATE_PREVIEW_CHARS


class CompetitionSummary(Protocol):
    top_authors: Sequence[tuple[str, int]]
    vote_stats: Mapping[str, float]
    date_range: tuple[str, str] | None


class CompetitionSummaryDatabase(Protocol):
    def competition_summary(self, competition_id: str) -> CompetitionSummary:
        ...


def show_competition_detail(
    console: Console,
    db: CompetitionSummaryDatabase,
    competition_id: str,
    *,
    count_attr: str,
    entity_plural: str,
    author_count_label: str,
) -> None:
    if not competition_id:
        raise ValueError("competition_id is required.")
    if not count_attr or not entity_plural or not author_count_label:
        raise ValueError("Display labels and count attribute are required.")

    summary = db.competition_summary(competition_id)
    entity_count = getattr(summary, count_attr)

    if entity_count == 0:
        console.print(
            f"[yellow]No {entity_plural} found for competition "
            f"'{competition_id}'[/yellow]"
        )
        return

    console.print(f"\n[bold]Competition:[/bold] {competition_id}")
    console.print(f"[bold]Total {entity_plural}:[/bold] {entity_count}")

    if summary.date_range:
        console.print(
            f"[bold]Date range:[/bold] {summary.date_range[0][:DATE_PREVIEW_CHARS]} "
            f"to {summary.date_range[1][:DATE_PREVIEW_CHARS]}"
        )

    console.print(
        f"\n[bold]Vote stats:[/bold] min={summary.vote_stats.get('min', 0)}, "
        f"max={summary.vote_stats.get('max', 0)}, "
        f"avg={summary.vote_stats.get('avg', 0)}"
    )

    author_table = Table(title="Top Authors")
    author_table.add_column("Author", style="cyan")
    author_table.add_column(author_count_label, justify="right", style="green")
    for author, count in summary.top_authors:
        author_table.add_row(author, str(count))
    console.print(author_table)
