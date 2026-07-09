#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Search and filter discussions in the local database."""

import argparse
import json

from rich.console import Console
from rich.table import Table

from discussions.database import DiscussionDatabase
from discussions.paths import default_db_path
from runtime import load_project_env
from constants import (
    DATE_PREVIEW_CHARS,
    DEFAULT_AUTHOR_COLUMN_WIDTH,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_TITLE_COLUMN_WIDTH,
)

load_project_env()

def query(
    competition_id: str,
    search: str = None,
    min_votes: int = None,
    author: str = None,
    sort_by: str = "votes",
    sort_order: str = "DESC",
    limit: int = DEFAULT_QUERY_LIMIT,
    as_json: bool = False,
):
    console = Console()
    db_path = default_db_path()

    with DiscussionDatabase(db_path) as db:
        discussions = db.query_discussions(
            competition_id,
            search=search,
            min_votes=min_votes,
            author=author,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )

    if not discussions:
        console.print(f"[yellow]No discussions found for '{competition_id}'[/yellow]")
        return

    if as_json:
        out = [d.model_dump(mode="json") for d in discussions]
        print(json.dumps(out, indent=2, default=str))
        return

    table = Table(title=f"Discussions for {competition_id} ({len(discussions)} results)")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Title", max_width=DEFAULT_TITLE_COLUMN_WIDTH)
    table.add_column("Author", style="green", max_width=DEFAULT_AUTHOR_COLUMN_WIDTH)
    table.add_column("Votes", justify="right", style="yellow")
    table.add_column("Comments", justify="right", style="dim")
    table.add_column("Created", style="dim")

    for d in discussions:
        created = str(d.created_at)[:DATE_PREVIEW_CHARS] if d.created_at else "-"
        table.add_row(str(d.discussion_id), d.title, d.author, str(d.votes), str(d.comment_count), created)

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Search and filter discussions")
    parser.add_argument("competition_id", help="Competition slug")
    parser.add_argument("--search", help="Free-text search in title, author, and body")
    parser.add_argument("--min-votes", type=int, help="Minimum vote count")
    parser.add_argument("--author", help="Filter by author name")
    parser.add_argument("--sort-by", default="votes", choices=["votes", "created_at", "updated_at", "comment_count", "title"])
    parser.add_argument("--sort-order", default="DESC", choices=["ASC", "DESC"])
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--as-json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    query(
        args.competition_id,
        search=args.search,
        min_votes=args.min_votes,
        author=args.author,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        limit=args.limit,
        as_json=args.as_json,
    )

if __name__ == "__main__":
    main()
