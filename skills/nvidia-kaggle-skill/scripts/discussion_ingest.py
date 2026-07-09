#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch Kaggle competition discussions and store them in the local database."""

import argparse
from datetime import UTC, datetime

from rich.console import Console
from rich.progress import Progress

from discussions.database import DiscussionDatabase
from discussions.discussion_client import DiscussionClient
from discussions.models import DiscussionRecord
from discussions.paths import default_db_path
from runtime import load_project_env
from constants import DEFAULT_DISCUSSION_MAX_PAGES, DEFAULT_PAGE_SIZE

load_project_env()

def _apply_author_details(discussion: DiscussionRecord, op_author: dict | None) -> None:
    if not op_author:
        return

    username = op_author.get("username")
    if username and not discussion.author_username:
        discussion.author_username = username

    tier = op_author.get("tier")
    if tier and not discussion.author_tier:
        discussion.author_tier = tier

def _store_comments(
    db: DiscussionDatabase,
    competition_id: str,
    comments: list,
) -> int:
    if not comments:
        return 0

    for comment in comments:
        comment.competition_id = competition_id
    db.upsert_comments(comments)
    return len(comments)

def _fetch_and_store_comments(
    client: DiscussionClient,
    db: DiscussionDatabase,
    discussions: list[DiscussionRecord],
    competition_id: str,
    console: Console,
) -> int:
    total_comments = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Fetching comments...", total=len(discussions))
        for discussion in discussions:
            try:
                total_msgs, comments, first_md, op_author = client.get_discussion_detail(
                    discussion.discussion_id
                )
                discussion.comment_count = total_msgs
                if first_md and not discussion.body_markdown.strip():
                    discussion.body_markdown = first_md
                _apply_author_details(discussion, op_author)
                total_comments += _store_comments(db, competition_id, comments)
            except Exception as e:
                console.print(f"[red]  Failed #{discussion.discussion_id}: {e}[/red]")
            progress.advance(task)
    return total_comments

def ingest(
    competition_id: str,
    max_pages: int = DEFAULT_DISCUSSION_MAX_PAGES,
    sort_by: str = "hotness",
    page_size: int = DEFAULT_PAGE_SIZE,
    fetch_comments: bool = True,
):
    console = Console()
    db_path = default_db_path()

    console.print(f"[bold]Fetching discussions for competition:[/bold] {competition_id}")
    console.print(f"[dim]  sort-by={sort_by}, max-pages={max_pages}[/dim]")

    with DiscussionClient() as client:
        discussions = client.list_discussions(
            competition_id,
            sort_by=sort_by,
            page_size=page_size,
            max_pages=max_pages,
        )

        console.print(f"[green]Fetched {len(discussions)} discussions from Kaggle API[/green]")

        if not discussions:
            console.print("[yellow]No discussions found.[/yellow]")
            return

        now = datetime.now(UTC).isoformat()
        for d in discussions:
            d.last_fetched_at = now

        with DiscussionDatabase(db_path) as db:
            inserted, updated_count = db.upsert_discussions(discussions)

            if fetch_comments:
                console.print("[dim]Fetching comments per discussion...[/dim]")
                total_comments = _fetch_and_store_comments(
                    client,
                    db,
                    discussions,
                    competition_id,
                    console,
                )
                db.upsert_discussions(discussions)
                console.print(f"[green]Fetched {total_comments} comments across {len(discussions)} discussions[/green]")

            try:
                comp_info = client.get_competition_info(competition_id)
                db.upsert_competition_info(comp_info)
                if comp_info.evaluation_metric:
                    console.print(f"[dim]Competition metric: {comp_info.evaluation_metric}[/dim]")
            except Exception:
                pass

    console.print(
        f"[bold green]Done:[/bold green] {inserted} new, {updated_count} updated in {db_path}"
    )

def main():
    parser = argparse.ArgumentParser(description="Fetch Kaggle competition discussions")
    parser.add_argument("competition_id", help="Competition slug (e.g. 'birdclef-2026')")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_DISCUSSION_MAX_PAGES,
        help=f"Max pages to fetch (default: {DEFAULT_DISCUSSION_MAX_PAGES})",
    )
    parser.add_argument("--sort-by", default="hotness", choices=["hotness", "votes", "comments", "created", "updated"])
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--nofetch-comments", action="store_true", help="Skip fetching comments")
    args = parser.parse_args()
    ingest(
        args.competition_id,
        max_pages=args.max_pages,
        sort_by=args.sort_by,
        page_size=args.page_size,
        fetch_comments=not args.nofetch_comments,
    )

if __name__ == "__main__":
    main()
