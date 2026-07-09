#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Search and filter kernels in the local database."""

import argparse
import json

from rich.console import Console
from rich.table import Table

from kernels.database import KernelDatabase
from kernels.paths import default_db_path
from runtime import load_project_env
from constants import (
    DATE_PREVIEW_CHARS,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_REF_COLUMN_WIDTH,
    DEFAULT_TITLE_COLUMN_WIDTH,
)

load_project_env()

def query(
    competition_id: str,
    search: str = None,
    min_votes: int = None,
    author: str = None,
    sort_by: str = "total_votes",
    sort_order: str = "DESC",
    limit: int = DEFAULT_QUERY_LIMIT,
    as_json: bool = False,
):
    console = Console()
    db_path = default_db_path()

    with KernelDatabase(db_path) as db:
        kernels = db.query_kernels(
            competition_id,
            search=search,
            min_votes=min_votes,
            author=author,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )

    if not kernels:
        console.print(f"[yellow]No kernels found for competition '{competition_id}'[/yellow]")
        return

    if as_json:
        out = [k.model_dump(mode="json") for k in kernels]
        print(json.dumps(out, indent=2, default=str))
        return

    table = Table(title=f"Kernels for {competition_id} ({len(kernels)} results)")
    table.add_column("Ref", style="cyan", max_width=DEFAULT_REF_COLUMN_WIDTH)
    table.add_column("Title", max_width=DEFAULT_TITLE_COLUMN_WIDTH)
    table.add_column("Author", style="green")
    table.add_column("Votes", justify="right", style="yellow")
    table.add_column("Last Run", style="dim")

    for k in kernels:
        last_run = str(k.last_run_time)[:DATE_PREVIEW_CHARS] if k.last_run_time else "-"
        table.add_row(k.ref, k.title, k.author, str(k.total_votes), last_run)

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Search and filter kernels")
    parser.add_argument("competition_id", help="Competition slug")
    parser.add_argument("--search", help="Free-text search in title, author, and kernel ref")
    parser.add_argument("--min-votes", type=int, help="Minimum vote count")
    parser.add_argument("--author", help="Filter by author")
    parser.add_argument("--sort-by", default="total_votes", choices=["total_votes", "last_run_time", "title", "author"])
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
