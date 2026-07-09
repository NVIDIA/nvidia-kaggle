#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch kernels for a Kaggle competition and store them in the local database."""

import argparse

from rich.console import Console

from kernels.database import KernelDatabase
from kernels.kaggle_client import KaggleKernelClient
from kernels.paths import default_db_path
from runtime import load_project_env
from constants import DEFAULT_PAGE_SIZE

load_project_env()

def ingest(
    competition_id: str,
    max_pages: int = None,
    sort_by: str = "hotness",
    page_size: int = DEFAULT_PAGE_SIZE,
):
    console = Console()
    db_path = default_db_path()

    console.print(f"[bold]Fetching kernels for competition:[/bold] {competition_id}")
    console.print(f"  sort-by={sort_by}, max-pages={max_pages or 'all'}")

    client = KaggleKernelClient()
    kernels = client.list_kernels(
        competition=competition_id,
        sort_by=sort_by,
        page_size=page_size,
        max_pages=max_pages,
    )

    console.print(f"[green]Fetched {len(kernels)} kernels from Kaggle API[/green]")

    if not kernels:
        console.print("[yellow]No kernels found. Check the competition ID.[/yellow]")
        return

    with KernelDatabase(db_path) as db:
        inserted, updated = db.upsert_kernels(kernels)

        try:
            comp_info = client.get_competition_info(competition_id)
            db.upsert_competition_info(comp_info)
            if comp_info.evaluation_metric:
                console.print(f"[dim]Competition metric: {comp_info.evaluation_metric}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not fetch competition info: {e}[/yellow]")

    console.print(f"[bold green]Done:[/bold green] {inserted} new, {updated} updated in {db_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch Kaggle competition kernels")
    parser.add_argument("competition_id", help="Competition slug (e.g. 'titanic')")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: fetch all)")
    parser.add_argument("--sort-by", default="hotness", choices=["hotness", "dateCreated", "dateRun", "voteCount"])
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = parser.parse_args()
    ingest(
        args.competition_id,
        max_pages=args.max_pages,
        sort_by=args.sort_by,
        page_size=args.page_size,
    )

if __name__ == "__main__":
    main()
