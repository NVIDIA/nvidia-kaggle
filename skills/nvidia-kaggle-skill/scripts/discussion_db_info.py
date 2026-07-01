#!/usr/bin/env python3
"""Show discussion database statistics and list available competitions."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from db_info import show_competition_detail
from discussions.database import DiscussionDatabase
from discussions.paths import default_db_path
from runtime import load_project_env
from constants import DATE_PREVIEW_CHARS

load_project_env()

def _show_overview(console: Console, db: DiscussionDatabase) -> None:
    competitions = db.list_competitions()

    if not competitions:
        console.print("[yellow]Database is empty. Run discussion_ingest.py to populate it.[/yellow]")
        return

    table = Table(title="Competitions in Database")
    table.add_column("Competition ID", style="cyan")
    table.add_column("Discussions", justify="right", style="green")
    table.add_column("First Ingest", style="dim")
    table.add_column("Last Ingest", style="dim")

    for comp in competitions:
        table.add_row(
            comp["competition_id"],
            str(comp["cnt"]),
            str(comp["first_ingest"])[:DATE_PREVIEW_CHARS],
            str(comp["last_ingest"])[:DATE_PREVIEW_CHARS],
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(competitions)} competitions, {sum(c['cnt'] for c in competitions)} discussions")

def db_info(competition_id: str = None):
    console = Console()
    db_path = default_db_path()

    if not Path(db_path).exists():
        console.print("[yellow]Database not found. Run discussion_ingest.py first.[/yellow]")
        return

    with DiscussionDatabase(db_path) as db:
        if competition_id:
            show_competition_detail(
                console,
                db,
                competition_id,
                count_attr="discussion_count",
                entity_plural="discussions",
                author_count_label="Discussions",
            )
        else:
            _show_overview(console, db)

def main():
    parser = argparse.ArgumentParser(description="Show discussion database statistics")
    parser.add_argument("competition_id", nargs="?", default=None, help="Show details for a specific competition")
    args = parser.parse_args()
    db_info(args.competition_id)

if __name__ == "__main__":
    main()
