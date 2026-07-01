#!/usr/bin/env python3
"""Display a Kaggle discussion thread with its comments."""

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from discussions.database import DiscussionDatabase
from discussions.paths import default_db_path
from runtime import load_project_env
from constants import DATE_PREVIEW_CHARS

load_project_env()

def read_discussion(discussion_id: int, competition_id: str = None):
    console = Console()
    db_path = default_db_path()

    with DiscussionDatabase(db_path) as db:
        if competition_id:
            discussion = db.get_discussion(competition_id, discussion_id)
        else:
            for comp in db.list_competitions():
                discussion = db.get_discussion(comp["competition_id"], discussion_id)
                if discussion:
                    competition_id = comp["competition_id"]
                    break
            else:
                discussion = None

        if not discussion:
            console.print(f"[red]Discussion {discussion_id} not found in database.[/red]")
            return

        comments = db.get_comments(competition_id, discussion_id)

    created = str(discussion.created_at)[:DATE_PREVIEW_CHARS] if discussion.created_at else "?"
    meta = f"Author: {discussion.author} | Votes: {discussion.votes} | Comments: {discussion.comment_count} | Created: {created}"
    if discussion.url:
        meta += f"\n{discussion.url}"

    console.print(Panel(
        meta,
        title=f"[bold]{discussion.title}[/bold]",
        title_align="left",
        border_style="cyan",
    ))

    if discussion.body_markdown:
        console.print()
        console.print(Markdown(discussion.body_markdown))

    if comments:
        console.print(f"\n[bold]── {len(comments)} Comments ──[/bold]\n")
        for i, c in enumerate(comments, 1):
            vote_str = f" [yellow]▲{c.votes}[/yellow]" if c.votes else ""
            date_str = f" [dim]{str(c.created_at)[:DATE_PREVIEW_CHARS]}[/dim]" if c.created_at else ""
            console.print(f"[bold green]{c.author}[/bold green]{vote_str}{date_str}")
            if c.body_markdown:
                console.print(Markdown(c.body_markdown))
            console.print()
    else:
        console.print("\n[dim]No comments.[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Display a discussion thread with comments")
    parser.add_argument("discussion_id", type=int, help="Numeric discussion ID")
    parser.add_argument("--competition-id", help="Competition slug (searches all if omitted)")
    args = parser.parse_args()
    read_discussion(args.discussion_id, competition_id=args.competition_id)

if __name__ == "__main__":
    main()
