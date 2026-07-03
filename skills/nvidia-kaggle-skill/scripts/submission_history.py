#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Show locally logged Kaggle submission attempts.

Reads the append-only log written by ``submit_kernel.py`` so a workflow can
see what previous runs already submitted — and how those submissions scored —
before spending another daily submission slot. This is local bookkeeping only;
it makes no Kaggle API calls and needs no credentials.
"""

import argparse
import json

from rich.console import Console
from rich.table import Table

from constants import DATE_PREVIEW_CHARS, DEFAULT_QUERY_LIMIT, DEFAULT_TITLE_COLUMN_WIDTH
from runtime import competition_slug, kernel_ref, load_project_env
from submission_log import default_log_path, read_records, submission_attempts

load_project_env()


def history(
    competition: str = None,
    kernel: str = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    as_json: bool = False,
):
    console = Console()
    log_path = default_log_path()
    attempts = submission_attempts(
        read_records(log_path),
        competition=competition,
        kernel=kernel,
        limit=limit,
    )

    if not attempts:
        scope = f" for '{competition}'" if competition else ""
        console.print(f"[yellow]No logged submissions{scope} in {log_path}[/yellow]")
        return

    if as_json:
        print(json.dumps(attempts, indent=2, default=str))
        return

    title_scope = f" for {competition}" if competition else ""
    table = Table(title=f"Logged submissions{title_scope} ({len(attempts)} results)")
    table.add_column("Logged (UTC)", style="dim")
    table.add_column("Kernel", style="cyan")
    table.add_column("Ver", justify="right")
    table.add_column("Competition", style="green")
    table.add_column("Message", max_width=DEFAULT_TITLE_COLUMN_WIDTH)
    table.add_column("Submit", style="yellow")
    table.add_column("Eval")
    table.add_column("Score", justify="right")

    # Newest first so the row that matters for a retry decision is on top.
    for attempt in reversed(attempts):
        logged = str(attempt.get("logged_at") or "-")
        logged = logged[: DATE_PREVIEW_CHARS + 9]  # date + HH:MM:SS
        version = attempt.get("version")
        table.add_row(
            logged,
            str(attempt.get("kernel") or "-"),
            str(version) if version is not None else "-",
            str(attempt.get("competition") or "-"),
            str(attempt.get("message") or "-"),
            "accepted" if attempt.get("accepted") else "failed",
            str(attempt.get("eval_status") or "-"),
            str(attempt.get("public_score") or "-"),
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Show locally logged Kaggle submission attempts")
    parser.add_argument("competition", nargs="?", help="Competition slug or URL to filter by")
    parser.add_argument("--kernel", help="Kernel ref (owner/slug or Kaggle code URL) to filter by")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--as-json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    history(
        competition=competition_slug(args.competition) if args.competition else None,
        kernel=kernel_ref(args.kernel) if args.kernel else None,
        limit=args.limit,
        as_json=args.as_json,
    )


if __name__ == "__main__":
    main()
