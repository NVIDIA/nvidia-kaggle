#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser import read_page_text
from runtime import competition_slug
from support.constants import DEFAULT_BROWSER_TIMEOUT_MS


def parse_slug(slug_or_url: str) -> str:
    """Extract competition slug from a Kaggle URL or return as-is."""
    return competition_slug(slug_or_url)


def get_competition_overview(competition_slug: str) -> str:
    url = f"https://www.kaggle.com/competitions/{competition_slug}"
    text = read_page_text(url, timeout=DEFAULT_BROWSER_TIMEOUT_MS)

    # The page body contains "Overview" multiple times: once in the nav tab
    # bar and once as the section heading before the real content.  Skip past
    # the nav occurrence by finding the *second* "Overview" and starting there.
    start_marker = "Overview"
    end_marker = "Citation"  # usually the last section on the overview page

    first = text.find(start_marker)
    if first == -1:
        return "Overview section not found."
    second = text.find(start_marker, first + len(start_marker))
    start = second if second != -1 else first

    end = text.find(end_marker, start)

    description = text[start + len(start_marker): end + len(end_marker) if end != -1 else None]

    # Drop residual nav-bar items that may slip through
    nav_items = {"Data", "Code", "Models", "Discussion", "Leaderboard", "Rules", "Overview"}
    lines = [line.strip() for line in description.splitlines()]
    lines = [line for line in lines if line and line not in nav_items]
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Kaggle competition overview")
    parser.add_argument("competition", help="Competition slug or URL")
    args = parser.parse_args()

    try:
        slug = parse_slug(args.competition)
        print(get_competition_overview(slug))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
