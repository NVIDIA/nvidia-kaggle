#!/usr/bin/env python3
"""Fetch a Kaggle competition overview through the Kaggle API (no browser)."""

import argparse
import sys

from runtime import competition_pages, competition_slug, html_to_markdown

def parse_slug(slug_or_url: str) -> str:
    """Extract competition slug from a Kaggle URL or return as-is."""
    return competition_slug(slug_or_url)

def get_competition_overview(slug: str) -> str:
    """Return the competition overview as markdown.

    The Kaggle API exposes the overview as separate content pages; the
    'Description' and 'Evaluation' pages together make up the overview prose.
    """
    pages = competition_pages(slug)

    sections: list[str] = []
    description = html_to_markdown(pages.get("description", ""))
    if description:
        sections.append(description)

    evaluation = html_to_markdown(pages.get("evaluation", ""))
    if evaluation:
        sections.append("## Evaluation\n\n" + evaluation)

    if not sections:
        return "Overview section not found."

    return "\n\n".join(sections)

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
