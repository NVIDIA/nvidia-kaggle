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


def get_dataset_description(competition_slug: str) -> str:
    url = f"https://www.kaggle.com/competitions/{competition_slug}/data"
    text = read_page_text(url, timeout=DEFAULT_BROWSER_TIMEOUT_MS)

    # Start at "Dataset Description", end at "Data Explorer" (the file browser UI)
    start_marker = "Dataset Description"
    end_marker = "Data Explorer"

    start = text.find(start_marker)
    end = text.find(end_marker, start)

    if start == -1:
        return "Dataset Description section not found."

    description = text[start + len(start_marker): end if end != -1 else None]

    # Clean blank lines
    lines = [line.strip() for line in description.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Kaggle competition dataset description")
    parser.add_argument("competition", help="Competition slug or URL")
    args = parser.parse_args()

    try:
        slug = parse_slug(args.competition)
        print(get_dataset_description(slug))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
