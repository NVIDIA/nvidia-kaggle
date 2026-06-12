#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser import read_page_text
from runtime import competition_slug
from support.constants import DEFAULT_BROWSER_TIMEOUT_MS
from support.page_text import clean_page_lines


def parse_slug(slug_or_url: str) -> str:
    """Extract competition slug from a Kaggle URL or return as-is."""
    return competition_slug(slug_or_url)


# Boundaries (matched as whole, stripped lines — NOT substrings) that mark the
# end of the description region and the start of file-browser / sign-in chrome.
# Line-anchored matching means a license value like "Subject to Competition
# Rules" does NOT trigger the bare "Competition Rules" boundary. Bounding here
# keeps us from falling through to the end of the page and slurping chrome when
# "Data Explorer" hydrated late or the data is sign-in gated.
DATASET_START_MARKER = "Dataset Description"
DATASET_END_MARKERS = frozenset(
    {
        "Data Explorer",
        "Competition Rules",
        "Data Sources",
        "To see this data you need to agree to the competition rules.",
    }
)


def get_dataset_description(competition_slug: str) -> str:
    url = f"https://www.kaggle.com/competitions/{competition_slug}/data"
    # Wait for an end-of-region boundary to render before reading, so the
    # description region is reliably bounded at read time (root-cause fix for
    # slow-hydration leaks) rather than relying only on the line cleaner.
    text = read_page_text(
        url,
        timeout=DEFAULT_BROWSER_TIMEOUT_MS,
        wait_for_text=tuple(DATASET_END_MARKERS),
    )

    lines = text.splitlines()

    # Locate the description start line.
    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == DATASET_START_MARKER),
        None,
    )
    if start_idx is None:
        # Fall back to a substring search for resilience to wrapping.
        if DATASET_START_MARKER not in text:
            return "Dataset Description section not found."
        start_idx = next(
            (i for i, ln in enumerate(lines) if DATASET_START_MARKER in ln), 0
        )

    # Stop at the first end-marker line after the start (line-anchored, so the
    # license value "Subject to Competition Rules" is not treated as a stop).
    region = []
    for ln in lines[start_idx + 1:]:
        if ln.strip() in DATASET_END_MARKERS:
            break
        region.append(ln)

    # Strip page chrome (icon ligatures like "fullscreen"/"chevron_right", nav
    # tabs, timeline/section labels) via the shared cleaner so the dataset
    # fetcher and the competition fetcher can't drift apart.
    cleaned = clean_page_lines(region)
    return "\n\n".join(cleaned)


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
