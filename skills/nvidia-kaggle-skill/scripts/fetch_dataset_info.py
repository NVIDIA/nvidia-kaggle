#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch a Kaggle competition dataset description through the Kaggle API (no browser)."""

import argparse
import sys

from runtime import competition_pages, competition_slug, html_to_markdown

def parse_slug(slug_or_url: str) -> str:
    """Extract competition slug from a Kaggle URL or return as-is."""
    return competition_slug(slug_or_url)

def get_dataset_description(slug: str) -> str:
    """Return the dataset description as markdown via the Kaggle API."""
    pages = competition_pages(slug)

    description = html_to_markdown(pages.get("data-description", ""))
    if description:
        return description

    return "Dataset Description section not found."

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
