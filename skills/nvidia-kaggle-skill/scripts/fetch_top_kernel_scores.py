#!/usr/bin/env python3
"""Fetch public kernel scores for a Kaggle competition through the Kaggle SDK.

Usage: python fetch_top_kernel_scores.py <competition-slug-or-url> [--sort ascending|descending|hotness]

Outputs CSV lines: ref,score
"""

import argparse
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from kernels.kaggle_client import KaggleKernelClient
from kernels.kaggle_search import KaggleKernelSearchClient, parse_competition_slug
from support.constants import DEFAULT_KERNEL_SCORE_PAGE_SIZE


def fetch_kernel_scores(
    competition_slug: str,
    sort: str = "descending",
    page_size: int = DEFAULT_KERNEL_SCORE_PAGE_SIZE,
    max_pages: int | None = None,
) -> list[dict]:
    """Fetch competition-scoped kernel refs, then enrich each with SDK score data."""
    competition_slug = parse_competition_slug(competition_slug)
    kernel_client = KaggleKernelClient()
    kernels = kernel_client.list_kernels(
        competition=competition_slug,
        sort_by="voteCount",
        page_size=page_size,
        max_pages=max_pages,
    )

    search_client = KaggleKernelSearchClient()
    score_map = search_client.get_kernel_scores(k.ref for k in kernels)

    rows = []
    for kernel in kernels:
        score = score_map.get(kernel.ref)
        rows.append(
            {
                "ref": kernel.ref,
                "title": score.title if score else kernel.title,
                "score": score.score if score else None,
                "votes": score.votes if score and score.votes is not None else kernel.total_votes,
                "has_linked_submission": score.has_linked_submission if score else False,
            }
        )

    if sort == "ascending":
        return sorted(
            rows,
            key=lambda item: (
                item["score"] is None,
                item["score"] if item["score"] is not None else float("inf"),
            ),
        )
    if sort == "descending":
        return sorted(
            rows,
            key=lambda item: (
                item["score"] is None,
                -(item["score"] if item["score"] is not None else float("-inf")),
            ),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public Kaggle kernel scores through the Kaggle SDK")
    parser.add_argument("competition", help="Competition slug or URL")
    parser.add_argument("--sort", default="descending", choices=["ascending", "descending", "hotness"])
    parser.add_argument("--page-size", type=int, default=DEFAULT_KERNEL_SCORE_PAGE_SIZE)
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N competition kernel-list pages")
    args = parser.parse_args()

    if args.page_size <= 0:
        parser.error("--page-size must be greater than 0")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be greater than 0 when provided")

    try:
        slug = parse_competition_slug(args.competition)
        kernels = fetch_kernel_scores(slug, args.sort, page_size=args.page_size, max_pages=args.max_pages)
        print("ref,score")
        for kernel in kernels:
            score = "" if kernel["score"] is None else kernel["score"]
            print(f"{kernel['ref']},{score}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
