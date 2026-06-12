#!/usr/bin/env python3
"""Generate a Competition Intel Briefing for a Kaggle competition.

Chains the existing skill scripts into a single polished markdown report.
The briefing is *token-optional*: it degrades gracefully per tier.

Tiers
-----
No-token (always runs, public web scraping):
  - Competition overview         (fetch_competition_info)
  - Dataset description          (fetch_dataset_info)
  - Top solution writeups        (fetch_leaderboard_writeups + fetch_writeup)

Auth (requires KAGGLE_API_TOKEN, rendered as clear placeholders otherwise):
  - Top public kernels + scores  (fetch_top_kernel_scores)
  - Top discussions              (discussion_ingest + discussion_query)

Usage:
    python generate_briefing.py <competition-slug-or-url> [options]

Options:
    --output PATH         Markdown output path (default: {slug}_briefing.md)
    --top-writeups N      Number of top writeups to fetch (default: 3, 0 to skip)
    --top-kernels N       Number of top kernels to list when token present (default: 10)
    --top-discussions N   Number of top discussions to list when token present (default: 10)
    --skip-writeups       Skip the writeup tier entirely (faster)
    --print               Also print the markdown to stdout
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from runtime import competition_slug, load_project_env

# Load KAGGLE_API_TOKEN (and any other vars) from the project .env so the auth
# tiers light up without the caller having to export the token by hand. The
# skill already depends on python-dotenv; load_project_env uses it.
load_project_env()

# Default counts for the briefing tiers.
DEFAULT_TOP_WRITEUPS = 3
DEFAULT_TOP_KERNELS = 10
DEFAULT_TOP_DISCUSSIONS = 10
# Cap how many discussion list pages we ingest for the briefing so a demo run
# stays fast even on large competitions.
BRIEFING_DISCUSSION_MAX_PAGES = 2


def _has_token() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN"))


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


_KAGGLE_BASE = "https://www.kaggle.com"

# Trailing writeup sections we drop so they don't bloat the briefing or get
# mistaken for top-level sections. Matched case-insensitively on the heading.
_WRITEUP_DROP_HEADINGS = re.compile(
    r"^#{1,6}\s+(\d+\s+)?(comments?|appreciation|citation)\b",
    re.IGNORECASE,
)

# A non-rendering thumbnail-image header Kaggle emits at the top of writeups:
#   [![](/competitions/<id>/images/thumbnail)\n<Competition Name>](/competitions/<slug>)
# We replace it with a plain competition link (per the writeups workflow).
_THUMBNAIL_LINK = re.compile(
    r"\[!\[\]\(/competitions/\d+/images/thumbnail\)\s*\n?\s*"
    r"([^\]]+)\]\((/competitions/[^)]+)\)",
)

# Markdown link/image targets that are Kaggle-relative (start with a single
# "/"), which don't resolve once the briefing is read outside kaggle.com.
_RELATIVE_TARGET = re.compile(r"(\]\()(/[^)]+)(\))")


def _absolutize_links(text: str) -> str:
    """Rewrite Kaggle-relative markdown link/image targets to absolute URLs."""
    return _RELATIVE_TARGET.sub(lambda m: f"{m.group(1)}{_KAGGLE_BASE}{m.group(2)}{m.group(3)}", text)


def _embed_writeup(body: str) -> str:
    """Prepare a fetched writeup body for embedding under a '### Rank N' heading.

    - Replaces the non-rendering thumbnail-image header with a competition link.
    - Drops trailing Comments/Appreciation/Citation sections (we don't surface
      comments per the writeups workflow), keeping the briefing focused.
    - Demotes the writeup's own markdown headings two levels so they nest under
      the rank heading instead of colliding with the briefing's top-level
      sections (a writeup '## Results' would otherwise read as a doc section).
    - Absolutizes Kaggle-relative links so they work outside kaggle.com.
    """
    # Replace the thumbnail-image header block first (it spans two lines).
    body = _THUMBNAIL_LINK.sub(
        lambda m: f"[{m.group(1).strip()}]({_KAGGLE_BASE}{m.group(2)})",
        body,
    )

    out_lines: list[str] = []
    for line in body.splitlines():
        if _WRITEUP_DROP_HEADINGS.match(line.strip()):
            # Stop at the first trailing/meta section.
            break
        m = re.match(r"^(#{1,6})(\s+.*)$", line)
        if m:
            demoted = "#" * min(6, len(m.group(1)) + 2)
            out_lines.append(f"{demoted}{m.group(2)}")
        else:
            out_lines.append(line)
    return _absolutize_links("\n".join(out_lines).strip())


def _token_placeholder(tier_name: str) -> str:
    return (
        f"> _Requires `KAGGLE_API_TOKEN`._ This section was skipped because no "
        f"Kaggle API token is set in the environment. Set `KAGGLE_API_TOKEN` and "
        f"re-run to populate **{tier_name}**.\n"
    )


# ---------------------------------------------------------------------------
# No-token tiers
# ---------------------------------------------------------------------------

def build_overview(slug: str) -> str:
    from fetch_competition_info import get_competition_overview

    try:
        text = get_competition_overview(slug)
        if not text or "Overview section not found" in text:
            return "_Overview could not be extracted from the competition page._"
        return text.strip()
    except Exception as exc:
        return f"_Failed to fetch overview: {exc}_"


def build_dataset(slug: str) -> str:
    from fetch_dataset_info import get_dataset_description

    try:
        text = get_dataset_description(slug)
        if not text:
            return "_Dataset description could not be extracted._"
        return text.strip()
    except Exception as exc:
        return f"_Failed to fetch dataset description: {exc}_"


def build_writeups(slug: str, top_k: int) -> str:
    """Discover top-k leaderboard writeups and fetch each as markdown."""
    if top_k <= 0:
        return "_Writeup tier skipped (--top-writeups 0)._"

    from fetch_leaderboard_writeups import fetch_writeup_links
    from fetch_writeup import fetch_writeup as fetch_writeup_markdown

    leaderboard_url = f"https://www.kaggle.com/competitions/{slug}/leaderboard"
    try:
        links = fetch_writeup_links(leaderboard_url)
    except Exception as exc:
        return f"_Failed to discover writeups: {exc}_"

    if not links:
        return (
            "_No public solution writeups were found on the leaderboard for this "
            "competition (tutorial/active competitions often have none)._"
        )

    parts: list[str] = []
    parts.append(f"Discovered **{len(links)}** writeup link(s); fetching the top {min(top_k, len(links))}.\n")

    # Summary table of all discovered links.
    parts.append("| Rank | Team | Link |")
    parts.append("|---|---|---|")
    for item in links:
        rank = item.get("rank") or "?"
        team = (item.get("team") or "—").replace("|", "\\|")
        url = item.get("writeup_url", "")
        parts.append(f"| {rank} | {team} | [writeup]({url}) |")
    parts.append("")

    # Fetch the top-k writeup bodies.
    for item in links[:top_k]:
        rank = item.get("rank") or "?"
        team = item.get("team") or "Unknown team"
        url = item.get("writeup_url", "")
        parts.append(f"### Rank {rank} — {team}")
        parts.append(f"Source: {url}\n")
        try:
            body = fetch_writeup_markdown(url)
            if body and "We can't find that page" not in body:
                parts.append(_embed_writeup(body))
            else:
                parts.append("_Writeup content unavailable (page missing or login-gated)._")
        except Exception as exc:
            parts.append(f"_Failed to fetch this writeup: {exc}_")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Auth tiers
# ---------------------------------------------------------------------------

def build_kernels(slug: str, top_k: int) -> str:
    if not _has_token():
        return _token_placeholder("Top Public Kernels")

    # Strategy: list top kernels via the REST API in a single paginated call
    # (fast, reliable, returns votes), then enrich ONLY the top_k with public
    # scores via the SDK search — best-effort. The SDK search is one call per
    # kernel and Kaggle rate-limits it aggressively (429), so we never enrich
    # the whole list and we swallow score-lookup failures instead of letting
    # them sink the whole tier.
    try:
        from kernels.kaggle_client import KaggleKernelClient

        client = KaggleKernelClient()
        kernels = client.list_kernels(
            competition=slug,
            sort_by="voteCount",
            page_size=max(top_k, 20),
            max_pages=1,
        )
    except Exception as exc:
        return f"_Failed to list kernels: {exc}_"

    if not kernels:
        return "_No public kernels found for this competition._"

    top = kernels[:top_k]

    # Best-effort score enrichment for just the displayed kernels.
    scores: dict[str, object] = {}
    score_note = ""
    try:
        from kernels.kaggle_search import KaggleKernelSearchClient

        search = KaggleKernelSearchClient()
        for k in top:
            try:
                scores[k.ref] = search.get_kernel_score(k.ref)
            except Exception:
                # Likely a 429 / transient; stop enriching and label the gap.
                score_note = (
                    "\n\n_Note: public-score lookup was rate-limited; scores "
                    "shown where available, votes always shown._"
                )
                break
    except Exception:
        score_note = "\n\n_Note: public-score lookup unavailable; showing votes only._"

    lines = ["| # | Kernel | Score | Votes |", "|---|---|---|---|"]
    for i, k in enumerate(top, start=1):
        ref = (k.ref or "").replace("|", "\\|")
        title = (k.title or "").replace("|", "\\|")
        sc = scores.get(k.ref)
        score = "" if sc is None or getattr(sc, "score", None) is None else sc.score
        votes = "" if k.total_votes is None else k.total_votes
        label = f"{title} (`{ref}`)" if title else f"`{ref}`"
        url = f"https://www.kaggle.com/code/{ref}" if ref else ""
        link = f"[{label}]({url})" if url else label
        lines.append(f"| {i} | {link} | {score} | {votes} |")
    return "\n".join(lines) + score_note


def build_discussions(slug: str, top_k: int) -> str:
    if not _has_token():
        return _token_placeholder("Top Discussions")

    try:
        # Ensure the local cache has data, then query it.
        from discussion_ingest import ingest
        from discussions.database import DiscussionDatabase
        from discussions.paths import default_db_path

        ingest(
            slug,
            max_pages=BRIEFING_DISCUSSION_MAX_PAGES,
            sort_by="votes",
            fetch_comments=False,
        )

        with DiscussionDatabase(default_db_path()) as db:
            discussions = db.query_discussions(
                slug, sort_by="votes", sort_order="DESC", limit=top_k
            )
    except Exception as exc:
        return f"_Failed to fetch discussions: {exc}_"

    if not discussions:
        return "_No discussions found for this competition._"

    lines = ["| Votes | Title | Author |", "|---|---|---|"]
    for d in discussions:
        title = (d.title or "").replace("|", "\\|")
        author = (d.author or "—").replace("|", "\\|")
        lines.append(f"| {d.votes} | {title} | {author} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def generate(
    slug: str,
    *,
    top_writeups: int,
    top_kernels: int,
    top_discussions: int,
    skip_writeups: bool,
) -> str:
    comp_url = f"https://www.kaggle.com/competitions/{slug}"
    token = _has_token()

    parts: list[str] = []
    parts.append(f"# Competition Intel Briefing — {slug}\n")
    parts.append(
        f"Competition: [{comp_url}]({comp_url})  \n"
        f"Generated by the **nvidia-kaggle-skill** briefing workflow.  \n"
        f"Auth tier: {'**enabled** (KAGGLE_API_TOKEN present)' if token else '**disabled** (no token — public tiers only)'}\n"
    )

    # No-token tiers
    parts.append(_section("Overview", build_overview(slug)))
    parts.append(_section("Dataset", build_dataset(slug)))
    if skip_writeups:
        parts.append(_section("Top Solution Writeups", "_Skipped (--skip-writeups)._"))
    else:
        parts.append(_section("Top Solution Writeups", build_writeups(slug, top_writeups)))

    # Auth tiers
    parts.append(_section("Top Public Kernels", build_kernels(slug, top_kernels)))
    parts.append(_section("Top Discussions", build_discussions(slug, top_discussions)))

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Kaggle Competition Intel Briefing")
    parser.add_argument("competition", help="Competition slug or URL")
    parser.add_argument("--output", help="Markdown output path (default: {slug}_briefing.md)")
    parser.add_argument("--top-writeups", type=int, default=DEFAULT_TOP_WRITEUPS)
    parser.add_argument("--top-kernels", type=int, default=DEFAULT_TOP_KERNELS)
    parser.add_argument("--top-discussions", type=int, default=DEFAULT_TOP_DISCUSSIONS)
    parser.add_argument("--skip-writeups", action="store_true", help="Skip the writeup tier")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print markdown to stdout")
    args = parser.parse_args()

    slug = competition_slug(args.competition)
    output_path = Path(args.output) if args.output else Path(f"{slug}_briefing.md")

    try:
        markdown = generate(
            slug,
            top_writeups=args.top_writeups,
            top_kernels=args.top_kernels,
            top_discussions=args.top_discussions,
            skip_writeups=args.skip_writeups,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Briefing written to {output_path} ({len(markdown):,} chars)")
    if args.do_print:
        print("\n" + "=" * 60 + "\n")
        print(markdown)


if __name__ == "__main__":
    main()
