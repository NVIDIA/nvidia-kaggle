#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch a Kaggle competition solution writeup as markdown via the Kaggle API.

Replaces the previous headless-browser scraper: Kaggle serves the writeup's
source markdown through its internal web service, so no browser or arbitrary
JavaScript execution is required.

A writeup URL has the form
``.../competitions/<competition>/writeups/<writeup-slug>``. The writeup is a
forum topic; its numeric topic id is found by matching the writeup slug against
the competition leaderboard's per-team ``solutionWriteUpUrl``. The topic's
``writeUp.message.rawMarkdown`` is the source markdown.
"""

import argparse
import re
import sys
from pathlib import Path

from runtime import kaggle_web_service

WRITEUP_URL_RE = re.compile(r"/competitions/(?P<competition>[^/]+)/writeups/(?P<slug>[^/?#]+)")

def _resolve_topic_id(client, competition: str, writeup_slug: str) -> int:
    """Map a writeup slug to its forum topic id via the competition leaderboard."""
    competition_info = client.post(
        "competitions.CompetitionService/GetCompetition",
        {"competitionName": competition},
    )
    competition_id = competition_info.get("id") or (
        competition_info.get("competition") or {}
    ).get("id")
    if not competition_id:
        raise RuntimeError(f"Could not resolve competition id for '{competition}'.")

    board = client.post(
        "competitions.LeaderboardService/GetLeaderboard",
        {"competitionId": competition_id},
    )
    for team in board.get("teams") or []:
        url = team.get("solutionWriteUpUrl") or ""
        topic_id = team.get("writeUpForumTopicId")
        if topic_id and url.rstrip("/").endswith("/" + writeup_slug):
            return topic_id

    raise RuntimeError(
        f"Could not find writeup '{writeup_slug}' in the leaderboard for "
        f"'{competition}'. The writeup may be unlisted or removed."
    )

def fetch_writeup(url: str) -> str:
    """Return the writeup's source markdown for a Kaggle writeup URL."""
    match = WRITEUP_URL_RE.search(url)
    if not match:
        raise RuntimeError(
            "URL must be a Kaggle writeup URL of the form "
            "https://www.kaggle.com/competitions/<competition>/writeups/<slug>"
        )

    client = kaggle_web_service()
    topic_id = _resolve_topic_id(client, match.group("competition"), match.group("slug"))

    topic = client.post(
        "discussions.DiscussionsService/GetForumTopicById",
        {"forumTopicId": topic_id, "includeComments": False},
    ).get("forumTopic", {})

    write_up = topic.get("writeUp") or {}
    message = write_up.get("message") or {}
    markdown = message.get("rawMarkdown") or message.get("content") or ""
    if not markdown.strip():
        raise RuntimeError(f"Writeup topic {topic_id} returned no content.")

    title = write_up.get("title") or topic.get("name") or ""
    body = markdown.strip()
    if title and not body.lstrip().startswith("#"):
        body = f"# {title}\n\n{body}"

    return re.sub(r"\n{3,}", "\n\n", body).strip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Kaggle writeup as markdown")
    parser.add_argument("url", help="Kaggle writeup URL")
    parser.add_argument("output_path", nargs="?", help="Optional markdown output path")
    args = parser.parse_args()

    if not re.match(r"^https?://", args.url):
        parser.error("url must be an absolute http(s) Kaggle writeup URL")

    try:
        content = fetch_writeup(args.url)
        if args.output_path:
            out_path = Path(args.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"Saved to {out_path} ({len(content)} chars)", file=sys.stderr)
        else:
            sys.stdout.buffer.write(content.encode("utf-8"))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
