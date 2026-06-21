#!/usr/bin/env python3
"""Fetch writeup URLs from a Kaggle competition leaderboard via the Kaggle API.

Replaces the previous headless-browser scraper: the Kaggle leaderboard web
service returns each team's solution writeup URL directly, so no browser or
arbitrary JavaScript execution is required.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import competition_slug, kaggle_web_service

WEB_BASE = "https://www.kaggle.com"


def fetch_writeup_links(leaderboard_url: str) -> list[dict]:
    """Return list of {rank, team, writeup_url} for teams that posted writeups.

    Accepts a competition slug, competition URL, or leaderboard URL. Joins the
    leaderboard's team list (which carries the writeup URL) with the private
    leaderboard ranking (falling back to the public leaderboard for live
    competitions that have no private standings yet).
    """
    slug = competition_slug(leaderboard_url)
    client = kaggle_web_service()

    competition = client.post(
        "competitions.CompetitionService/GetCompetition",
        {"competitionName": slug},
    )
    competition_id = competition.get("id") or (competition.get("competition") or {}).get("id")
    if not competition_id:
        raise RuntimeError(f"Could not resolve competition id for '{slug}'.")

    board = client.post(
        "competitions.LeaderboardService/GetLeaderboard",
        {"competitionId": competition_id},
    )

    teams = board.get("teams") or []
    private_board = board.get("privateLeaderboard") or []
    public_board = board.get("publicLeaderboard") or []

    # Map teamId -> rank. Prefer the private leaderboard's explicit rank; for
    # live competitions with no private board, fall back to public position.
    rank_by_team: dict[int, int] = {}
    for row in private_board:
        team_id = row.get("teamId")
        if team_id is not None and row.get("rank") is not None:
            rank_by_team[team_id] = row["rank"]
    if not rank_by_team:
        for position, row in enumerate(public_board, start=1):
            team_id = row.get("teamId")
            if team_id is not None:
                rank_by_team.setdefault(team_id, position)

    results: list[dict] = []
    for team in teams:
        writeup_path = team.get("solutionWriteUpUrl")
        if not writeup_path:
            continue
        url = writeup_path if writeup_path.startswith("http") else WEB_BASE + writeup_path
        rank = rank_by_team.get(team.get("teamId"))
        results.append(
            {
                "rank": str(rank) if rank is not None else "",
                "team": team.get("teamName", ""),
                "writeup_url": url,
            }
        )

    # Sort by numeric rank when available; rank-less entries go last.
    results.sort(key=lambda r: int(r["rank"]) if r["rank"].isdigit() else 10**9)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch writeup URLs from a Kaggle leaderboard"
    )
    parser.add_argument(
        "leaderboard_url",
        help="Competition slug, competition URL, or leaderboard URL",
    )
    args = parser.parse_args()

    try:
        links = fetch_writeup_links(args.leaderboard_url)
        print(json.dumps(links, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
