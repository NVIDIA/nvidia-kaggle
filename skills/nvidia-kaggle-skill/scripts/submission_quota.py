#!/usr/bin/env python3
"""Report today's Kaggle submission quota for a competition.

Kaggle limits daily submissions per competition and resets the count at 00:00
UTC. This reads the limit from the Kaggle SDK (``max_daily_submissions``) and
counts submissions made since UTC midnight via the Kaggle CLI, so a workflow can
check headroom BEFORE spending a submission slot.

This is a best-effort proactive guard; the submit-time quota/429 error remains
the authoritative backstop.
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import competition_slug, load_project_env


def competition_daily_submission_limit(slug: str) -> int | None:
    """Return the competition's ``max_daily_submissions`` via the SDK, or None."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        resp = api.competitions_list(search=slug)
        comps = getattr(resp, "competitions", None) or resp
        for comp in comps:
            ref = str(getattr(comp, "ref", ""))
            if ref.rstrip("/").endswith(slug):
                limit = getattr(comp, "max_daily_submissions", 0)
                return int(limit) if limit else None
        if comps:
            limit = getattr(comps[0], "max_daily_submissions", 0)
            return int(limit) if limit else None
    except Exception as exc:  # noqa: BLE001 — best-effort detection
        print(f"[quota] could not read max_daily_submissions for {slug}: {exc}", file=sys.stderr)
    return None


def _submission_rows(slug: str, page_size: int = 100) -> list[dict[str, str]] | None:
    """Return recent submission rows via the kaggle CLI, or None on failure."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", slug, "-v", "--page-size", str(page_size)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[quota] submissions fetch failed: {exc}", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"[quota] submissions fetch returned {result.returncode}", file=sys.stderr)
        return None
    # Some CLI versions print a deprecation notice before the CSV; start at the header.
    lines = result.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("fileName,")), None)
    if start is None:
        return None
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return list(reader)


def _parse_submission_date(value: str) -> datetime | None:
    """Parse a Kaggle submission ``date`` string to an aware UTC datetime."""
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def submissions_used_today(slug: str, *, now: datetime | None = None) -> int | None:
    """Count submissions made since today's 00:00 UTC, or None if unknown."""
    rows = _submission_rows(slug)
    if rows is None:
        return None
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    for row in rows:
        dt = _parse_submission_date(row.get("date", ""))
        if dt is not None and dt >= midnight:
            count += 1
    return count


def submission_quota(slug: str, *, limit_fallback: int = 5, now: datetime | None = None) -> dict:
    """Return today's submission-quota state for a competition.

    Keys: ``competition``, ``limit``, ``limit_source`` ("sdk"|"fallback"),
    ``used`` (None if unknown), ``remaining`` (None if used unknown),
    ``exhausted`` (True iff remaining is known and <= 0).
    """
    sdk_limit = competition_daily_submission_limit(slug)
    limit = sdk_limit if sdk_limit is not None else limit_fallback
    used = submissions_used_today(slug, now=now)
    remaining = None if used is None else max(limit - used, 0)
    return {
        "competition": slug,
        "limit": limit,
        "limit_source": "sdk" if sdk_limit is not None else "fallback",
        "used": used,
        "remaining": remaining,
        "exhausted": remaining is not None and remaining <= 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report today's Kaggle submission quota for a competition")
    parser.add_argument("competition", help="Competition slug or URL")
    parser.add_argument(
        "--limit-fallback",
        type=int,
        default=5,
        help="Daily limit to assume when the SDK does not report one (default 5)",
    )
    parser.add_argument("--as-json", action="store_true", help="Print the quota state as JSON")
    args = parser.parse_args()

    load_project_env()
    slug = competition_slug(args.competition)
    state = submission_quota(slug, limit_fallback=args.limit_fallback)

    if args.as_json:
        print(json.dumps(state, indent=2))
        return

    used = "unknown" if state["used"] is None else state["used"]
    remaining = "unknown" if state["remaining"] is None else state["remaining"]
    print(f"Competition: {slug}")
    print(f"Daily limit: {state['limit']} ({state['limit_source']})")
    print(f"Used today (since 00:00 UTC): {used}")
    print(f"Remaining: {remaining}")
    if state["exhausted"]:
        print("Status: EXHAUSTED — no submissions left today.")
    elif state["remaining"] is None:
        print("Status: UNKNOWN — could not count today's submissions; rely on submit-time errors.")
    else:
        print("Status: OK — submissions available.")


if __name__ == "__main__":
    main()
