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


def _fetch_submissions(slug: str, *, page_size: int = 200) -> list[tuple[str, datetime]] | None:
    """Return ``[(submitter, utc_datetime), ...]`` for a competition via the SDK.

    Uses ``competition_submissions``, whose ``ApiSubmission`` objects carry
    ``submitted_by`` — the CLI CSV has no submitter column. Visibility is limited
    to what the authenticated account can see (own submissions, and teammates'
    where the API returns them). Returns None if the list cannot be fetched.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        subs = api.competition_submissions(slug, page_size=page_size)
    except Exception as exc:  # noqa: BLE001 — best-effort, like the rest of this module
        print(f"[quota] could not fetch submissions for {slug}: {exc}", file=sys.stderr)
        return None

    out: list[tuple[str, datetime]] = []
    for sub in subs or []:
        date_val = getattr(sub, "date", None)
        # ApiSubmission.date may be a datetime or a string depending on SDK version.
        dt = date_val if isinstance(date_val, datetime) else _parse_submission_date(str(date_val or ""))
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        user = getattr(sub, "submitted_by", None) or getattr(sub, "submitted_by_ref", None) or "unknown"
        out.append((user, dt.astimezone(timezone.utc)))
    return out


def submissions_by_user_today(
    slug: str, *, now: datetime | None = None, page_size: int = 200
) -> dict[str, int] | None:
    """Return ``{username: count}`` of today's (since 00:00 UTC) submissions."""
    subs = _fetch_submissions(slug, page_size=page_size)
    if subs is None:
        return None
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    counts: dict[str, int] = {}
    for user, dt in subs:
        if dt >= midnight:
            counts[user] = counts.get(user, 0) + 1
    return counts


def submissions_overall(slug: str, *, page_size: int = 200) -> dict[str, int] | None:
    """Return ``{username: count}`` across all visible submissions (all dates)."""
    subs = _fetch_submissions(slug, page_size=page_size)
    if subs is None:
        return None
    counts: dict[str, int] = {}
    for user, _dt in subs:
        counts[user] = counts.get(user, 0) + 1
    return counts


def submissions_by_day(slug: str, *, page_size: int = 200) -> dict[str, dict[str, int]] | None:
    """Return ``{YYYY-MM-DD (UTC): {username: count}}`` across all visible submissions."""
    subs = _fetch_submissions(slug, page_size=page_size)
    if subs is None:
        return None
    by_day: dict[str, dict[str, int]] = {}
    for user, dt in subs:
        day = dt.date().isoformat()
        day_counts = by_day.setdefault(day, {})
        day_counts[user] = day_counts.get(user, 0) + 1
    # Sort newest day first for readable output.
    return {day: by_day[day] for day in sorted(by_day, reverse=True)}


def submission_quota(
    slug: str,
    *,
    limit_fallback: int = 5,
    by_user: bool = False,
    by_day: bool = False,
    overall: bool = False,
    now: datetime | None = None,
) -> dict:
    """Return today's submission-quota state for a competition.

    Keys: ``competition``, ``limit``, ``limit_source`` ("sdk"|"fallback"),
    ``used`` (None if unknown), ``remaining`` (None if used unknown),
    ``exhausted`` (True iff remaining is known and <= 0).

    Optional breakdowns (all SDK-based, attributing the team-wide quota):
      - ``by_user`` set -> add ``by_user`` ({username: count}, today) and use
        its total as ``used``.
      - ``by_day`` set  -> add ``by_day`` ({date: {username: count}}, all dates).
      - ``overall`` set -> add ``overall`` ({username: count}, all dates).
    """
    sdk_limit = competition_daily_submission_limit(slug)
    limit = sdk_limit if sdk_limit is not None else limit_fallback

    per_user: dict[str, int] | None = None
    if by_user:
        per_user = submissions_by_user_today(slug, now=now)
        used = None if per_user is None else sum(per_user.values())
    else:
        used = submissions_used_today(slug, now=now)

    remaining = None if used is None else max(limit - used, 0)
    state = {
        "competition": slug,
        "limit": limit,
        "limit_source": "sdk" if sdk_limit is not None else "fallback",
        "used": used,
        "remaining": remaining,
        "exhausted": remaining is not None and remaining <= 0,
    }
    if by_user:
        state["by_user"] = per_user
    if by_day:
        state["by_day"] = submissions_by_day(slug)
    if overall:
        state["overall"] = submissions_overall(slug)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Report today's Kaggle submission quota for a competition")
    parser.add_argument("competition", help="Competition slug or URL")
    parser.add_argument(
        "--limit-fallback",
        type=int,
        default=5,
        help="Daily limit to assume when the SDK does not report one (default 5)",
    )
    parser.add_argument(
        "--by-user",
        action="store_true",
        help="Break today's submissions down by submitter (username: count) via the SDK. "
        "Useful when teaming up; note the daily limit is team-wide, not per-user.",
    )
    parser.add_argument(
        "--by-day",
        action="store_true",
        help="Add a per-day breakdown ({date: {username: count}}) across all visible submissions.",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Add an all-time breakdown ({username: count}) across all visible submissions.",
    )
    parser.add_argument("--as-json", action="store_true", help="Print the quota state as JSON")
    args = parser.parse_args()

    load_project_env()
    slug = competition_slug(args.competition)
    state = submission_quota(
        slug,
        limit_fallback=args.limit_fallback,
        by_user=args.by_user,
        by_day=args.by_day,
        overall=args.overall,
    )

    if args.as_json:
        print(json.dumps(state, indent=2))
        return

    used = "unknown" if state["used"] is None else state["used"]
    remaining = "unknown" if state["remaining"] is None else state["remaining"]
    print(f"Competition: {slug}")
    print(f"Daily limit: {state['limit']} ({state['limit_source']})")
    print(f"Used today (since 00:00 UTC): {used}")
    print(f"Remaining: {remaining}")
    if args.by_user:
        per_user = state.get("by_user")
        if per_user is None:
            print("By user: unknown (could not fetch submissions)")
        elif not per_user:
            print("By user: no submissions today")
        else:
            print("By user (team-wide limit; counts are attribution, not per-user caps):")
            for user, count in sorted(per_user.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {user}: {count}")
    if args.overall:
        overall = state.get("overall")
        if overall is None:
            print("Overall: unknown (could not fetch submissions)")
        else:
            total = sum(overall.values())
            print(f"Overall (all dates, {total} total):")
            for user, count in sorted(overall.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {user}: {count}")
    if args.by_day:
        by_day = state.get("by_day")
        if by_day is None:
            print("By day: unknown (could not fetch submissions)")
        elif not by_day:
            print("By day: no submissions")
        else:
            print("By day (UTC, newest first):")
            for day, users in by_day.items():
                parts = ", ".join(f"{u}: {c}" for u, c in sorted(users.items(), key=lambda kv: (-kv[1], kv[0])))
                print(f"  {day}: {parts}")
    if state["exhausted"]:
        print("Status: EXHAUSTED — no submissions left today.")
    elif state["remaining"] is None:
        print("Status: UNKNOWN — could not count today's submissions; rely on submit-time errors.")
    else:
        print("Status: OK — submissions available.")


if __name__ == "__main__":
    main()
