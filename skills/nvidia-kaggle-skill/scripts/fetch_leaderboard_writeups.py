#!/usr/bin/env python3
"""Fetch writeup URLs from a Kaggle competition leaderboard page."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser import evaluate_page
from support.constants import LONG_BROWSER_TIMEOUT_MS


def fetch_writeup_links(leaderboard_url: str) -> list[dict]:
    """Return list of {rank, team, writeup_url} from the leaderboard."""
    results = evaluate_page(
        leaderboard_url,
        r"""
        () => {
            const base = 'https://www.kaggle.com';
            const writeupLinks = [];

            // Find all writeup anchor tags (their href contains /writeups/)
            document.querySelectorAll('a[href*="/writeups/"]').forEach(a => {
                const href = a.getAttribute('href') || '';
                if (!href.includes('/writeups/')) return;

                // Walk up the DOM to find the leaderboard row container.
                // Kaggle renders rows as divs with role or class patterns.
                let row = a.closest('tr, [class*="row"], [class*="Row"], [class*="list-item"], [class*="ListItem"]');
                if (!row) {
                    // Fallback: walk up several levels
                    let el = a;
                    for (let i = 0; i < 8 && el.parentElement; i++) {
                        el = el.parentElement;
                        const text = el.innerText || '';
                        // A leaderboard row typically contains a rank number at the start
                        if (/^\s*\d+\s/.test(text) && text.length < 1000) {
                            row = el;
                            break;
                        }
                    }
                }

                let rank = '';
                let team = '';

                if (row) {
                    const text = row.innerText || '';
                    // Rank: first standalone number in the row text
                    const rankMatch = text.match(/^\s*(\d+)/);
                    if (rankMatch) rank = rankMatch[1];

                    // Team name: extract from row text by splitting lines.
                    // The team name line is typically after the rank and change indicator.
                    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
                    for (const line of lines) {
                        // Skip rank numbers, arrows, score-like floats, date-like strings,
                        // navigation labels, and the "description" icon text
                        if (/^\d+$/.test(line)) continue;
                        if (/^(arrow_drop_|change_history|—|description)/.test(line)) continue;
                        if (/^\d+\.\d{4}/.test(line)) continue;
                        if (/^\d+[dm]$/.test(line)) continue;
                        // This should be the team name
                        team = line;
                        break;
                    }
                }

                const fullUrl = href.startsWith('http') ? href : base + href;
                writeupLinks.push({rank, team, writeup_url: fullUrl});
            });

            return writeupLinks;
        }
        """,
        timeout=LONG_BROWSER_TIMEOUT_MS,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch writeup URLs from a Kaggle leaderboard page")
    parser.add_argument("leaderboard_url", help="Kaggle competition leaderboard URL")
    args = parser.parse_args()

    if not re.match(r"^https?://", args.leaderboard_url):
        parser.error("leaderboard_url must be an absolute http(s) Kaggle leaderboard URL")

    try:
        links = fetch_writeup_links(args.leaderboard_url)
        print(json.dumps(links, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
