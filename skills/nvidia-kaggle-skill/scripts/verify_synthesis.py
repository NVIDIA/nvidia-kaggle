#!/usr/bin/env python3
"""Deterministic invariant checks for LLM-written synthesis output.

The synthesis section ("## Key Strategies") is written by a model, so it cannot
be asserted by exact match. Instead we check *invariants* that must hold for any
valid output — a deterministic test of a non-deterministic artifact.

Scope (locked with the team):
  HARD-FAIL invariants (every violation is a genuine defect):
    - the required section header is present
    - no hallucinated CLOSED-CLASS entities — kernel refs (owner/slug),
      competition slug, and verbatim-quoted team/author/title handles that do
      NOT appear in the briefing's entity allow-list
    - no scraped page chrome leaked through
  WARN invariants (style drift, not correctness):
    - bullet count outside the prompt's 3–6 range
    - word count over the prompt's max

Anti-hallucination is scoped to CLOSED-CLASS, HIGH-PRECISION entities only,
matched verbatim against the JSON allow-list — never fuzzy, never free prose.
Precision over recall: every reported hallucination is a real fabricated ref;
subtler issues fall to the human eyeball pass.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthesize import (
    SYNTHESIS_HEADER,
    SYNTHESIS_MAX_BULLETS,
    SYNTHESIS_MAX_WORDS,
    SYNTHESIS_MIN_BULLETS,
)
from support.page_text import (
    FILE_COUNT_RE,
    FILE_LISTING_RE,
    ICON_LIGATURE_RE,
    STANDALONE_ICON_WORDS,
)

# A kernel ref is owner/slug. To stay HIGH-PRECISION we only treat a token as a
# kernel ref when it appears in an unambiguous kernel context — never bare in
# prose (so "pair guesser/answerer" is NOT mistaken for a ref). Two contexts:
#   1. backtick-quoted:        `owner/slug`
#   2. in a Kaggle code URL:   kaggle.com/code/owner/slug
# Bare owner/slug tokens in free prose are out of scope (eyeball pass), matching
# the locked "closed-class, verbatim, precision-over-recall" contract.
_REF_BODY = r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*"
KERNEL_REF_BACKTICK_RE = re.compile(r"`(" + _REF_BODY + r")`")
KERNEL_REF_URL_RE = re.compile(r"kaggle\.com/code/(" + _REF_BODY + r")")


@dataclass
class VerificationResult:
    """Structured outcome of verifying one synthesis output."""

    ok: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Entity classes we could NOT verify because their allow-list was empty
    # (the tier wasn't gathered). Reported explicitly so an empty allow-list is
    # a visible coverage gap, never a silent pass. e.g. ["teams", "authors"].
    unverified: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # truthy iff no hard failures
        return self.ok


def _bullet_lines(text: str) -> list[str]:
    """Markdown bullet lines (-, *, +) ignoring indentation."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s[:2] in ("- ", "* ", "+ "):
            out.append(s)
    return out


# Verbatim-quoted spans: backtick-wrapped `like this` or double-quoted "like
# this". Display-name entity checks operate ONLY on these — never on free prose
# — so a sentence merely containing an author surname can't false-match.
_QUOTED_SPAN_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"")
# A filename-ish span (has an extension) is not a team/author handle; skip it to
# avoid warning on quoted output files like `submission.csv`.
_FILENAME_RE = re.compile(r"^[\w.\-/ ]+\.\w+$")


def _extract_quoted_spans(text: str) -> list[str]:
    """Return the stripped contents of every backtick/quote-wrapped span."""
    spans = []
    for m in _QUOTED_SPAN_RE.finditer(text):
        span = (m.group(1) or m.group(2) or "").strip()
        if span:
            spans.append(span)
    return spans


def _is_distinctive_handle_shape(s: str) -> bool:
    """A span shape that is unlikely to be a plain technical term:
    multi-word with a Titlecase word, OR a single token carrying a digit or an
    internal separator (`c-number`, `yukky_maru`, `team_42`). A lone lowercase
    word (`ensembling`, `lora`) is deliberately excluded — indistinguishable
    from a technical term, so never a candidate.
    """
    words = s.split()
    if len(words) >= 2 and any(w[:1].isupper() for w in words):
        return True
    if len(words) == 1 and (any(c.isdigit() for c in s) or any(c in "_-" for c in s)):
        return True
    return False


def _looks_like_handle(span: str) -> bool:
    """True if a quoted span plausibly IS a team/author handle (vs a technical
    term). Lone lowercase words are never candidates → no spurious warn on
    `ensembling`/`LoRA`; distinctive shapes (multi-word Titlecase, or a single
    token with a digit/separator) are.
    """
    return _is_distinctive_handle_shape(span)


# Kaggle URL path prefixes that are owner/slug-shaped but are NOT kernel refs
# (they're site paths). Excluded so they aren't mislabeled as hallucinated refs
# — a wrong competition path is caught separately by the slug check.
_NON_REF_PREFIXES = (
    "competitions",
    "code",
    "datasets",
    "models",
    "discussions",
    "www.kaggle.com",
    "kaggle.com",
)


def _extract_kernel_refs(text: str) -> set[str]:
    """Closed-class kernel refs from UNAMBIGUOUS contexts only.

    Only backtick-quoted refs and refs inside a kaggle.com/code/ URL are
    considered — bare owner/slug tokens in prose are deliberately ignored to
    keep the check high-precision (no false positives on phrases like
    "guesser/answerer"). Kaggle URL path prefixes are excluded as a safety net.
    """
    refs = set()
    for rx in (KERNEL_REF_BACKTICK_RE, KERNEL_REF_URL_RE):
        for m in rx.finditer(text):
            ref = m.group(1)
            if ref.count("/") == 1 and ref.split("/", 1)[0] not in _NON_REF_PREFIXES:
                refs.add(ref)
    return refs


def _verbatim_present(handle: str, text: str) -> bool:
    """True if `handle` appears verbatim (case-insensitive) as a substring."""
    return handle.lower() in text.lower()


def verify_synthesis(synthesis_text: str, briefing_json: dict[str, Any]) -> VerificationResult:
    """Check synthesis invariants against the briefing's closed-class entities.

    `briefing_json` is the dict emitted by `generate_briefing.py --json`; its
    `entities` block is the allow-list.
    """
    hard: list[str] = []
    warn: list[str] = []

    text = synthesis_text or ""
    entities = (briefing_json or {}).get("entities", {}) or {}

    # Track which closed-class allow-lists are empty → unverifiable, not "clean".
    # Reported explicitly so an empty allow-list is a visible coverage gap.
    unverified: list[str] = [
        cls
        for cls in ("kernel_refs", "teams", "authors", "discussion_titles")
        if not entities.get(cls)
    ]

    # --- HARD: required header present (exact level) ---
    # Line-anchored exact match so a wrong heading level (### Key Strategies)
    # does NOT pass — `"## ..." in text` would, since `##` is a substring of
    # `###`. The prompt mandates exactly `## Key Strategies`.
    header_re = re.compile(r"^" + re.escape(SYNTHESIS_HEADER) + r"\s*$", re.MULTILINE)
    if not header_re.search(text):
        hard.append(f"missing required header '{SYNTHESIS_HEADER}' (exact level)")

    # --- HARD: empty / no content ---
    if not text.strip():
        hard.append("synthesis is empty")

    # --- HARD: hallucinated kernel refs (high-precision anchor) ---
    # Only assert when we actually gathered a kernel list. With an EMPTY
    # allow-list we have no basis to call a cited ref fabricated (e.g. no-token
    # tier where a writeup legitimately cites one) — that's the locked
    # "don't assert against data we didn't gather" rule. The empty case is
    # surfaced via `unverified=['kernel_refs']` instead.
    allowed_refs = {r.lower() for r in entities.get("kernel_refs", [])}
    if allowed_refs:
        used_refs = _extract_kernel_refs(text)
        for ref in sorted(used_refs):
            if ref.lower() not in allowed_refs:
                hard.append(f"hallucinated kernel ref not in briefing: '{ref}'")

    # --- HARD: hallucinated competition slug ---
    # Only assert if the synthesis uses a slug-shaped token that looks like a
    # competition ref but isn't the real one. We keep this narrow: a token of
    # the exact slug form appearing after "competitions/" — anything else is
    # too ambiguous to assert (out of scope, eyeball pass).
    real_slug = (entities.get("competition_slug") or "").lower()
    for m in re.finditer(r"competitions/([a-z0-9][a-z0-9-]*)", text.lower()):
        if real_slug and m.group(1) != real_slug:
            hard.append(f"references a different competition slug: '{m.group(1)}'")

    # --- HARD: leaked page chrome ---
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if (
            s in STANDALONE_ICON_WORDS
            or ICON_LIGATURE_RE.match(s)
            or FILE_LISTING_RE.match(s)
            or FILE_COUNT_RE.match(s)
        ):
            hard.append(f"leaked page chrome: '{s}'")

    # --- WARN: bullet count outside prompt's stated range ---
    n_bullets = len(_bullet_lines(text))
    if n_bullets and not (SYNTHESIS_MIN_BULLETS <= n_bullets <= SYNTHESIS_MAX_BULLETS):
        warn.append(
            f"bullet count {n_bullets} outside expected "
            f"{SYNTHESIS_MIN_BULLETS}-{SYNTHESIS_MAX_BULLETS}"
        )

    # --- WARN: word count over the prompt's max ---
    n_words = len(text.split())
    if n_words > SYNTHESIS_MAX_WORDS:
        warn.append(f"word count {n_words} exceeds max {SYNTHESIS_MAX_WORDS}")

    # --- WARN: display-name entities (teams / authors / discussion_titles) ---
    # Per the settled spec, these are display strings (fuzzy by nature), so:
    #   * WARN-level only (a paraphrase like "the top team" must NOT fail),
    #   * matched ONLY against verbatim-quoted spans (backtick/quote-wrapped) —
    #     never `if name in text` substring matching (the false-match trap),
    #   * allow-list is deduped + CASE-SENSITIVE exact full-string,
    #   * empty allow-list → out-of-scope (already in `unverified`), no warn.
    # The trigger: the synthesis *quotes* a span that looks like a handle but is
    # NOT in any display-name allow-list → likely a fabricated/misquoted name.
    display_allow: set[str] = set()
    for cls in ("teams", "authors", "discussion_titles"):
        display_allow.update(entities.get(cls, []))  # exact, case-sensitive

    if display_allow:  # only assert when we have data to assert against
        for span in _extract_quoted_spans(text):
            if "/" in span or _FILENAME_RE.match(span):
                # kernel refs (covered elsewhere) / filenames are not handles.
                continue
            # Only warn if the span actually RESEMBLES a handle — otherwise a
            # backtick-quoted technical term (`ensembling`, `LoRA`, `XGBoost`)
            # would warn on every real synthesis (noise floor that trains us to
            # ignore the gate). A span is a candidate handle if it is multi-word
            # Titlecase OR matches the token-shape of a real allow-list entry.
            if not _looks_like_handle(span):
                continue
            if span not in display_allow:
                warn.append(
                    f"quoted entity not found verbatim in briefing "
                    f"(teams/authors/titles): '{span}'"
                )

    return VerificationResult(
        ok=not hard, hard_failures=hard, warnings=warn, unverified=unverified
    )


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Verify a synthesis section against a briefing JSON")
    parser.add_argument("synthesis", help="Path to the synthesis .md/.txt file (or '-' for stdin)")
    parser.add_argument("briefing_json", help="Path to the briefing .json (from generate_briefing --json)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings too")
    args = parser.parse_args()

    synth = sys.stdin.read() if args.synthesis == "-" else Path(args.synthesis).read_text(encoding="utf-8")
    briefing = json.loads(Path(args.briefing_json).read_text(encoding="utf-8"))

    result = verify_synthesis(synth, briefing)
    for f in result.hard_failures:
        print(f"FAIL: {f}")
    for w in result.warnings:
        print(f"WARN: {w}")
    if result.unverified:
        print(f"UNVERIFIED (no allow-list data): {', '.join(result.unverified)}")
    if result.ok and not result.warnings:
        print("OK: all hard invariants pass"
              + (f" (note: {len(result.unverified)} entity class(es) unverifiable)"
                 if result.unverified else ""))

    if not result.ok or (args.strict and result.warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
