#!/usr/bin/env python3
"""Synthesis layer for the Competition Intel Briefing.

This module is the *single source of truth for the synthesis prompt*. Both
agent-framework harnesses (Claude Agent SDK, OpenAI Codex) build the prompt
here and run it through their own model loop, so the prompt is written once and
never duplicated per runtime.

Two entry points:
  - ``build_synthesis_prompt(briefing)`` — PURE. No API call, no token, no
    network. Returns the prompt string. Unit-testable and deterministic.
  - ``synthesize(briefing, provider=...)`` — convenience that actually calls a
    model (provider-pluggable). This is the deterministic-fallback / tool-style
    path; the headline demo uses the agent-loop path via build_synthesis_prompt.

``briefing`` may be either the rendered markdown string or the structured dict
from ``generate_briefing.collect_briefing_data`` (preferred — cleaner input).
"""

from __future__ import annotations

import json
from typing import Any

# The synthesis instruction, written ONCE. Both runtimes consume this verbatim.
# Constraints are deliberately framed to keep the output verifiable: a fixed
# section header, bounded length, and an explicit grounding rule that maps to
# the closed-class entity allow-list the verifier checks.
SYNTHESIS_INSTRUCTION = """\
You are a Kaggle competition strategist. You are given an intelligence briefing \
for a single Kaggle competition: its overview, dataset description, and \
(where available) top solution writeups, public kernels, and discussions.

Write a concise **Key Strategies** section in Markdown that synthesizes what it \
takes to do well in THIS competition. Requirements:

- Start with exactly this header line: `## Key Strategies`
- Then 3–6 bullet points, each a specific, actionable strategy grounded in the \
briefing (e.g. a technique a top team used, a dataset quirk, an evaluation nuance).
- Be specific to this competition — no generic ML advice that would apply to any contest.
- Ground every claim in the briefing. Do NOT invent kernel references \
(owner/slug), team names, or discussion authors that are not present in the \
briefing. If you name an entity, it must appear in the briefing.
- Keep it under 250 words. No preamble, no closing remarks — just the section.
"""

# Bounds the verifier can assert against (kept here so the contract lives with
# the prompt that promises it). The verifier imports these — it can't read the
# prose, so the numbers must live in one place to avoid drift.
SYNTHESIS_HEADER = "## Key Strategies"
SYNTHESIS_MAX_WORDS = 250
SYNTHESIS_MIN_BULLETS = 3
SYNTHESIS_MAX_BULLETS = 6


def _render_briefing_for_prompt(briefing: Any) -> str:
    """Normalize a briefing (markdown str or structured dict) into prompt text."""
    if isinstance(briefing, str):
        return briefing.strip()

    if isinstance(briefing, dict):
        # Render the structured data into a compact, readable block. Using the
        # structured form keeps the model anchored on clean fields rather than
        # scraped prose, and mirrors exactly the entities the verifier knows.
        lines: list[str] = []
        slug = briefing.get("competition_slug", "")
        lines.append(f"# Competition: {slug}")
        if briefing.get("overview"):
            lines.append("\n## Overview\n" + str(briefing["overview"]).strip())
        if briefing.get("dataset"):
            lines.append("\n## Dataset\n" + str(briefing["dataset"]).strip())

        writeups = briefing.get("writeups") or []
        if writeups:
            lines.append("\n## Top Solution Writeups")
            for w in writeups:
                rank = w.get("rank") or "?"
                team = w.get("team") or "Unknown"
                lines.append(f"- Rank {rank}: {team} ({w.get('url', '')})")

        kernels = briefing.get("kernels") or []
        if kernels:
            lines.append("\n## Top Public Kernels")
            for k in kernels:
                score = k.get("score")
                score_s = f", score={score}" if score is not None else ""
                lines.append(
                    f"- {k.get('title', '')} (`{k.get('ref', '')}`, "
                    f"votes={k.get('votes')}{score_s})"
                )

        discussions = briefing.get("discussions") or []
        if discussions:
            lines.append("\n## Top Discussions")
            for d in discussions:
                lines.append(
                    f"- {d.get('title', '')} — {d.get('author', '')} "
                    f"(votes={d.get('votes')})"
                )
        return "\n".join(lines).strip()

    raise TypeError(
        f"briefing must be a markdown str or structured dict, got {type(briefing).__name__}"
    )


def build_synthesis_prompt(briefing: Any) -> str:
    """Return the synthesis prompt for a briefing. PURE — no API call.

    ``briefing`` is the rendered markdown string OR the structured dict from
    ``collect_briefing_data``. The returned string is what each runtime feeds to
    its own model loop.
    """
    rendered = _render_briefing_for_prompt(briefing)
    if not rendered:
        raise ValueError("briefing is empty; nothing to synthesize.")

    return (
        f"{SYNTHESIS_INSTRUCTION}\n"
        f"--- BEGIN BRIEFING ---\n"
        f"{rendered}\n"
        f"--- END BRIEFING ---\n"
    )


def synthesize(briefing: Any, *, provider: str | None = None, model: str | None = None) -> str:
    """Build the prompt and call a model. Provider-pluggable convenience path.

    This is the tool-style / deterministic-fallback path (NOT the headline
    agent-loop demo). Provider is chosen by ``provider`` or inferred from env:
    OPENAI_API_KEY -> openai, ANTHROPIC_API_KEY -> anthropic.
    """
    import os

    prompt = build_synthesis_prompt(briefing)
    chosen = provider or _infer_provider()

    if chosen == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or os.environ.get("CODEX_MODEL") or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    if chosen == "anthropic":
        from anthropic import Anthropic  # may require `pip install anthropic`

        client = Anthropic()
        resp = client.messages.create(
            model=model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()

    raise RuntimeError(
        "No LLM provider available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
        "or pass provider=."
    )


def _infer_provider() -> str | None:
    import os

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


if __name__ == "__main__":
    # CLI: build (and optionally run) the synthesis prompt from a briefing file.
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build or run the synthesis prompt for a briefing")
    parser.add_argument("briefing", help="Path to a briefing .md or .json file")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually call a model (default: just print the prompt)",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    raw = Path(args.briefing).read_text(encoding="utf-8")
    briefing: Any = json.loads(raw) if args.briefing.endswith(".json") else raw

    if args.run:
        print(synthesize(briefing, provider=args.provider, model=args.model))
    else:
        print(build_synthesis_prompt(briefing))
