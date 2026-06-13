# Demo Exhibits — frozen, verified artifacts

These are the **only** artifacts the demo presents. Each is frozen (immutable
dir, no further runs write here) and independently re-derivable from its raw
`trace.jsonl`. Other `runs/*` dirs are scratch/earlier iterations — ignore them.

Verify with: `python analyze_run.py runs/<dir>/trace.jsonl --runtime <rt> --synthesis runs/<dir>/brief.md`

| # | Exhibit dir | runtime | brief.md md5 | verdict |
|---|---|---|---|---|
| 1 | `runs/codex_FAIL_caught_hallucination/` | codex | `700b4caab8acf6400fe38934a3aa8d14` | **FAIL** — gate caught fabricated `vdebout/315987` (discussion id conflated into a kernel ref) |
| 2 | `runs/codex_spaceship-titanic_002/` | codex | `068a995079028b6daaa31c0ac18d4b47` | **PASS** — 6 cited refs all in gathered set |
| 3 | `runs/claude_spaceship-titanic_003/` | claude | `4b98b4219ec71f333fb712ca2ef2a355` | **PASS** — 6 cited refs all in gathered set; 0 subagent dispatches (fully recoverable) |

## The story this set tells

- **Exhibit 1 (headline):** the trace-grounded verifier catches a *real*
  hallucination — the agent fabricated `vdebout/315987` by conflating a
  discussion id/author into a kernel `owner/slug`. Caught with receipts.
- **Exhibits 2 & 3:** the same gate PASSES grounded briefs from *both*
  frameworks — so it distinguishes grounded from fabricated, it isn't just
  rejecting everything.
- **Cross-runtime divergence:** Codex and Claude chose different workflow mixes
  and produced md5-distinct briefs from the identical goal-only prompt.

## Claude orchestration framing (honest, do not overstate)

This environment's `~/.claude/CLAUDE.md` carries an "invoke Agent Skills inside a
subagent" instruction. During development Claude's orchestration depth was
**non-deterministic** across runs: the committed `claude_spaceship-titanic_003`
exhibit stayed fully parent-level (0 subagents, recoverable, grounded-PASS), while
other runs delegated into subagents whose outputs weren't exposed in the parent
trace (reported honestly as DEGRADED, never a pass). We did **not** isolate the
cause (config vs. runtime), and make no claim that a prompt nudge "fixed" it or
that this is a controlled two-runtime comparison. The committed exhibit is a
genuine parent-level grounded-PASS; the variability is an honest observation.

## Grounding claim — exact wording (defensible)

"Cited `owner/slug` refs ⊆ the **gathered set**, where *gathered* = refs that
appeared in the agent's actual tool **output** or **command arguments** during
the run (reconstructed from the independent trace; cached-notebook *path* refs
are admitted only when corroborated by a command-arg invocation)."

NOTE: agents typically cite refs they saw in `fetch_top_kernel_scores` /
`kernel_query` **list output**, not only ones they ran `kernel_read` on — so the
honest claim is *output-or-command* provenance, NOT the stricter
*command-arg-only* provenance. Both are sound grounding (the ref genuinely
appeared in gathered data, never minted from an ambiguous path split).
