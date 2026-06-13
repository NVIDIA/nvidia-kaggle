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
| 4 | `runs/codex_rogii-wellbore-geology-prediction_002/` | codex | `40c87f04b3fd25df1461192d9e272b89` | **PASS** — 17 cited refs all in gathered set; hyperlinked citations + 4 plots (see provenance breakdown below) |

## The story this set tells

- **Exhibit 1 (headline):** the trace-grounded verifier catches a *real*
  hallucination — the agent fabricated `vdebout/315987` by conflating a
  discussion id/author into a kernel `owner/slug`. Caught with receipts.
- **Exhibits 2 & 3:** the same gate PASSES grounded briefs from *both*
  frameworks — so it distinguishes grounded from fabricated, it isn't just
  rejecting everything.
- **Cross-runtime divergence:** Codex and Claude chose different workflow mixes
  and produced md5-distinct briefs from the identical goal-only prompt.

## Hyperlinks + plots (Exhibit 4, rogii) — provenance breakdown (auditor voice)

Exhibit 4 adds two features: kernels cited as clickable `kaggle.com/code/<owner>/<slug>`
markdown links, and agent-generated plots. This breakdown is *our verification
verdict*, not the agent's self-report — the agent's brief makes no "trace-verified"
claim (that is our vocabulary). The frozen brief is untouched.

**Hyperlinks:** 17 kernels cited as `/code/` links; all 17 `owner/slug` extracted
and ⊆ the gathered set (grounding PASS). The verifier extracts refs equally from
backtick and link form; the local-file-path exclusion (`plots/x.png`, `raw/y.json`)
applies to backtick tokens only — never to `/code/` URLs, which are unambiguous
kernels — so it cannot drop a genuinely-cited kernel.

**Plots — coupling:** every plot script reads its sidecar `plots/<name>.json` and
renders the PNG from it (in one pass from the same gathered rows), so the PNG cannot
disagree with the verified JSON.

**Plots — per-plot provenance:** 3 of 4 are *gathered-value-verified* (each plotted
bar IS a datum present in the trace); 1 is a *derived metric* (plotted values are
computed from gathered data, so "every value appears in the trace" would be a
coincidental match, not provenance):

| plot | shape | provenance |
|---|---|---|
| `top_kernel_votes` | kernel `owner/slug` → vote count | **gathered-value verified** — each (ref, votes) pair traces to `kernel_query` output |
| `discussion_comment_engagement` | discussion id → comment count | **gathered-value verified** — traces to `discussion_query` output |
| `leaderboard_top20_scores` | team → public RMSE | **gathered-value verified** — distinctive floats (e.g. 5.928, 6.487) trace to leaderboard output |
| `kernel_vote_distribution` | vote-range bucket (`0-9`…`200-999`) → kernel count | **derived metric** — a histogram *computed* over gathered votes; its *input* (per-kernel votes) traces, but the plotted bucket-counts are computed, not gathered. Labeled illustrative/derived, NOT trace-verified insight. |

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
