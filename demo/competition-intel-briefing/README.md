# Competition Intel Briefing — Agentic Demo

One skill, two agent frameworks, real autonomous orchestration.

Given only a research **goal** (no script named), an LLM agent — running under
either **OpenAI Codex** or **Claude Code** — loads the `nvidia-kaggle-skill`,
decides on its own which of the skill's individual workflows to run, chains them
(competition info, dataset, discussions, kernels, writeups), and writes a
strategy brief. We then verify the brief against what the agent *actually
gathered*, reconstructed from its own execution trace.

## Run it

```bash
cd demo/competition-intel-briefing
./run_demo.sh --runtime codex  <competition-slug>
./run_demo.sh --runtime claude <competition-slug>
```

The `<competition-slug>` is the last path segment of a Kaggle competition URL —
e.g. `kaggle.com/competitions/spaceship-titanic` → `spaceship-titanic`.

```bash
./run_demo.sh --runtime codex spaceship-titanic
./run_demo.sh --runtime claude titanic
```

Each run writes to its own immutable dir `runs/<runtime>_<slug>/`:

- `brief.md` — the agent's strategy brief
- `trace.jsonl` — the structured agent trace (what the agent actually did)
- `cmd.txt` — the exact invocation

The harness then runs `analyze_run.py` automatically, which reports:

1. **Agentic-behavior evidence** — from the trace: did the agent read SKILL.md
   and chain multiple distinct individual workflows on its own?
2. **Grounding** — every `owner/slug` ref the brief cites must be in the
   **gathered set** (refs that appeared in the agent's actual tool output or
   command args, reconstructed from the independent trace). PASS / FAIL /
   DEGRADED (gathered set unrecoverable).

## Requirements

- `KAGGLE_API_TOKEN` (loaded automatically from the project `.env`) — the agent
  hits the live Kaggle API for kernels and discussions.
- Auth for the runtime you invoke (Codex / Claude Code), already configured in
  this environment. Codex here runs on NVIDIA's `gpt-5.5` inference API.

A run takes a few minutes (the agent is doing live research). Pick a competition
with public kernels and discussions — the brief is only as rich as what's
publicly available.

## How verification works

`analyze_run.py` emits **two independent verdicts** off the run's structured
trace, so accuracy and presentation never get conflated:

- **`GROUNDING RESULT`** (accuracy floor, applies to every run) — every value a
  plot shows and every `owner/slug` ref the brief cites must trace to the
  agent's *actual gathered output* (trace + the run's `research/`/`raw/` query
  files). A value present in no gathered artifact is a fabrication → **FAIL**. A
  value tagged `provenance:"verified"` that doesn't trace is a hard FAIL on any
  plot (the tag is falsifiable; the gate checks it). Dataset-derived values are
  re-computed against the downloaded CSVs, not coincidence-matched.
- **`SCHEMA CONFORMANCE`** — every plot sidecar entry must carry a `provenance`
  from the closed set `{verified, title-claim, derived}`.

Exit codes are distinct (1 = accuracy fail, 2 = grounding not-exercised,
4 = schema-only fail, 0 = clean) so a schema lapse is never misread as a
fabrication.

## Verified exhibits

Two frozen, independently re-derived runs are committed under `runs/` as the
demo's evidence:

- **`runs/codex_rogii-wellbore-geology-prediction_022/`** — the accuracy
  guarantee. In its score-ladder plot the agent invented two notebooks that
  appear in no gathered artifact — `ROGII: Geostat, Softmax NCC Hybrid` (LB
  `9.946`) and `ROGII 10.239 Weblore Predicition` (LB `10.239`). The committed
  gate catches both: **`GROUNDING RESULT: FAIL`**, while **`SCHEMA CONFORMANCE:
  PASS`** — so the run is presentation-clean yet accuracy-caught, which is
  precisely what the two-verdict design is for. This exhibit is *expected* to
  fail grounding; that is its whole purpose. The `9.946` and `10.239` rows are
  the real fabrications, and they FAIL on any clone because the values are
  absent from the run's shipped trace.

  The grounding FAIL also lists a third value — `median hidden eval fraction =
  0.7399…` from the dataset-shape plot. That one is a *legitimately-derived*
  metric the agent computed from the downloaded competition CSVs, not a
  fabrication. The gate flags it because it can only token-trace values against
  gathered text and can't recompute this aggregate without the local data — and
  those CSVs are deliberately not shipped in the PR (multi-GB, gitignored). This
  is a known, honest narrowness: a derived value the gate can't auto-verify
  from PR-included artifacts. It does not undermine the exhibit — the two real
  fabrications are the headline, and they are provably ungathered from the
  shipped trace alone.
- **`runs/codex_rogii-wellbore-geology-prediction_019/`** — the clean
  readability exemplar: human-legible plot labels (no bare ids), claim-vs-verified
  visual honesty (verified public-LB bars solid, author title-claims hatched),
  every plotted value traces. **`GROUNDING: PASS` + `SCHEMA: PASS`.**

**What this demo honestly establishes:** the gate *guarantees* the plots don't
lie (it mechanically caught every fabrication/schema lapse across many runs);
it does **not** guarantee they're *useful* — e.g. a leaderboard plot can be
gate-clean yet show a wall of teams the brief never discusses. "Is this plot
useful to a reader?" is editorial judgment, not a falsifiable check, so it stays
a human-review responsibility. The skill conventions (`research-brief.md`) raise
the typical quality; the gate guarantees accuracy; usefulness still needs a human
reading the brief.

## Where the logic lives

The agent drives the skill itself — this folder is just the harness. The skill
is the source of truth at `skills/nvidia-kaggle-skill/` (`SKILL.md` plus the
individual workflow scripts the agent chooses to invoke). `run_demo.sh` hands
the agent a goal and captures its trace; `analyze_run.py` verifies the result.
It does not orchestrate the skill — the agent does.
