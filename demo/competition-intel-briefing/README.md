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
demo's evidence — both on the same competition (`rogii-wellbore-geology-prediction`)
and, crucially, spanning **both** agent frameworks so the "one skill, two
frameworks, gate-verified" claim has evidence on each side: a clean Codex run and
a clean Claude run, each PASS on both verdicts and clone-reproducible.

- **`runs/codex_rogii-wellbore-geology-prediction_019/`** — the clean Codex
  readability exemplar: human-legible plot labels (no bare ids), claim-vs-verified
  visual honesty (verified public-LB bars solid, author title-claims hatched),
  every plotted value traces. **`GROUNDING: PASS` + `SCHEMA: PASS`.**
- **`runs/claude_rogii-wellbore-geology-prediction_001/`** — the second
  framework: the *same* skill driven end-to-end by **Claude Code** instead of
  Codex. Claude read `SKILL.md` and chained 10 distinct workflows on its own,
  then wrote a brief citing 14 gathered refs. **`GROUNDING: PASS` + `SCHEMA:
  PASS`**, re-derived from a clean `git archive` export (no `~/.claude`, no local
  data) — clone-reproducible, self-contained (1.4 MB, no `data/`). Its
  `leaderboard_frontier` plot is the readability convention done right: a *score
  distribution*, not a name-wall — `#1 SaintLouis` (`5.785`) and the best public
  notebook (`pilkwang` geosteering, `7.501`) as solid verified bars, with
  top-3/10/20 cutoffs (`6.333`/`6.83`/`7.116`) as derived band markers, so a
  competitor sees the frontier and the gap to beat. Both verified entities are
  discussed in the brief's prose (not orphans). This run happened to execute
  main-thread; had Claude delegated, the harness would have captured the
  sub-agent traces into the run dir so the gate verifies them too — but here the
  parent trace is complete.

**What this demo honestly establishes:** both committed exemplars are clean —
they PASS because every plotted value and cited ref traces to gathered data and
every plot sidecar is schema-conformant. The *guarantee* comes from the verifier,
not from the runs being hand-picked: `analyze_run.py` FAILs `GROUNDING RESULT` on
any plotted value or cited ref that appears in no gathered artifact (a fabricated
score-ladder row, an invented leaderboard team, a conflated discussion id), and
that is what lets a clean PASS mean something. Re-run the gate on these runs —
or on any run of your own — and a fabrication would FAIL; these simply don't
contain one.

What the gate does **not** guarantee is that a plot is *useful* — e.g. a
leaderboard plot can be gate-clean yet show a wall of teams the brief never
discusses. "Is this plot useful to a reader?" is editorial judgment, not a
falsifiable check, so it stays a human-review responsibility. The skill
conventions (`research-brief.md`) raise the typical quality; the gate guarantees
accuracy; usefulness still needs a human reading the brief.

## Where the logic lives

The agent drives the skill itself — this folder is just the harness. The skill
is the source of truth at `skills/nvidia-kaggle-skill/` (`SKILL.md` plus the
individual workflow scripts the agent chooses to invoke). `run_demo.sh` hands
the agent a goal and captures its trace; `analyze_run.py` verifies the result.
It does not orchestrate the skill — the agent does.
