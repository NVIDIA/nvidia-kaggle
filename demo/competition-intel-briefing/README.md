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

## Verified exhibits

Three frozen, independently re-derived runs are committed under `runs/` as the
demo's evidence. See [`EXHIBITS.md`](./EXHIBITS.md) for the index, md5s, and the
exact grounding claim, and [`DESIGN.md`](./DESIGN.md) for the design and the
honest cross-runtime findings.

## Where the logic lives

The agent drives the skill itself — this folder is just the harness. The skill
is the source of truth at `skills/nvidia-kaggle-skill/` (`SKILL.md` plus the
individual workflow scripts the agent chooses to invoke). `run_demo.sh` hands
the agent a goal and captures its trace; `analyze_run.py` verifies the result.
It does not orchestrate the skill — the agent does.
