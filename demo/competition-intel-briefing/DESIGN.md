# Demo Design — One Skill, Two Agent Frameworks, Real LLM Synthesis

> **Status:** The sections "Goal", "Architecture", and "Build order → layer 4"
> below capture the **original plan, which was SUPERSEDED during the build.**
> An early version had the agent run the single `generate_briefing.py`
> aggregator and write a fixed "Key Strategies" summary — i.e. the agent made
> one decision (run our script). That is *not* agentic, so we pivoted. **What we
> actually shipped is described in "Results — verified exhibits" at the bottom;
> that section is authoritative where it conflicts with the original plan.**
> The superseded sections are kept for design history and marked inline.

## Goal (original plan — SUPERSEDED, see Results)

> Superseded: the agent is NOT told to run `generate_briefing.py`. It receives a
> goal-only research prompt, autonomously loads the skill, and chains the skill's
> *individual* workflows itself. `generate_briefing.py` is demoted to one
> optional tool. The gate is grounding (cited refs ⊆ trace-gathered set), not a
> fixed header/length contract.

Prove the `nvidia-kaggle-skill` is genuinely runtime-portable by driving it
from **both** major agent frameworks present in this environment — **Claude
Code (Claude Agent framework)** and **OpenAI Codex** — where in each case the
LLM agent:

1. Loads the skill, and
2. Runs the deterministic data-gathering workflow (`generate_briefing.py`), then
3. **Synthesizes** the result into an LLM-written "Key Strategies" executive
   summary — the part that showcases the model, not just our glue code.

One skill. Two frameworks. Real LLM output. Same verifier over both.

## Environment (verified, not assumed)

| Component | Reality in this env |
|---|---|
| `claude` CLI | Claude Code 2.1.175 — headless via `claude -p`, skill-aware |
| `codex` CLI | codex-cli 0.139.0 — headless via `codex exec`, skill-aware |
| Codex model | **NVIDIA Inference API, `gpt-5.5`** (`~/.codex/config.toml`); `/nvidia-kaggle` already `trust_level = "trusted"` |
| Claude auth | `ANTHROPIC_API_KEY` set |
| Codex auth | `NVIDIA_INFERENCE_KEY` set (provider `nvidia`) |
| Kaggle | `KAGGLE_API_TOKEN` in `/nvidia-kaggle/.env` |
| Skill dirs | Codex discovers `~/.codex/skills/`; Claude discovers `.claude/skills/` and `--plugin-dir` |

Both frameworks are **natively skill-aware** — so the demo is "the framework
loads and drives the skill," not "we hand-wrote a tool loop." Stronger claim.

## Architecture (original plan — SUPERSEDED, see Results)

> Superseded: in the shipped demo each harness hands the agent a goal-only
> prompt and the agent decides which individual workflows to chain
> (`fetch_competition_info`, `fetch_dataset_info`, `discussion_*`, `kernel_*`,
> `fetch_top_kernel_scores`). The harness does NOT instruct it to "run briefing
> + synthesis". The shared `build_synthesis_prompt`/`synthesize` seam still
> exists but is not on the headline path.

```
                    skills/nvidia-kaggle-skill/  (source of truth)
                    ├─ generate_briefing.py   --json   ← (1) deterministic data
                    ├─ synthesize.py
                    │    ├─ build_synthesis_prompt(briefing) -> str   ← (2) shared prompt, pure
                    │    └─ synthesize(briefing, provider=…) -> str   ← deterministic fallback
                    └─ verify_synthesis(text, briefing_json) -> [violation]  ← (3) the gate
                                        │
                 ┌──────────────────────┴──────────────────────┐
        Claude Code harness                          OpenAI Codex harness
   (claude -p, points at skill,              (codex exec, points at skill,
    runs briefing + synthesis)                runs briefing + synthesis)
                 └──────────────────────┬──────────────────────┘
                              verify_synthesis() over BOTH outputs
```

**The only shared reasoning artifact is the prompt** (`build_synthesis_prompt`).
Each framework runs that prompt through *its own* model loop. Neither harness
duplicates logic; both are thin.

## Build order (layers)

Layers 1–3 are framework-independent and already greenlit / in progress.

1. **`generate_briefing.py --json`** — emit structured tiers. The JSON's
   closed-class fields ARE the verifier allow-list:
   `{competition_slug, kernels:[{ref,…}], discussions:[{author,title}], writeups:[{team,rank,url}]}`
2. **`build_synthesis_prompt(briefing) -> str`** — pure, no API call,
   unit-tested. Single source of the synthesis instruction.
3. **`verify_synthesis(text, briefing_json) -> [violation]`** — the gate:
   - section header present, non-empty, within length bounds → exact
   - **anti-hallucination (closed-class only):** kernel refs (`owner/slug`),
     competition slug, and verbatim-quoted handles in the synthesis must be a
     subset of the JSON entity sets. Free prose / domain vocab NOT asserted —
     precision over recall.
   - no leaked page chrome → reuse `support/page_text` patterns.
4. **Two thin harnesses — SUPERSEDED prompt shape, see Results.**
   > Superseded: the task prompt is **goal-only and names no script** ("research
   > the competition and brief me on the strategies that win"), identical across
   > both runtimes. The agent chooses and chains the skill's individual
   > workflows itself; it is NOT told to run `generate_briefing.py` or to emit a
   > fixed "Key Strategies" section. Verification is grounding-based (cited
   > `owner/slug` refs ⊆ the gathered set reconstructed from the run's trace).
   - **Claude:** `claude -p "<goal-only prompt>"` with skill exposed via
     `.claude/skills`, `--add-dir /nvidia-kaggle`, `--allowedTools`,
     `--output-format stream-json` (trace captured per run).
   - **Codex:** `codex exec --json --skip-git-repo-check -C /nvidia-kaggle
     "<same goal-only prompt>"` with skill in `~/.codex/skills/`,
     sandbox bypass (containerized). Uses NVIDIA `gpt-5.5`.
   - One `run_demo.sh --runtime claude|codex <slug>`; writes each run to an
     immutable `runs/<rt>_<slug>_NNN/` dir (trace + brief, md5-pinned).

## Demo deliverable

`demo/competition-intel-briefing/` gains:
- `run_demo.sh --runtime claude|codex <slug>` — runs the chosen framework
  end-to-end, writes `<slug>_briefing.md` + `<slug>_strategies_<runtime>.md`.
- `README.md` updated: the one-skill-two-runtimes story + sample outputs from
  both frameworks side by side.
- Sample artifacts from both runtimes committed.

## Verification contract (how we sign off non-deterministic output)

- **Deterministic parts** (1–3): gated exactly like everything else this
  session — unit tests + `verify_synthesis` returning zero violations across
  repeated runs.
- **The agent's brief** (non-deterministic, free-form): gated by *grounding* —
  `analyze_run.py` reconstructs the gathered-entity allow-list from the run's
  own trace (tool outputs + cached-notebook paths, corroborated against command
  args), and every cited `owner/slug` ref must be in it. Honest states:
  PASS (≥1 cited ref, all grounded) / FAIL (a cited ref not in the gathered
  set — real fabrication) / DEGRADED (gathered-set unrecoverable, e.g. outputs
  hidden inside subagents → chrome + eyeball only, never a vacuous pass).
  Header/length are **informational notes**, not hard-fails — the brief is
  free-form, not the old fixed `## Key Strategies` contract.
- Reviewer owns the gate: independently re-derives the allow-list and per-ref
  reconciliation from each frozen trace (not the analyzer's self-report), and
  signs off only when the committed analyzer's verdict matches that re-derivation.

## Open risks / to verify during build

- Exact headless flags for skill exposure differ per CLI — confirm each CLI
  actually invokes our skill (not just answers from general knowledge) with a
  smoke test before relying on it.
- Codex `gpt-5.5` via NVIDIA API: confirm it can run local python + write the
  output file under `workspace-write`.
- Model-id pinning: Claude side pin an explicit model id; Codex uses the
  configured `gpt-5.5`. Don't hardcode date-suffixed ids.
- Keep live Kaggle calls bounded (use `--skip-writeups` or cached tiers) so a
  stage run is fast and not rate-limited.

## Results — verified exhibits (spaceship-titanic)

Three frozen, independently re-derived exhibits make up the demo. "Independently
re-derived" = the gathered-set allow-list was rebuilt from each run's raw
`trace.jsonl` (output + cached-notebook paths, corroboration rule applied) and
every cited `owner/slug` ref reconciled against it — not taken from the
analyzer's self-reported summary.

1. **Codex — caught fabrication (headline).** `runs/codex_FAIL_caught_hallucination/`.
   The agent cited `vdebout/315987` as a kernel, but `315987` is a *discussion*
   id (author Vincent Debout) it had read — conflated into a fake `owner/slug`.
   The trace-grounded gate caught it: `GROUNDING RESULT: FAIL`. A genuine
   anti-hallucination catch with receipts, not a vacuous pass.

2. **Codex — clean grounded PASS.** `runs/codex_spaceship-titanic_002/`. SKILL.md
   read, 6 distinct workflows chosen by the agent, 6 cited refs all in the
   gathered set. The companion that shows the gate *distinguishes* grounded from
   fabricated.

3. **Claude — clean grounded PASS.** `runs/claude_spaceship-titanic_003/`
   (md5 `4b98b421`). SKILL.md read, **9 distinct workflows, 0 subagent
   dispatches (fully parent-level → recoverable), 0 `generate_briefing`
   invocations** (pure granular orchestration), 6 cited refs all in the gathered
   set (5 real kernels + the harmless literal `owner/slug` placeholder echoed
   from the citation instruction, which is itself in tool output → no false
   FAIL).

**Divergence we can honestly claim:** same byte-identical goal-only prompt, two
runtimes, md5-distinct briefs, different workflow mixes (Codex used
`discussion_read`-heavy; Claude chained 9 workflows including the `*_ingest`
steps). Both grounded-PASS.

**Honest framing of the Claude result (do not overstate):** this environment's
global `~/.claude/CLAUDE.md` carries a "invoke Agent Skills inside a subagent"
instruction. Across runs with that config **ON**, Claude's orchestration depth is
**non-deterministic**: `claude_003` stayed fully parent-level (0 subagents,
recoverable, grounded-PASS), while `_001`/`_002` delegated into subagents whose
outputs weren't exposed in the parent trace (reported honestly as DEGRADED, never
a pass). A single run with that config line **removed** (`_004`, see below) also
stayed parent-level — but it timed out before writing a brief, so n=1 and
incomplete. Taken together the evidence supports: **the config plausibly
increases the propensity to delegate, but is NOT proven to be the sole cause** —
`claude_003` (config-ON, parent-level) directly shows config-ON does not force
delegation. We make **no** claim that "the prompt nudge fixed it," that this is a
"fair two-runtime comparison," or that delegation is purely config- vs.
runtime-driven.

**Supporting note — controlled suppression attempt (`runs/claude_spaceship-titanic_004/`):**
a run with `~/.claude/CLAUDE.md:8` (the skill-in-subagent line) neutralized for
the run and restored immediately after (audit trail in `cmd.txt`; `--bare` was
tested and rejected because it disables the Skill tool entirely). Behaviorally it
showed **0 subagent dispatches, 7 workflows parent-level**, but **timed out before
producing a brief** (slow API, not a delegation issue). It is **n=1 and not a
grounded-PASS exemplar** — kept in the repo only as the verifiable artifact behind
the "config increases delegation propensity" observation, NOT as a 4th headline
exhibit. The cached-notebook recovery path in `analyze_run.py` is likewise proven
on a synthetic trace of the right shape, **not** exercised end-to-end by any of
the three frozen exhibits (their cited refs reconcile via tool stdout).
