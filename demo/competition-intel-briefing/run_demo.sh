#!/usr/bin/env bash
#
# Layer-4 demo harness: ONE skill, TWO agent frameworks, AGENTIC orchestration.
#
# The agent is handed a RESEARCH GOAL (not a script to run). It reads the
# skill's SKILL.md, decides which of the skill's INDIVIDUAL research workflows
# to use, chains them in an order IT chooses, and synthesizes a strategy brief
# from what it gathered. Claude and Codex make different choices — that
# divergence is the showcase.
#
# The task prompt is a NATURAL research goal — it names no scripts and adds NO
# "you may not use X" clause. We want the agent's REAL orchestration choices,
# observed not engineered: it may chain granular workflows, and it may also call
# the convenience generator — whatever it judges best. We report what it
# actually did. The structured trace is the first-class deliverable: it proves
# what the agent orchestrated and is the source of truth for the grounding
# allow-list.
#
# Usage:
#   ./run_demo.sh --runtime claude|codex <competition-slug>
#
# Outputs (under runs/<runtime>_<slug>/):
#   trace.jsonl   the structured agent trace (codex --json / claude stream-json)
#   brief.md      the agent's final strategy brief
# Then analyze_run.py reports agentic-behavior evidence + grounding-eval.

set -euo pipefail

RUNTIME=""
SLUG=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --runtime) RUNTIME="${2:-}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) SLUG="$1"; shift ;;
    esac
done

if [ "$RUNTIME" != "claude" ] && [ "$RUNTIME" != "codex" ]; then
    echo "ERROR: --runtime must be 'claude' or 'codex'." >&2; exit 2
fi
if [ -z "$SLUG" ]; then
    echo "ERROR: competition slug required. Usage: $0 --runtime claude|codex <slug>" >&2; exit 2
fi

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
ANALYZE="$DEMO_DIR/analyze_run.py"

# Load Kaggle token so the agent's skill workflows can reach the auth tiers.
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

# Each run writes to a UNIQUE, immutable dir so artifacts never overwrite each
# other — every re-run is frozen and independently re-derivable, which stops the
# "everyone analyzed a since-overwritten file" churn. Sequential id (no Date in
# this env). A `latest` symlink points at the newest run for convenience.
RUNS_ROOT="$DEMO_DIR/runs"
mkdir -p "$RUNS_ROOT"
_n=1
while [ -e "$RUNS_ROOT/${RUNTIME}_${SLUG}_$(printf '%03d' "$_n")" ]; do _n=$((_n+1)); done
RUN_DIR="$RUNS_ROOT/${RUNTIME}_${SLUG}_$(printf '%03d' "$_n")"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$RUNS_ROOT/${RUNTIME}_${SLUG}_latest"
TRACE="$RUN_DIR/trace.jsonl"
BRIEF="$RUN_DIR/brief.md"
PLOTS_DIR="$RUN_DIR/plots"
mkdir -p "$PLOTS_DIR"

# The research GOAL — a natural, thorough investigation. Names no scripts and
# adds NO artificial "you may not use file X" clause: we want the agent's REAL
# orchestration choices, not choices distorted to manufacture a clean trace.
# The framing (overview, data, community, top notebooks) naturally favors the
# granular workflows; if an agent also calls the convenience generator, that's
# the demotion working, reported as a note — not engineered away.
#
# Two quality asks added this iteration, both designed to stay AUDITABLE:
#  - Hyperlinks: cite KERNELS as kaggle.com/code/<owner>/<slug> markdown links
#    so the cited owner/slug stays extractable by the grounding verifier
#    (discussions/competitions as links are fine but don't feed the kernel gate).
#  - Plots: the agent writes+runs its OWN plotting code. To make each plot's
#    numbers AUDITABLE (and impossible for the PNG to disagree with a parallel
#    claim), the flow is coupled: gather data -> write a (label,value) sidecar
#    JSON -> the plot script READS that JSON and renders the PNG from it. So the
#    PNG is provably a rendering of the JSON, and a verifier checking each
#    (label,value) against the trace governs the image. Plots whose values can't
#    be traced to gathered data are illustrative-only. Agent decides WHAT to plot.
# Prompt variant. Default is `natural` — a real-user-style ask ("research this
# competition and brief me, with links and a few charts") that names no scripts
# and no schema. The citation + plot-auditability conventions live in the SKILL
# (`research-brief.md`), so the agent picks them up automatically; validated to
# hold the full bar (agentic + grounded hyperlinks + coupled provenance-traceable
# plots) on rogii + spaceship under independent re-derive. This is the shipped
# behavior and what we iterate on. The older spelled-out prompts are retained as
# `PROMPT_VARIANT` options: `moderate` (13-line explicit), `full` (30-line
# verbose); `aggressive` (8 lines) is kept for reference but was REJECTED —
# cutting the explicit plot-data schema made the agent emit bare [{label,value}]
# arrays with truncated labels, breaking plot-provenance auditability.
# Override with PROMPT_VARIANT=moderate|full|aggressive.
PROMPT_VARIANT="${PROMPT_VARIANT:-natural}"

case "$PROMPT_VARIANT" in
  natural)
    # A real-user-style request: no schema, no step-by-step. The citation +
    # plot-auditability conventions live in the skill (research-brief.md), so
    # the agent should pick them up automatically. The open question this
    # variant tests: does the agent honor skill-embedded conventions as
    # reliably as prompt-spelled-out ones (schema + coupling intact)?
read -r -d '' TASK <<EOF || true
Research the ${SLUG} Kaggle competition with the nvidia-kaggle skill and write me
a strategy brief on what it takes to do well. Include the key public notebooks
and discussions as links, and a few plots for insight. Save the brief to ${BRIEF}
and put any plot files under ${PLOTS_DIR}.
EOF
    ;;
  full)
read -r -d '' TASK <<EOF || true
I'm starting the ${SLUG} Kaggle competition. Use the nvidia-kaggle skill to
research it and brief me on the strategies that win. Investigate the
competition overview, the dataset, what the community discusses, and what the
top public notebooks do — then write me a strategy brief grounded in what you
find.

Cite your sources as clickable markdown links. For public notebooks/kernels,
link them as [title](https://www.kaggle.com/code/<owner>/<slug>) using the real
owner/slug you gathered — do not invent refs. Link discussions and the
competition page too where relevant.

Include 2-4 plots that give real insight (e.g. distribution of kernel votes,
discussion engagement, public-score spread, leaderboard trends). For EACH plot,
follow this exact coupled flow so the figure is auditable:
  1. Write the data you will plot to ${PLOTS_DIR}/<name>.json in this schema:
     {"title": "...",
      "source": "<which skill workflow produced it: kernel_query | fetch_top_kernel_scores | discussion_query | leaderboard>",
      "series": [{"label": "<owner/slug for kernels, or discussion id>", "value": <number>}, ...]}
     Each value MUST be a number you actually gathered from the skill — no
     invented or interpolated values. If a quantity is incomplete (e.g. some
     public scores 429'd), include only what you have and note it in the title.
  2. Write ${PLOTS_DIR}/<name>.py that READS ${PLOTS_DIR}/<name>.json and renders
     ${PLOTS_DIR}/<name>.png from it (matplotlib). The PNG must be a rendering of
     that JSON — do not plot from any other in-memory data.
  3. Run the script to produce the PNG, then embed it in the brief with the
     relative path plots/<name>.png.

Give me a focused brief — prioritize the highest-signal sources rather than
reading everything exhaustively, and make sure you finish and save the brief to
${BRIEF}.
EOF
    ;;
  moderate)
    # Compress research+hyperlink preamble to ~3 lines; keep the plot coupling
    # block essentially intact (it's the fragile, must-be-explicit part).
read -r -d '' TASK <<EOF || true
Research the ${SLUG} Kaggle competition with the nvidia-kaggle skill (overview,
data, discussions, top notebooks) and write me a strategy brief grounded in what
you gather. Cite kernels as clickable links [title](https://www.kaggle.com/code/<owner>/<slug>)
using the real owner/slug you gathered — never invent refs.

Add 2-4 insight plots. For EACH, the figure must be auditable via this coupled flow:
  1. Write its data to ${PLOTS_DIR}/<name>.json as
     {"title","source","series":[{"label":"<owner/slug or discussion id>","value":<number>}]}
     — only numbers you actually gathered, no invented/interpolated values.
  2. Write ${PLOTS_DIR}/<name>.py that READS that JSON and renders
     ${PLOTS_DIR}/<name>.png from it (matplotlib) — do not plot from any other data.
  3. Run it; embed plots/<name>.png in the brief.
Save the brief to ${BRIEF}.
EOF
    ;;
  aggressive)
    # Shrink the plot block too — test whether the agent infers coupling from a
    # terse instruction (Reviewer wants repeat runs here; coupling is at risk).
read -r -d '' TASK <<EOF || true
Use the nvidia-kaggle skill to research ${SLUG} (overview, data, discussions, top
notebooks) and write a strategy brief to ${BRIEF}, grounded in what you gather.
Cite kernels as [title](https://www.kaggle.com/code/<owner>/<slug>) links with the
real owner/slug — no invented refs.
Include 2-4 insight plots; for each, save its plotted data to
${PLOTS_DIR}/<name>.json (a {label,value} series of numbers you actually gathered),
a ${PLOTS_DIR}/<name>.py that reads that JSON and renders ${PLOTS_DIR}/<name>.png,
then embed plots/<name>.png.
EOF
    ;;
  *) echo "ERROR: unknown PROMPT_VARIANT '$PROMPT_VARIANT'" >&2; exit 2 ;;
esac

echo "=============================================================="
echo " Agentic Competition Intel demo — agent orchestrates the skill"
echo "   runtime:     $RUNTIME"
echo "   competition: $SLUG"
echo "   prompt:      $PROMPT_VARIANT"
echo "   trace:       $TRACE"
echo "=============================================================="

# Record the prompt variant + the exact TASK for BOTH runtimes, so every run is
# independently auditable (which variant + the literal user prompt used). codex
# --json does not echo the prompt into its trace, so without this the prompt is
# unrecoverable from artifacts.
{
    echo "# runtime=$RUNTIME  competition=$SLUG  prompt_variant=$PROMPT_VARIANT"
    echo "# --- TASK (verbatim user prompt) ---"
    printf '%s\n' "$TASK"
} > "$RUN_DIR/prompt.txt"

case "$RUNTIME" in
    codex)
        # Codex (NVIDIA gpt-5.5), headless, STRUCTURED trace via --json.
        # Sandbox bypass is REQUIRED here: Codex's default bubblewrap sandbox
        # cannot create user namespaces inside this container ("bwrap: No
        # permissions to create a new namespace"), so every model-run shell
        # command fails. We are already containerized, so bypassing is safe.
        echo "" | codex exec --json \
            --skip-git-repo-check \
            -C "$REPO_ROOT" \
            --dangerously-bypass-approvals-and-sandbox \
            "$TASK" > "$TRACE" 2>&1 || true
        ;;
    claude)
        # Claude Code, headless, STRUCTURED trace via stream-json. Normal
        # `claude -p` — this main entry point does NOT mutate any global config.
        #
        # NOTE on Claude orchestration depth: in an environment whose global
        # ~/.claude/CLAUDE.md instructs "invoke Agent Skills inside a subagent",
        # Claude may delegate skill work to sub-agents, which can make the
        # gathered set unrecoverable from the parent trace (analyze_run.py then
        # reports DEGRADED, honestly). That is an ENVIRONMENT CONFIG effect, not
        # a property of this runner. We deliberately do NOT neutralize it here:
        # editing a global file from the casual demo command is a footgun. See
        # DESIGN.md for the orchestration-depth finding.
        CLAUDE_CMD=(claude -p "$TASK"
            --add-dir "$REPO_ROOT"
            --allowedTools "Bash Read Write Skill"
            --output-format stream-json --verbose)
        {
            echo "# Claude invocation (normal claude -p; no config mutation)"
            printf '%q ' "${CLAUDE_CMD[@]}"
            echo
        } > "$RUN_DIR/cmd.txt"
        "${CLAUDE_CMD[@]}" > "$TRACE" 2>&1 || true
        ;;
esac

echo
if [ ! -s "$TRACE" ]; then
    echo "ERROR: no trace captured at $TRACE — cannot verify the run." >&2
    exit 1
fi

# Trace-based verification: agentic-behavior evidence + grounding-eval against
# the entity set reconstructed from the agent's OWN tool outputs in the trace.
SYN_ARG=()
[ -f "$BRIEF" ] && SYN_ARG=(--synthesis "$BRIEF")
python "$ANALYZE" "$TRACE" --runtime "$RUNTIME" "${SYN_ARG[@]}"
