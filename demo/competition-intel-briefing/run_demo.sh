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
# The TASK is a single natural, real-user-style ask ("research this competition
# and brief me, with links and a few charts") — it names no scripts and no
# schema. The citation + plot-auditability conventions live in the SKILL
# (`research-brief.md`), so the agent picks them up automatically; validated to
# hold the full bar (agentic + grounded hyperlinks + coupled provenance-traceable
# plots) on rogii + spaceship under independent re-derive. This is the shipped
# behavior. (Earlier iterations A/B-tested more spelled-out prompts — moderate /
# full / aggressive — against this one; the natural prompt won, so those
# variants and the PROMPT_VARIANT switch were removed.)
read -r -d '' TASK <<EOF || true
Research the ${SLUG} Kaggle competition with the nvidia-kaggle skill and write me
a strategy brief on what it takes to do well. Include the key public notebooks
and discussions as links, and a few plots for insight. Save the brief to ${BRIEF}
and put any plot files under ${PLOTS_DIR}.
EOF

echo "=============================================================="
echo " Agentic Competition Intel demo — agent orchestrates the skill"
echo "   runtime:     $RUNTIME"
echo "   competition: $SLUG"
echo "   prompt:      natural"
echo "   trace:       $TRACE"
echo "=============================================================="

# Record the exact TASK for BOTH runtimes, so every run is independently
# auditable (the literal user prompt used). codex --json does not echo the
# prompt into its trace, so without this the prompt is unrecoverable from
# artifacts.
{
    echo "# runtime=$RUNTIME  competition=$SLUG  prompt_variant=natural"
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
