#!/usr/bin/env bash
#
# Agentic Competition Intel demo: one skill, two agent frameworks.
#
# The agent is handed a natural research goal (not a script to run). It reads the
# skill's SKILL.md, chooses which of the skill's research workflows to run, chains
# them itself, and writes a strategy brief. Claude and Codex make different
# choices — that divergence is the showcase. The structured trace is the source
# of truth for verification.
#
# Usage:
#   ./run_demo.sh --runtime claude|codex <competition-slug>
#
# Outputs (under runs/<runtime>_<slug>_NNN/):
#   trace.jsonl   structured agent trace (codex --json / claude stream-json)
#   brief.md      the agent's strategy brief
#   plots/        agent-generated plots + sidecar JSON
# analyze_run.py then reports agentic-behavior evidence + grounding/schema verdicts.

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

# Each run writes to a unique, immutable dir (sequential NNN) so artifacts are
# never overwritten and every run stays independently re-derivable. A `latest`
# symlink points at the newest run.
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

# A natural, real-user-style research goal — names no scripts and no schema. The
# citation and plot-auditability conventions live in the skill (research-brief.md),
# so the agent picks them up on its own.
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

# Record the verbatim TASK — codex --json does not echo the prompt, so without
# this the prompt is unrecoverable from artifacts.
{
    echo "# runtime=$RUNTIME  competition=$SLUG  prompt_variant=natural"
    echo "# --- TASK (verbatim user prompt) ---"
    printf '%s\n' "$TASK"
} > "$RUN_DIR/prompt.txt"

case "$RUNTIME" in
    codex)
        # Codex headless, structured trace via --json. Sandbox bypass is required:
        # Codex's bubblewrap sandbox can't create user namespaces inside this
        # container, so model-run shell commands fail without it. Safe here because
        # we are already containerized.
        echo "" | codex exec --json \
            --skip-git-repo-check \
            -C "$REPO_ROOT" \
            --dangerously-bypass-approvals-and-sandbox \
            "$TASK" > "$TRACE" 2>&1 || true
        ;;
    claude)
        # Claude Code headless, structured trace via stream-json.
        CLAUDE_CMD=(claude -p "$TASK"
            --add-dir "$REPO_ROOT"
            --allowedTools "Bash Read Write Skill"
            --output-format stream-json --verbose)
        {
            echo "# Claude invocation"
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

# Trace-based verification: agentic-behavior evidence + grounding/schema eval.
SYN_ARG=()
[ -f "$BRIEF" ] && SYN_ARG=(--synthesis "$BRIEF")
python "$ANALYZE" "$TRACE" --runtime "$RUNTIME" "${SYN_ARG[@]}"
