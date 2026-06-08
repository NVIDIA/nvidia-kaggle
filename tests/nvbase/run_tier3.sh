#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NV-BASE Tier 3 — Live Agent Skills Evaluation
#
# Runs agents WITH and WITHOUT each skill, measures lift across
# 5 dimensions (Security, Correctness, Discoverability, Effectiveness,
# Efficiency). Generates HTML/JSON/CLI reports.
#
# Usage:
#   bash tests/nvbase/run_tier3.sh                          # all skills, default agents
#   bash tests/nvbase/run_tier3.sh kaggle-kernels           # one skill
#   bash tests/nvbase/run_tier3.sh --agents claude-code     # specific agent
#   bash tests/nvbase/run_tier3.sh --agents claude-code,codex --skip-baseline
#   bash tests/nvbase/run_tier3.sh --agent-model codex=openai/gpt-5.5
#   bash tests/nvbase/run_tier3.sh --env-mode local         # local mode (no Docker)
#   bash tests/nvbase/run_tier3.sh --gen-only               # generate evals.json only
#
# Prerequisites:
#   export NVIDIA_INFERENCE_KEY=nvapi-...
#   Docker must be running (default env-mode=docker)
#   uv tool install --default-index \
#     https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base
# ──────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
REPORTS_DIR="$REPO_ROOT/tests/nvbase/reports/tier3"
SKILL_FILTER=""
AGENTS="claude-code,codex"
ENV_MODE="docker"
SKIP_BASELINE=false
GEN_ONLY=false
N_ATTEMPTS=2
AGENT_MODEL=""

# ── Parse arguments ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agents)       AGENTS="$2"; shift 2 ;;
        --env-mode)     ENV_MODE="$2"; shift 2 ;;
        --skip-baseline) SKIP_BASELINE=true; shift ;;
        --gen-only)     GEN_ONLY=true; shift ;;
        --n-attempts)   N_ATTEMPTS="$2"; shift 2 ;;
        --agent-model)  AGENT_MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [skill-name] [OPTIONS]"
            echo ""
            echo "  skill-name          Evaluate a single skill (e.g. kaggle-kernels)"
            echo "  --agents AGENTS     Comma-separated agents (default: claude-code,codex)"
            echo "  --env-mode MODE     docker|local|astra-sandbox (default: docker)"
            echo "  --skip-baseline     Skip without-skill baseline (faster, no lift)"
            echo "  --gen-only          Generate evals/evals.json only, don't run eval"
            echo "  --n-attempts N      Attempts per eval case for Pass@k (default: 2)"
            echo "  --agent-model SPEC  Per-agent model override (e.g. codex=openai/gpt-5.5)"
            exit 0
            ;;
        *)  SKILL_FILTER="$1"; shift ;;
    esac
done

# ── Preflight ────────────────────────────────────────────────────
if ! command -v nv-base &>/dev/null; then
    echo "ERROR: nv-base not found. See run_tier1.sh for install instructions."
    exit 2
fi

if [[ -z "${NVIDIA_INFERENCE_KEY:-}" ]]; then
    echo "ERROR: NVIDIA_INFERENCE_KEY not set."
    echo "  Generate at: https://inference.nvidia.com/"
    exit 2
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "ERROR: KAGGLE_API_TOKEN not set."
    echo "  Get yours at: https://www.kaggle.com/settings → API → Create New Token"
    echo "  export KAGGLE_API_TOKEN='{\"username\":\"...\",\"key\":\"...\"}'"
    exit 2
fi

# ── Auto-patch Harbor for KAGGLE_API_TOKEN passthrough ──────────
PATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/patch_harbor_kaggle.py"
if [[ -f "$PATCH_SCRIPT" ]]; then
    if python3 "$PATCH_SCRIPT" --check &>/dev/null; then
        echo "INFO: Harbor KAGGLE_API_TOKEN patch already applied."
    else
        echo "INFO: Applying Harbor KAGGLE_API_TOKEN patch..."
        if python3 "$PATCH_SCRIPT"; then
            echo "INFO: Patch applied successfully."
        else
            echo "WARNING: Patch failed. KAGGLE_API_TOKEN may not be available in containers."
            echo "  Try manually: python3 $PATCH_SCRIPT"
        fi
    fi
else
    echo "WARNING: patch_harbor_kaggle.py not found at $PATCH_SCRIPT"
    echo "  KAGGLE_API_TOKEN may not be available in agent containers."
fi
echo ""

# Ensure astra-skill-eval is on PATH (bundled with nv-base via uv tool)
if ! command -v astra-skill-eval &>/dev/null; then
    NV_BASE_BIN=$(dirname "$(command -v nv-base)")
    ASTRA_BIN=$(find "$(dirname "$NV_BASE_BIN")" -name "astra-skill-eval" -type f 2>/dev/null | head -1)
    if [[ -z "$ASTRA_BIN" ]]; then
        # Try uv tool environment
        ASTRA_BIN=$(find ~/.local/share/uv/tools/nv-base -name "astra-skill-eval" -type f 2>/dev/null | head -1)
    fi
    if [[ -n "$ASTRA_BIN" ]]; then
        export PATH="$(dirname "$ASTRA_BIN"):$PATH"
        echo "INFO: Added astra-skill-eval to PATH: $(dirname "$ASTRA_BIN")"
    else
        echo "ERROR: astra-skill-eval not found. Reinstall nv-base."
        exit 2
    fi
fi

if [[ "$GEN_ONLY" == false && "$ENV_MODE" == "docker" ]]; then
    if ! command -v docker &>/dev/null; then
        echo "ERROR: Docker not found. Required for --env-mode docker."
        echo "  Use --env-mode local or --env-mode astra-sandbox instead."
        exit 2
    fi
fi

# ── Collect target skills ────────────────────────────────────────
SKILL_PATHS=()
if [[ -n "$SKILL_FILTER" ]]; then
    TARGET="$SKILLS_DIR/$SKILL_FILTER"
    if [[ ! -d "$TARGET" ]]; then
        echo "ERROR: Skill directory not found: $TARGET"
        echo "Available skills:"
        ls -1 "$SKILLS_DIR"
        exit 2
    fi
    SKILL_PATHS+=("$TARGET")
else
    for d in "$SKILLS_DIR"/*/; do
        [[ -f "$d/SKILL.md" ]] && SKILL_PATHS+=("$d")
    done
fi

mkdir -p "$REPORTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "════════════════════════════════════════════════════════════"
echo "NV-BASE Tier 3 — Live Agent Evaluation"
echo "  Skills:   ${#SKILL_PATHS[@]}"
echo "  Agents:   $AGENTS"
echo "  Env mode: $ENV_MODE"
echo "  Attempts: $N_ATTEMPTS"
echo "  Model:    ${AGENT_MODEL:-default}"
echo "  Baseline: $(if $SKIP_BASELINE; then echo 'skipped'; else echo 'enabled'; fi)"
echo "  Reports:  $REPORTS_DIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Generate eval datasets ──────────────────────────────
echo "── Step 1: Ensure evals/evals.json exists ───────────────"
GEN_FAILURES=()
for skill_path in "${SKILL_PATHS[@]}"; do
    skill_name=$(basename "$skill_path")
    eval_json="$skill_path/evals/evals.json"

    if [[ -f "$eval_json" ]]; then
        echo "  ✓ $skill_name — evals.json exists"
    else
        echo "  ⋯ $skill_name — generating evals.json..."
        if nv-base create-eval-dataset "$skill_path" --full 2>&1 | tail -3; then
            if [[ -f "$eval_json" ]]; then
                echo "  ✓ $skill_name — generated"
            else
                echo "  ✗ $skill_name — generation failed"
                GEN_FAILURES+=("$skill_name")
            fi
        else
            echo "  ✗ $skill_name — generation failed"
            GEN_FAILURES+=("$skill_name")
        fi
    fi
done

if [[ ${#GEN_FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: Failed to generate evals.json for: ${GEN_FAILURES[*]}"
fi

echo ""

if [[ "$GEN_ONLY" == true ]]; then
    echo "════════════════════════════════════════════════════════════"
    echo "--gen-only: stopping after dataset generation."
    echo "════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Step 2: Run agent evaluation ─────────────────────────────────
echo "── Step 2: Run agent evaluation ─────────────────────────"
EVAL_PASS=0
EVAL_FAIL=0
EVAL_SKIP=0

for skill_path in "${SKILL_PATHS[@]}"; do
    skill_name=$(basename "$skill_path")
    eval_json="$skill_path/evals/evals.json"

    if [[ ! -f "$eval_json" ]]; then
        echo "  SKIP $skill_name — no evals.json"
        ((EVAL_SKIP++))
        continue
    fi

    echo ""
    echo "── Evaluating: $skill_name ──────────────────────────────"
    SKILL_REPORT_DIR="$REPORTS_DIR/$skill_name-$TIMESTAMP"
    SKILL_RESULTS_DIR="$REPORTS_DIR/$skill_name-$TIMESTAMP/results"
    mkdir -p "$SKILL_REPORT_DIR" "$SKILL_RESULTS_DIR"

    EVAL_ARGS=(
        nv-base agent-eval "$skill_path"
        -a "$AGENTS"
        --env-mode "$ENV_MODE"
        --n-attempts "$N_ATTEMPTS"
        --harbor-keep-jobs
        --results-dir "$SKILL_RESULTS_DIR"
        -r cli,html,json
        -o "$SKILL_REPORT_DIR"
    )

    if [[ "$SKIP_BASELINE" == true ]]; then
        EVAL_ARGS+=(--skip-baseline)
    fi

    if [[ -n "$AGENT_MODEL" ]]; then
        EVAL_ARGS+=(--agent-model "$AGENT_MODEL")
    fi

    echo "  CMD: ${EVAL_ARGS[*]}"
    if "${EVAL_ARGS[@]}" 2>&1 | tee "$SKILL_REPORT_DIR/eval.log"; then
        echo "  ✓ $skill_name — PASS"
        ((EVAL_PASS++))
    else
        RC=$?
        echo "  ✗ $skill_name — FAIL (exit $RC)"
        ((EVAL_FAIL++))

        # Try to render reports from whatever results exist
        LATEST_RUN=$(find "$SKILL_RESULTS_DIR" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)
        if [[ -n "$LATEST_RUN" ]]; then
            echo "  ⋯ Rendering reports from partial results..."
            nv-base agent-eval-report "$LATEST_RUN" \
                -r cli,html,json \
                -o "$SKILL_REPORT_DIR" 2>&1 | tail -3 || true
        fi
    fi
done

# ── Summary ──────────────────────────────────────────────────────
TOTAL=$((EVAL_PASS + EVAL_FAIL + EVAL_SKIP))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Tier 3 Results: $EVAL_PASS passed, $EVAL_FAIL failed, $EVAL_SKIP skipped ($TOTAL total)"
echo "Reports: $REPORTS_DIR"
echo "════════════════════════════════════════════════════════════"

[[ $EVAL_FAIL -eq 0 && $EVAL_SKIP -eq 0 ]]
