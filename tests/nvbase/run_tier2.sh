#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NV-BASE Tier 2 — Semantic Governance (Duplication Detection)
#
# Usage:
#   bash tests/nvbase/run_tier2.sh                  # all checks
#   bash tests/nvbase/run_tier2.sh nvidia-kaggle-skill     # one skill (intra only)
#
# Prerequisites:
#   export NVIDIA_INFERENCE_KEY=nvapi-...
#   uv tool install --default-index \
#     https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base
#
# Requires NVIDIA_INFERENCE_KEY for embedding-based similarity.
# ──────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
SKILL_FILTER="${1:-}"

if ! command -v nv-base &>/dev/null; then
    echo "ERROR: nv-base not found. See run_tier1.sh for install instructions."
    exit 2
fi

if [[ -z "${NVIDIA_INFERENCE_KEY:-}" ]]; then
    echo "ERROR: NVIDIA_INFERENCE_KEY not set. Required for Tier 2 embeddings."
    echo "  Generate at: https://inference.nvidia.com/"
    exit 2
fi

if [[ -n "$SKILL_FILTER" ]]; then
    TARGET="$SKILLS_DIR/$SKILL_FILTER"
    if [[ ! -d "$TARGET" ]]; then
        echo "ERROR: Skill directory not found: $TARGET"
        echo "Available skills:"
        ls -1 "$SKILLS_DIR"
        exit 2
    fi
else
    TARGET="$SKILLS_DIR"
fi

OVERALL_RC=0

echo "════════════════════════════════════════════════════════════"
echo "NV-BASE Tier 2 — Semantic Governance"
echo "  Target: $TARGET"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Inter-skill similarity (only when targeting all skills) ──────
if [[ -z "$SKILL_FILTER" ]]; then
    echo "── Inter-Skill Similarity Check ─────────────────────────"
    nv-base similarity-check "$TARGET" --type skill --threshold 0.50 --full-body 2>&1 || OVERALL_RC=1
    echo ""
fi

# ── Intra-skill context optimization ────────────────────────────
echo "── Intra-Skill Context Optimization ─────────────────────"
nv-base context-optimization-check "$TARGET" 2>&1 || OVERALL_RC=1
echo ""

echo "════════════════════════════════════════════════════════════"
if [[ $OVERALL_RC -eq 0 ]]; then
    echo "RESULT: All Tier 2 checks PASSED"
else
    echo "RESULT: Some Tier 2 checks FAILED"
fi
echo "════════════════════════════════════════════════════════════"

exit $OVERALL_RC
