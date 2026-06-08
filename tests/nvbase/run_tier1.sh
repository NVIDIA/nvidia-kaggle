#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# NV-BASE Tier 1 — Static & Security Validation for NVIDIA Kaggle
#
# Usage:
#   bash tests/nvbase/run_tier1.sh                    # full validation
#   bash tests/nvbase/run_tier1.sh nvidia-kaggle-skill       # one skill
#   bash tests/nvbase/run_tier1.sh --internal         # internal profile
#   bash tests/nvbase/run_tier1.sh --quality          # quality only
#   bash tests/nvbase/run_tier1.sh --security         # security only
#
# Prerequisites:
#   uv tool install --default-index \
#     https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base
#
# No LLM or API keys required — Tier 1 is fully deterministic.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
REPORTS_DIR="$REPO_ROOT/tests/nvbase/reports"
PROFILE="external"
SKILL_FILTER=""
MODE="full"   # full | quality | security

# ── Parse arguments ──────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --internal)  PROFILE="internal" ;;
        --external)  PROFILE="external" ;;
        --quality)   MODE="quality" ;;
        --security)  MODE="security" ;;
        --help|-h)
            echo "Usage: $0 [skill-name] [--internal|--external] [--quality|--security]"
            echo ""
            echo "  skill-name   Validate a single skill (e.g. nvidia-kaggle-skill)"
            echo "  --internal   Use NVIDIA internal validation profile"
            echo "  --external   Use external/public validation profile (default)"
            echo "  --quality    Run quality scoring only (4-dimension grades)"
            echo "  --security   Run security checks only (vuln, PII, unicode)"
            exit 0
            ;;
        *)  SKILL_FILTER="$arg" ;;
    esac
done

# ── Preflight ────────────────────────────────────────────────────
if ! command -v nv-base &>/dev/null; then
    echo "ERROR: nv-base not found. Install with:"
    echo "  uv tool install --default-index https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base"
    exit 2
fi

# ── Determine target ─────────────────────────────────────────────
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

# ── Quality-only mode ────────────────────────────────────────────
if [[ "$MODE" == "quality" ]]; then
    nv-base quality-check "$TARGET" 2>&1
    exit $?
fi

# ── Security-only mode ───────────────────────────────────────────
if [[ "$MODE" == "security" ]]; then
    echo "NV-BASE Security Scanning"
    echo "  Target: $TARGET"
    echo "══════════════════════════════════════════════════════════"
    echo ""
    OVERALL_RC=0

    echo "── Security Vulnerability Scan ──────────────────────────"
    nv-base security-scan "$TARGET" 2>&1 || OVERALL_RC=1
    echo ""

    echo "── PII Detection ────────────────────────────────────────"
    nv-base pii-scan "$TARGET" 2>&1 || OVERALL_RC=1
    echo ""

    echo "── Unicode Smuggling Detection ──────────────────────────"
    nv-base unicode-scan "$TARGET" 2>&1 || OVERALL_RC=1
    echo ""

    echo "══════════════════════════════════════════════════════════"
    if [[ $OVERALL_RC -eq 0 ]]; then
        echo "RESULT: All security checks PASSED"
    else
        echo "RESULT: Some security checks FAILED"
    fi
    echo "══════════════════════════════════════════════════════════"
    exit $OVERALL_RC
fi

# ── Full validation mode ─────────────────────────────────────────
mkdir -p "$REPORTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_NAME="tier1-${SKILL_FILTER:-all}-${PROFILE}-${TIMESTAMP}"

echo "════════════════════════════════════════════════════════════"
echo "NV-BASE Tier 1 Validation"
echo "  Target:  $TARGET"
echo "  Profile: $PROFILE"
echo "  Reports: $REPORTS_DIR/$REPORT_NAME.*"
echo "════════════════════════════════════════════════════════════"
echo ""

nv-base validate "$TARGET" \
    --no-dedup \
    --profile "$PROFILE" \
    -r cli,html,json \
    -o "$REPORTS_DIR" \
    2>&1 | tee "$REPORTS_DIR/${REPORT_NAME}.log"

RC=${PIPESTATUS[0]}

# ── Rename HTML/JSON reports with our timestamp ──────────────────
for ext in html json; do
    latest=$(ls -t "$REPORTS_DIR"/nv-base-output-*."$ext" 2>/dev/null | head -1)
    if [[ -n "$latest" ]]; then
        mv "$latest" "$REPORTS_DIR/${REPORT_NAME}.$ext"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
if [[ $RC -eq 0 ]]; then
    echo "RESULT: PASS"
else
    echo "RESULT: FAIL (exit code $RC)"
fi
echo "Reports saved to: $REPORTS_DIR/${REPORT_NAME}.*"
echo "════════════════════════════════════════════════════════════"

exit $RC
