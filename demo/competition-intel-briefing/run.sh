#!/usr/bin/env bash
#
# Thin demo runner for the Competition Intel Briefing.
#
# This is a presentation wrapper only — the actual logic lives in the skill at
# skills/nvidia-kaggle-skill/scripts/generate_briefing.py (the source of truth,
# a registered SKILL.md workflow). This script just invokes it with sensible
# demo defaults and writes the artifact next to this README.
#
# Usage:
#   ./run.sh <competition-slug-or-url> [extra generate_briefing.py args...]
#
# Examples:
#   ./run.sh llm-20-questions
#   ./run.sh titanic --skip-writeups
#
# Token: the auth tiers (Top Public Kernels, Top Discussions) populate when
# KAGGLE_API_TOKEN is available. generate_briefing.py loads it from the project
# .env automatically; without it those sections render clear placeholders and
# the public tiers (Overview, Dataset, Top Writeups) still run.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <competition-slug-or-url> [extra args...]" >&2
    exit 1
fi

SLUG="$1"
shift

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
SKILL_SCRIPT="$REPO_ROOT/skills/nvidia-kaggle-skill/scripts/generate_briefing.py"

OUTPUT="$DEMO_DIR/${SLUG}_briefing.md"

echo "Generating Competition Intel Briefing for '$SLUG'..."
echo "  skill script: $SKILL_SCRIPT"
echo "  output:       $OUTPUT"
echo

python "$SKILL_SCRIPT" "$SLUG" --output "$OUTPUT" "$@"
