#!/usr/bin/env bash
# Master pipeline — generates evaluation data for all configured domains.
#
# Usage:
#   ./scripts/generate_all.sh [ENV_TAG]
#
# Example:
#   ./scripts/generate_all.sh STAGING

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Export .env so child Python processes inherit credentials
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Optional environment tag from CLI argument
if [[ -n "${1:-}" ]]; then
    export ENV_TAG="$1"
    echo "Environment tag: $ENV_TAG"
fi

echo "============================================"
echo "  Eval QA Factory — Full Generation Pipeline"
echo "============================================"
echo ""

START_TIME=$SECONDS

# Add your domain configs here. Each line runs the generator with a
# different config module from configs/.
python "$PROJECT_DIR/generator.py" --config example_config --batch

# Example additional domains:
# python "$PROJECT_DIR/generator.py" --config compensation_config --batch
# python "$PROJECT_DIR/generator.py" --config benefits_config --batch
# python "$PROJECT_DIR/generator.py" --config talent_config --batch

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "============================================"
echo "  Pipeline complete in ${ELAPSED}s"
echo "============================================"
echo ""
echo "Output directory:"
ls -la "$PROJECT_DIR/output/" 2>/dev/null || echo "  (no output yet)"
