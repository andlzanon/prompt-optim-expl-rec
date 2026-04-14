#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPTIMIZATION_SCRIPT="$SCRIPT_DIR/run_llm_for_optimization.sh"
EXPLAINABILITY_SCRIPT="$SCRIPT_DIR/run_llm_for_explainability.sh"

echo "========================================"
echo "Step 1/2: Running explainability"
echo "Script: $EXPLAINABILITY_SCRIPT"
echo "========================================"

bash "$EXPLAINABILITY_SCRIPT"

echo "========================================"
echo "Step 2/2: Running prompt optimization"
echo "Script: $OPTIMIZATION_SCRIPT"
echo "========================================"

bash "$OPTIMIZATION_SCRIPT"

echo "========================================"
echo "Pipeline finished successfully."
echo "Repository: $REPO_DIR"
echo "Explainability output root: $REPO_DIR/out/test_explainability"
echo "Optimization output root: $REPO_DIR/out/prompt_optimization"
echo "========================================"
