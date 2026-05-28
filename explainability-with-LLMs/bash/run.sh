#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PREPARE_SCRIPT="$SCRIPT_DIR/run_prepare_selected_paths.sh"
OPTIMIZATION_SCRIPT="$SCRIPT_DIR/run_llm_for_optimization.sh"
EXPLAINABILITY_SCRIPT="$SCRIPT_DIR/run_llm_for_explainability.sh"

echo "========================================"
echo "Full pipeline"
echo "Repository: $REPO_DIR"
echo "========================================"

echo "========================================"
echo "Step 1/3: Preparing selected paths"
echo "Script: $PREPARE_SCRIPT"
echo "========================================"
bash "$PREPARE_SCRIPT"

echo "========================================"
echo "Step 2/3: Running prompt optimization"
echo "Script: $OPTIMIZATION_SCRIPT"
echo "========================================"
bash "$OPTIMIZATION_SCRIPT"

echo "========================================"
echo "Step 3/3: Running explainability evaluation"
echo "Script: $EXPLAINABILITY_SCRIPT"
echo "========================================"
bash "$EXPLAINABILITY_SCRIPT"

echo "========================================"
echo "Pipeline finished successfully."
echo "Selected paths root: $REPO_DIR/../datasets/preselected_explanation_paths"
echo "Optimization output root: $REPO_DIR/out/prompt_optimization"
echo "Explainability output root: $REPO_DIR/out/test_explainability"
echo "========================================"
