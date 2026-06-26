#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PREPARE_SCRIPT="$SCRIPT_DIR/run_prepare_selected_paths.sh"
OPTIMIZATION_SCRIPT="$SCRIPT_DIR/run_llm_for_optimization.sh"
EXPLAINABILITY_SCRIPT="$SCRIPT_DIR/run_llm_for_explainability.sh"
PIPELINE_METRICS=("sep" "etd" "sep_etd_f1")

echo "========================================"
echo "Full pipeline"
echo "Repository: $REPO_DIR"
echo "Metrics order: ${PIPELINE_METRICS[*]}"
echo "========================================"

echo "========================================"
echo "Step 1/3: Preparing selected paths"
echo "Script: $PREPARE_SCRIPT"
echo "========================================"
bash "$PREPARE_SCRIPT"

for metric in "${PIPELINE_METRICS[@]}"; do
  echo "========================================"
  echo "Metric block: $metric"
  echo "========================================"

  echo "========================================"
  echo "Step 2/3: Running prompt optimization"
  echo "Script: $OPTIMIZATION_SCRIPT"
  echo "Metric: $metric"
  echo "========================================"
  METRICS_OVERRIDE="$metric" bash "$OPTIMIZATION_SCRIPT"

  echo "========================================"
  echo "Step 3/3: Running explainability evaluation"
  echo "Script: $EXPLAINABILITY_SCRIPT"
  echo "Metric: $metric"
  echo "========================================"
  METRICS_OVERRIDE="$metric" bash "$EXPLAINABILITY_SCRIPT"
done

echo "========================================"
echo "Pipeline finished successfully."
echo "========================================"
