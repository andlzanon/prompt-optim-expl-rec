#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="../datasets"
SELECTED_PATHS_ROOT="../datasets/preselected_explanation_paths"

# Algorithms covered by the preselection batch
ALGORITHMS=("user_knn" "item_knn" "ncf" "bprmf")

# User scopes:
# - "optimization" -> train + val users
# - "explainability" -> test users
# - "all" -> train + val + test users
USER_SCOPES=("optimization" "explainability")

# Path-selection settings
SELECTION_STRATEGY="random"
NUM_RECOMMENDATIONS=10
NUM_PATHS_PER_RECOMMENDATION=10
SEED=2026

run_prepare_selected_paths() {
  local algorithm="$1"
  local user_scope="$2"
  local out_root="$3"

  local out_dir="${out_root}/${algorithm}/${user_scope}/${SELECTION_STRATEGY}/recs_${NUM_RECOMMENDATIONS}_paths_${NUM_PATHS_PER_RECOMMENDATION}/seed_${SEED}"

  echo "========================================"
  echo "Preparing selected candidate paths"
  echo "Algorithm=${algorithm}"
  echo "User scope=${user_scope}"
  echo "Selection strategy=${SELECTION_STRATEGY}"
  echo "Num recommendations=${NUM_RECOMMENDATIONS}"
  echo "Num paths per recommendation=${NUM_PATHS_PER_RECOMMENDATION}"
  echo "Seed=${SEED}"
  echo "Output directory=${out_dir}"
  echo "========================================"

  if compgen -G "${out_dir}/selected_paths*" > /dev/null; then
    echo "Skipping because selected paths already exist under ${out_dir}"
    return 0
  fi

  python3.10 run_prepare_selected_paths.py \
    --datain "$DATA_DIR" \
    --algorithm "$algorithm" \
    --selection_strategy "$SELECTION_STRATEGY" \
    --num_recommendations "$NUM_RECOMMENDATIONS" \
    --num_paths_per_recommendation "$NUM_PATHS_PER_RECOMMENDATION" \
    --user_scope "$user_scope" \
    --seed "$SEED" \
    --out "$out_root"
}

echo "Algorithms configured in this batch: ${ALGORITHMS[*]}"
echo "User scopes configured in this batch: ${USER_SCOPES[*]}"

for algorithm in "${ALGORITHMS[@]}"; do
  for user_scope in "${USER_SCOPES[@]}"; do
    run_prepare_selected_paths "$algorithm" "$user_scope" "$SELECTED_PATHS_ROOT"
  done
done
