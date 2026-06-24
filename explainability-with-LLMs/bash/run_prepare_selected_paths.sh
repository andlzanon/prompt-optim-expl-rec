#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
source "$SCRIPT_DIR/shared_llm_batch_config.sh"

# User scopes:
# - "optimization" -> train + val users
# - "explainability" -> test users
# - "all" -> train + val + test users
USER_SCOPES=("optimization" "explainability")

run_prepare_selected_paths() {
  local algorithm="$1"
  local user_scope="$2"

  local out_dir
  local selected_paths_csv
  local selected_paths_metadata_json
  out_dir="$(selected_paths_dir "$algorithm" "$user_scope")"
  selected_paths_csv="$(selected_paths_csv_path "$algorithm" "$user_scope")"
  selected_paths_metadata_json="$(selected_paths_metadata_path "$algorithm" "$user_scope")"

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

  if ! prepare_output_slot \
    "selected_paths" \
    "$out_dir" \
    "$out_dir" \
    "$selected_paths_csv" \
    "$selected_paths_metadata_json"; then
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
    --out "$SELECTED_PATHS_ROOT"
}

echo "Algorithms configured in this batch: ${ALGORITHMS[*]}"
echo "User scopes configured in this batch: ${USER_SCOPES[*]}"

for algorithm in "${ALGORITHMS[@]}"; do
  for user_scope in "${USER_SCOPES[@]}"; do
    run_prepare_selected_paths "$algorithm" "$user_scope"
  done
done
