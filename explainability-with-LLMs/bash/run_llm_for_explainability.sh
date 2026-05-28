#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="../datasets"
KG_PATH="../datasets/knowledge-graphs/props_wikidata_movielens_small.csv"
SELECTED_PATHS_ROOT="../datasets/preselected_explanation_paths"
PROMPT_OPT_ROOT="out/prompt_optimization"
TEST_USERS_PATH="$DATA_DIR/user_split_train_val_test/test_users.csv"
TEST_EXPLAINABILITY_ROOT="out/test_explainability"
WITHOUT_OPT_ROOT="${TEST_EXPLAINABILITY_ROOT}/without_optimization"
WITH_OPT_ROOT="${TEST_EXPLAINABILITY_ROOT}/with_optimization"

# Execution mode
RUN_DEFAULT="true"
RUN_OPTIMIZED="true"

# Algorithms covered by the explainability batch
ALGORITHMS=("user_knn" "item_knn" "ncf" "bprmf")

# Default run configuration
DEFAULT_MODEL="Llama3.1-I"
DEFAULT_METRIC="sep"

# Explainability settings
SELECTION_STRATEGY="random"
NUM_RECOMMENDATIONS=10
NUM_PATHS_PER_RECOMMENDATION=10
INCLUDE_USER_HISTORY="true"
SEED=2026
SEP_BETA=0.3
ETD_K=5

is_selected_algorithm() {
  local algorithm="$1"

  for selected_algorithm in "${ALGORITHMS[@]}"; do
    if [ "$selected_algorithm" = "$algorithm" ]; then
      return 0
    fi
  done

  return 1
}

run_explainability() {
  local model="$1"
  local algorithm="$2"
  local prompt_source="$3"
  local out_dir="$4"
  local metric="$5"
  local best_prompt_path="${6:-}"
  local selected_paths_input_path="${SELECTED_PATHS_ROOT}/${algorithm}/explainability/${SELECTION_STRATEGY}/recs_${NUM_RECOMMENDATIONS}_paths_${NUM_PATHS_PER_RECOMMENDATION}/seed_${SEED}/selected_paths.csv"

  local cmd=(
    python3.10 run_llm_explainability.py
    --datain "$DATA_DIR"
    --kg_path "$KG_PATH"
    --algorithm "$algorithm"
    --llm_method "$model"
    --selection_strategy "$SELECTION_STRATEGY"
    --num_recommendations "$NUM_RECOMMENDATIONS"
    --num_paths_per_recommendation "$NUM_PATHS_PER_RECOMMENDATION"
    --include_user_history "$INCLUDE_USER_HISTORY"
    --selected_paths_input_path "$selected_paths_input_path"
    --prompt_source "$prompt_source"
    --metric "$metric"
    --out "$out_dir"
  )

  if [ "$metric" = "sep" ]; then
    cmd+=( --sep_beta "$SEP_BETA" )
  elif [ "$metric" = "etd" ]; then
    cmd+=( --etd_k "$ETD_K" )
  fi

  if [ "$prompt_source" = "best_prompt" ]; then
    cmd+=( --best_prompt_path "$best_prompt_path" )
  fi

  echo "========================================"
  echo "Running model=${model}"
  echo "Algorithm=${algorithm}"
  echo "Prompt source=${prompt_source}"
  echo "Test users file=${TEST_USERS_PATH}"
  echo "Metric=${metric}"
  echo "Selection strategy=${SELECTION_STRATEGY}"
  echo "Num recommendations=${NUM_RECOMMENDATIONS}"
  echo "Num paths per recommendation=${NUM_PATHS_PER_RECOMMENDATION}"
  echo "Include user history=${INCLUDE_USER_HISTORY}"
  echo "Selected paths input=${selected_paths_input_path}"
  echo "Best prompt path=${best_prompt_path:--}"
  echo "Output directory=${out_dir}"
  echo "========================================"

  if compgen -G "${out_dir}/responses*" > /dev/null; then
    echo "Skipping because output already exists under ${out_dir}"
    return 0
  fi

  if [ ! -f "$selected_paths_input_path" ]; then
    echo "Missing selected paths file: $selected_paths_input_path"
    echo "Run bash/run_prepare_selected_paths.sh before explainability."
    exit 1
  fi

  "${cmd[@]}"
}

mapfile -t BEST_PROMPT_FILES < <(find "$PROMPT_OPT_ROOT" -path '*/best_prompt.json' | sort)

echo "Algorithms configured in this batch: ${ALGORITHMS[*]}"

if [ "${#BEST_PROMPT_FILES[@]}" -eq 0 ]; then
  echo "No best_prompt.json files found under ${PROMPT_OPT_ROOT}."
  echo "Only fallback default runs will be available."
fi

if [ "$RUN_DEFAULT" = "true" ]; then
  if [ "${#ALGORITHMS[@]}" -eq 0 ]; then
    echo "ALGORITHMS is empty. Cannot run the default explainability pass."
    exit 1
  fi

  # The built-in prompt must also be evaluated per algorithm because the
  # explanation paths and recommendation context depend on the recommender.
  for algorithm in "${ALGORITHMS[@]}"; do
    out_dir="${WITHOUT_OPT_ROOT}/${DEFAULT_MODEL}/${algorithm}/${DEFAULT_METRIC}"
    run_explainability "$DEFAULT_MODEL" "$algorithm" "default" "$out_dir" "$DEFAULT_METRIC"
  done
fi

if [ "$RUN_OPTIMIZED" = "true" ]; then
  for best_prompt_path in "${BEST_PROMPT_FILES[@]}"; do
    rel_path="${best_prompt_path#${PROMPT_OPT_ROOT}/}"
    IFS='/' read -r model algorithm metric repr_dir early_dir lambda_dir pool_dir _rest <<< "$rel_path"

    if ! is_selected_algorithm "$algorithm"; then
      echo "Skipping optimized run outside configured algorithm list: ${algorithm}"
      continue
    fi

    out_dir="${WITH_OPT_ROOT}/${model}/${algorithm}/${metric}/${repr_dir}/${early_dir}/${lambda_dir}/${pool_dir}"

    run_explainability \
      "$model" \
      "$algorithm" \
      "best_prompt" \
      "$out_dir" \
      "$metric" \
      "$best_prompt_path"
  done
fi
