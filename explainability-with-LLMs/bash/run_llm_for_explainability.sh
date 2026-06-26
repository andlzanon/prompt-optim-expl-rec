#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
source "$SCRIPT_DIR/shared_llm_batch_config.sh"

# Script-specific configuration for explainability runs.
KG_PATH="${DATA_DIR}/knowledge-graphs/props_wikidata_movielens_small.csv"
MODELS=("Llama3.1-I")
METRICS=("sep_etd_f1" "sep" "etd")
INCLUDE_USER_HISTORY="true"
SEP_BETA=0.3

if [ -n "${METRICS_OVERRIDE:-}" ]; then
  IFS=',' read -r -a METRICS <<< "$METRICS_OVERRIDE"
fi

PROMPT_OPT_ROOT="out/prompt_optimization"
TEST_USERS_PATH="$DATA_DIR/user_split_train_val_test/test_users.csv"
TEST_EXPLAINABILITY_ROOT="out/test_explainability"
WITHOUT_OPT_ROOT="${TEST_EXPLAINABILITY_ROOT}/without_optimization"
WITH_OPT_ROOT="${TEST_EXPLAINABILITY_ROOT}/with_optimization"

# Execution mode
RUN_DEFAULT="true"
RUN_OPTIMIZED="true"

# Default run configuration
DEFAULT_MODELS=("${MODELS[@]}")
DEFAULT_METRICS=("${METRICS[@]}")

contains_value() {
  local target="$1"
  shift
  local value

  for value in "$@"; do
    if [ "$value" = "$target" ]; then
      return 0
    fi
  done

  return 1
}

is_selected_model() {
  local model="$1"
  contains_value "$model" "${MODELS[@]}"
}

is_selected_algorithm() {
  local algorithm="$1"
  contains_value "$algorithm" "${ALGORITHMS[@]}"
}

is_selected_metric() {
  local metric="$1"
  contains_value "$metric" "${METRICS[@]}"
}

run_explainability() {
  local model="$1"
  local algorithm="$2"
  local prompt_source="$3"
  local out_dir="$4"
  local metric="$5"
  local best_prompt_path="${6:-}"
  local selected_paths_input_path
  local responses_csv
  local responses_metadata_json
  selected_paths_input_path="$(selected_paths_csv_path "$algorithm" "explainability")"
  responses_csv="${out_dir}/responses.csv"
  responses_metadata_json="${out_dir}/responses_metadata.json"

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

  if [ "$metric" = "sep" ] || [ "$metric" = "sep_etd_f1" ]; then
    cmd+=( --sep_beta "$SEP_BETA" )
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

  if ! prepare_output_slot \
    "explainability" \
    "$out_dir" \
    "$out_dir" \
    "$responses_csv" \
    "$responses_metadata_json"; then
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

echo "Models configured in this script: ${MODELS[*]}"
echo "Algorithms configured in this script: ${ALGORITHMS[*]}"
echo "Metrics configured in this script: ${DEFAULT_METRICS[*]}"

if [ "${#BEST_PROMPT_FILES[@]}" -eq 0 ]; then
  echo "No best_prompt.json files found under ${PROMPT_OPT_ROOT}."
  echo "Only fallback default runs will be available."
fi

if [ "$RUN_DEFAULT" = "true" ]; then
  if [ "${#DEFAULT_MODELS[@]}" -eq 0 ]; then
    echo "MODELS is empty. Cannot run the default explainability pass."
    exit 1
  fi

  if [ "${#ALGORITHMS[@]}" -eq 0 ]; then
    echo "ALGORITHMS is empty. Cannot run the default explainability pass."
    exit 1
  fi

  if [ "${#DEFAULT_METRICS[@]}" -eq 0 ]; then
    echo "METRICS is empty. Cannot run the default explainability pass."
    exit 1
  fi

  # The built-in prompt must also be evaluated per algorithm because the
  # explanation paths and recommendation context depend on the recommender.
  for model in "${DEFAULT_MODELS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
      for metric in "${DEFAULT_METRICS[@]}"; do
        out_dir="${WITHOUT_OPT_ROOT}/${model}/${algorithm}/${metric}"
        run_explainability "$model" "$algorithm" "default" "$out_dir" "$metric"
      done
    done
  done
fi

if [ "$RUN_OPTIMIZED" = "true" ]; then
  for best_prompt_path in "${BEST_PROMPT_FILES[@]}"; do
    rel_path="${best_prompt_path#${PROMPT_OPT_ROOT}/}"
    IFS='/' read -r model algorithm metric repr_dir early_dir lambda_dir pool_dir _rest <<< "$rel_path"

    if ! is_selected_model "$model"; then
      echo "Skipping optimized run outside configured model list: ${model}"
      continue
    fi

    if ! is_selected_algorithm "$algorithm"; then
      echo "Skipping optimized run outside configured algorithm list: ${algorithm}"
      continue
    fi

    if ! is_selected_metric "$metric"; then
      echo "Skipping optimized run outside configured metric list: ${metric}"
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
