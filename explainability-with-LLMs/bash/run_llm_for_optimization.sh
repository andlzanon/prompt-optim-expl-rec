#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
source "$SCRIPT_DIR/shared_llm_batch_config.sh"

# Script-specific configuration for optimization runs.
KG_PATH="${DATA_DIR}/knowledge-graphs/props_wikidata_movielens_small.csv"
MODELS=("Llama3.1-I")
METRICS=("sep_etd_f1" "sep" "etd")
INCLUDE_USER_HISTORY="true"
SEP_BETA=0.3

if [ -n "${METRICS_OVERRIDE:-}" ]; then
  IFS=',' read -r -a METRICS <<< "$METRICS_OVERRIDE"
fi

# Representation models
# REPRESENTATION_MODELS=("sbert")
REPRESENTATION_MODELS=("llm2vec" "sbert")
EPOCHS=10
TOTAL_INSTRUCTIONS_PER_ITERATION=1
META_PROMPT_INSTRUCTION_QUANTITY=3

# Validation / Early stopping
EVAL_EVERY=1
PATIENCE=3
MIN_DELTA=0.03
EARLY_STOPPING_VALUES=("false")

# MMR settings
# MMR_LAMBDA_QUALITIES=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
MMR_LAMBDA_QUALITIES=("0.0" "0.5" "1.0")
MMR_POOL_MULTIPLIERS=(10)

# Main Loop
for model in "${MODELS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    for representation_model in "${REPRESENTATION_MODELS[@]}"; do
      for metric in "${METRICS[@]}"; do
        for early_stopping in "${EARLY_STOPPING_VALUES[@]}"; do
          for mmr_lambda_quality in "${MMR_LAMBDA_QUALITIES[@]}"; do
            for mmr_pool_multiplier in "${MMR_POOL_MULTIPLIERS[@]}"; do

              lambda_tag="${mmr_lambda_quality/./_}"
              OUT_DIR="out/prompt_optimization/${model}/${algorithm}/${metric}/repr_${representation_model}/early_${early_stopping}/mmr_lambda_${lambda_tag}/mmr_pool_${mmr_pool_multiplier}"
              SELECTED_PATHS_INPUT_PATH="$(selected_paths_csv_path "$algorithm" "optimization")"
              FINAL_OUTPUT_DIR="${OUT_DIR}/${model}/prompt_opt/${metric}"
              BEST_PROMPT_PATH="${FINAL_OUTPUT_DIR}/best_prompt.json"
              OPTIMIZATION_METADATA_PATH="${FINAL_OUTPUT_DIR}/optimization_process_metadata.json"

              echo "========================================"
              echo "Model: $model"
              echo "Representation model: $representation_model"
              echo "Algorithm: $algorithm"
              echo "Metric: $metric"
              echo "Early stopping: $early_stopping"
              echo "MMR lambda quality: $mmr_lambda_quality"
              echo "MMR pool multiplier: $mmr_pool_multiplier"
              echo "Output directory: $OUT_DIR"
              echo "Selected paths input: $SELECTED_PATHS_INPUT_PATH"
              echo "========================================"

              if [ ! -f "$SELECTED_PATHS_INPUT_PATH" ]; then
                echo "Missing selected paths file: $SELECTED_PATHS_INPUT_PATH"
                echo "Run bash/run_prepare_selected_paths.sh before optimization."
                exit 1
              fi

              if ! prepare_output_slot \
                "optimization" \
                "$FINAL_OUTPUT_DIR" \
                "$FINAL_OUTPUT_DIR" \
                "$BEST_PROMPT_PATH" \
                "$OPTIMIZATION_METADATA_PATH"; then
                continue
              fi

              COMMON_ARGS=(
                --datain "$DATA_DIR"
                --kg_path "$KG_PATH"

                # explainability args
                --algorithm "$algorithm"
                --selection_strategy "$SELECTION_STRATEGY"
                --num_recommendations "$NUM_RECOMMENDATIONS"
                --num_paths_per_recommendation "$NUM_PATHS_PER_RECOMMENDATION"
                --include_user_history "$INCLUDE_USER_HISTORY"
                --selected_paths_input_path "$SELECTED_PATHS_INPUT_PATH"

                # seed/model
                --seed "$SEED"
                --llm_method "$model"
                --representation_model "$representation_model"

                # optimization args
                --epochs "$EPOCHS"
                --total_instructions_per_iteration "$TOTAL_INSTRUCTIONS_PER_ITERATION"
                --meta_prompt_instruction_quantity "$META_PROMPT_INSTRUCTION_QUANTITY"
                --mmr_lambda_quality "$mmr_lambda_quality"
                --mmr_pool_multiplier "$mmr_pool_multiplier"

                # validation control
                --eval_every "$EVAL_EVERY"
                --patience "$PATIENCE"
                --min_delta "$MIN_DELTA"
                --early_stopping "$early_stopping"

                # metric + output
                --metric "$metric"
                --out "$OUT_DIR"
              )

              # Metric-specific parameter
              if [ "$metric" = "sep" ] || [ "$metric" = "sep_etd_f1" ]; then
                COMMON_ARGS+=( --sep_beta "$SEP_BETA" )
              fi

              python3.10 run_prompt_optimizer.py "${COMMON_ARGS[@]}"

            done
          done
        done
      done
    done
  done
done
