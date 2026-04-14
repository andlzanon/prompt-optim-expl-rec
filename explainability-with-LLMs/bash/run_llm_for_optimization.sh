#!/bin/bash

# Paths
DATA_DIR="../datasets"
KG_PATH="../datasets/knowledge-graphs/props_wikidata_movielens_small.csv"

# Models & Algorithms
MODELS=("Llama3.1-I")
ALGORITHMS=("user_knn" "item_knn" "ncf" "bprmf")   # ("user_knn" "item_knn" "ncf" "bprmf")

# Representation models
# REPRESENTATION_MODELS=("sbert")
REPRESENTATION_MODELS=("llm2vec" "sbert")

# Explainability settings
SELECTION_STRATEGY="random"
NUM_RECOMMENDATIONS=10
NUM_PATHS_PER_RECOMMENDATION=10
INCLUDE_USER_HISTORY="true"

# Optimization settings
SEED=2026
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

# Metrics
METRICS=("sep") # ("etd" "sep")

SEP_BETA=0.3
ETD_K=5

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

              echo "========================================"
              echo "Model: $model"
              echo "Representation model: $representation_model"
              echo "Algorithm: $algorithm"
              echo "Metric: $metric"
              echo "Early stopping: $early_stopping"
              echo "MMR lambda quality: $mmr_lambda_quality"
              echo "MMR pool multiplier: $mmr_pool_multiplier"
              echo "Output directory: $OUT_DIR"
              echo "========================================"

              COMMON_ARGS=(
                --datain "$DATA_DIR"
                --kg_path "$KG_PATH"

                # explainability args
                --algorithm "$algorithm"
                --selection_strategy "$SELECTION_STRATEGY"
                --num_recommendations "$NUM_RECOMMENDATIONS"
                --num_paths_per_recommendation "$NUM_PATHS_PER_RECOMMENDATION"
                --include_user_history "$INCLUDE_USER_HISTORY"

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
              if [ "$metric" = "sep" ]; then
                COMMON_ARGS+=( --sep_beta "$SEP_BETA" )
              elif [ "$metric" = "etd" ]; then
                COMMON_ARGS+=( --etd_k "$ETD_K" )
              fi

              python3.10 run_prompt_optimizer.py "${COMMON_ARGS[@]}"

            done
          done
        done
      done
    done
  done
done
