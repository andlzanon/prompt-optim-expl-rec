#!/bin/bash

DATA_DIR="../datasets"

MODELS=("Llama3.1-I")
ALGORITHMS=("user_knn")  # item_knn)

# Hyperparameters
SELECTION_STRATEGY="random"
NUM_RECOMMENDATIONS=10
NUM_PATHS_PER_RECOMMENDATION=20
INCLUDE_USER_HISTORY="true"

for model in "${MODELS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do

    OUT_DIR="out/explainability/${model}/${algorithm}"

    echo "========================================"
    echo "Running model=${model}"
    echo "Algorithm=${algorithm}"
    echo "Selection strategy=${SELECTION_STRATEGY}"
    echo "Num recommendations=${NUM_RECOMMENDATIONS}"
    echo "Num paths per recommendation=${NUM_PATHS_PER_RECOMMENDATION}"
    echo "Include user history=${INCLUDE_USER_HISTORY}"
    echo "Output directory: ${OUT_DIR}"
    echo "========================================"

    python3.10 run_llm_explainability.py \
      --datain "$DATA_DIR" \
      --algorithm "$algorithm" \
      --llm_method "$model" \
      --selection_strategy "$SELECTION_STRATEGY" \
      --num_recommendations "$NUM_RECOMMENDATIONS" \
      --num_paths_per_recommendation "$NUM_PATHS_PER_RECOMMENDATION" \
      --include_user_history "$INCLUDE_USER_HISTORY" \
      --out "$OUT_DIR"

  done
done