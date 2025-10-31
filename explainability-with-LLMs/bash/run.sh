#!/bin/bash

LLM_DIR="/home/$USER/OTIMIAZAO_RECOMENDACAO/pos_WebMedia/explainability_with_LLMs"
cd "$LLM_DIR"

DATA_DIR="resources/data"

for model in Llama3.1-I; do

    RECOMMENDATIONS_DIR="resources/out/recommendations/${model}/responses.csv"

    OUT_DIR="resources/out/explainability/${model}"

    python3.10 run_llm_explainability.py \
        --datain "$DATA_DIR" \
        --inputdir_recommendation "$RECOMMENDATIONS_DIR" \
        --llm_method "$model" \
        --out "$OUT_DIR"

done