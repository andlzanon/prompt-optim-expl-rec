from src.utils.args import args_llm
from src.llm.llm_for_explainability import LLM
from src.utils.geral import (
    build_metric_fn,
    explanations_df_to_blocks,
    load_best_prompt,
    load_required_selected_paths_csv,
    prepare_explainability_inputs,
    save_explanations_csv,
    save_metadata_json,
)

import time
import pandas as pd

if __name__ == "__main__":
    """
    Entry-point script for the LLM-based explainability generation workflow.

    This script parses runtime arguments, prepares the interaction data used to
    construct prompts, loads the configured LLM wrapper, generates one explanation
    path selection per recommendation, and persists both the explanations and the
    run metadata to disk.
    """

    args, info = args_llm()

    # Load the interaction data and the users that will be processed.
    interactions_df, users = prepare_explainability_inputs(args)
    selected_paths_input_df = load_required_selected_paths_csv(
        args.selected_paths_input_path
    )
    
    props_df = pd.read_csv(args.kg_path)
    metric_name, metric_fn = build_metric_fn(
        metric=args.metric,
        metric_params=args.metric_params,
        props_df=props_df,
    )

    # Initialize and load the LLM wrapper before generating explanations.
    llm = LLM(llm_method=args.llm_method, seed=args.seed)
    llm.set_model(metric=args.metric)

    if args.prompt_source == "best_prompt":
        best_prompt_payload = load_best_prompt(args.best_prompt_path)
        llm.system_prompt = best_prompt_payload["best_prompt"]
        llm.prompt = [{"role": "system", "content": llm.system_prompt}]

        info["prompt_source"] = "best_prompt"
        info["best_prompt_path"] = args.best_prompt_path
        info["best_prompt_model"] = best_prompt_payload.get(
            "model_that_generated_the_prompt"
        )
    else:
        info["prompt_source"] = "default"
        info["best_prompt_path"] = None

    # Measure the end-to-end time spent generating explanation selections.
    start_time = time.time()

    user_explanations = llm.generate_explanations(
        users=users,
        interactions_df=interactions_df,
        explanation_paths_prefix=args.explanation_paths_prefix,
        selection_strategy=args.selection_strategy,
        num_recommendations=args.num_recommendations,
        num_paths_per_recommendation=args.num_paths_per_recommendation,
        selected_paths_df=selected_paths_input_df,
        include_user_history=args.include_user_history 
    )

    end_time = time.time()
    explanation_blocks = explanations_df_to_blocks(user_explanations)
    metric_payload = metric_fn(explanation_blocks)
    metric_value = float(metric_payload["objective_value"])
    metric_scores = metric_payload["scores"]

    # Record summary metadata about the run for later inspection.
    info["time_to_explain"] = float(end_time - start_time)
    info["system_prompt"] = llm.system_prompt
    info["n_users"] = len(users)
    info["users_path"] = args.test_users_path
    info["metric"] = args.metric
    info["metric_name"] = metric_name
    info["metric_value"] = metric_value
    info["metric_scores"] = metric_scores
    info["sep_value"] = float(metric_scores["sep"])
    info["etd_value"] = float(metric_scores["etd"])
    info["sep_etd_f1_value"] = float(metric_scores["sep_etd_f1"])
    info["metric_params"] = args.metric_params
    info["kg_path"] = args.kg_path
    info["selection_strategy"] = args.selection_strategy
    info["selected_paths_input_path"] = args.selected_paths_input_path

    # Persist metadata and generated explanations as separate artifacts.
    output_json = args.outfilename + "_metadata.json"
    save_metadata_json(output_json, info)

    output_csv = args.outfilename + ".csv"
    save_explanations_csv(output_csv, user_explanations)

    print(f"\nExplanations saved to {output_csv}")
    print(f"{metric_name}={metric_value:.6f}")
    print(
        "Score breakdown: "
        f"SEP={metric_scores['sep']:.6f}, "
        f"ETD={metric_scores['etd']:.6f}, "
        f"SEP_ETD_F1={metric_scores['sep_etd_f1']:.6f}"
    )
    print(f"Metadata saved to {output_json}")
