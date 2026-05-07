from src.utils.args import args_prompt_optimizer
from src.llm.llm_for_prompt_optimization import PromptOptimizer
from src.llm.llm_for_explainability import LLM
from src.utils.geral import (
    prepare_optimization_inputs,
    save_metadata_json,
    save_best_prompt,
    build_metric_fn,
)

import time
import pandas as pd

if __name__ == "__main__":
    """
    Entry-point script for the prompt-optimization workflow.

    This script parses optimization arguments, prepares the datasets and metadata
    required for training and validation evaluation, loads the LLM used for
    explanation-path selection, runs the prompt-optimization loop, and persists
    both the optimization metadata and the best prompt found during the run.
    """

    args, info = args_prompt_optimizer()
    selected_paths_input_df = (
        pd.read_csv(args.selected_paths_input_path)
        if args.selected_paths_input_path
        else None
    )

    # Load the datasets and auxiliary data required by the optimization flow.
    data = prepare_optimization_inputs(args)
    props_df = data["props_df"]

    # Build the metric callable used to score generated explanations.
    metric_name, metric_fn = build_metric_fn(
        metric=args.metric,
        metric_params=args.metric_params,
        props_df=props_df,
    )

    # Initialize and load the LLM used for explanation-path selection.
    llm = LLM(llm_method=args.llm_method, seed=args.seed)
    llm.set_model()

    # Record the initial fixed system prompt and the default user-guidance
    # block before optimization starts.
    info["baseline_prompt"] = llm.metric_selection_guidance
    info["baseline_system_prompt"] = llm.system_prompt
    info["baseline_metric_selection_guidance"] = llm.metric_selection_guidance
    info["selected_paths_input_path"] = args.selected_paths_input_path

    # Configure the optimizer controller with the requested runtime settings.
    prompt_optimizer = PromptOptimizer(
        epochs=args.epochs,
        # total_instructions_per_iteration=args.total_instructions_per_iteration,
        meta_prompt_instruction_quantity=args.meta_prompt_instruction_quantity,
        eval_every=args.eval_every,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stopping=args.early_stopping,
        save_dir=args.outputdir,
        mmr_lambda_quality=args.mmr_lambda_quality,
        mmr_pool_multiplier=args.mmr_pool_multiplier,
        representation_model=args.representation_model,
    )

    print(f"Optimization_process! metric={metric_name}")
    optimization_process_start_time = time.time()

    # Pack the explanation-generation arguments reused at each epoch.
    explain_kwargs = dict(
        explanation_paths_prefix=args.explanation_paths_prefix,
        selection_strategy=args.selection_strategy,
        num_recommendations=args.num_recommendations,
        num_paths_per_recommendation=args.num_paths_per_recommendation,
        selected_paths_df=selected_paths_input_df,
        include_user_history=args.include_user_history,
    )

    info_process, ranked = prompt_optimizer.run_optimize_process(
        llm=llm,
        metric_fn=metric_fn,
        train_user_ids=data["train_user_ids"],
        val_user_ids=data["val_user_ids"],
        interactions_df_train=data["interactions_df_train"],
        interactions_df_val=data["interactions_df_val"],
        explain_kwargs=explain_kwargs,
    )

    optimization_process_end_time = time.time()
    info_process["time_prompt_optimization"] = (
        optimization_process_end_time - optimization_process_start_time
    )

    best_prompt = info_process["best_on_train"]["best_train_prompt"]

    # Persist the best prompt and the full optimization metadata.
    save_best_prompt(args.outputdir, best_prompt, args.llm_method)

    info = info | info_process
    output_json = args.outfilename + "_metadata.json"
    save_metadata_json(output_json, info)
    print(f"Metadata saved to {output_json}")
