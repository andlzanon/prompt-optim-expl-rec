from src.utils.args import args_llm
from src.llm.llm_for_explainability import LLM
from src.utils.geral import prepare_explainability_inputs, save_metadata_json, save_explanations_csv

import time

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

    # Optional debug slice for running the pipeline on only a subset of users.
    # interactions_df = interactions_df[interactions_df["userId"].isin(users[:20])].reset_index(drop=True)
    # users = interactions_df["userId"].unique().tolist()

    # Initialize and load the LLM wrapper before generating explanations.
    llm = LLM(llm_method=args.llm_method, seed=args.seed)
    llm.set_model()

    # Measure the end-to-end time spent generating explanation selections.
    start_time = time.time()

    user_explanations = llm.generate_explanations(
        users=users,
        interactions_df=interactions_df,
        explanation_paths_prefix=args.explanation_paths_prefix,
        num_recommendations=args.num_recommendations,
        num_paths_per_recommendation=args.num_paths_per_recommendation,
        include_user_history=args.include_user_history 
    )

    end_time = time.time()

    # Record summary metadata about the run for later inspection.
    info["time_to_explain"] = float(end_time - start_time)
    info["system_prompt"] = llm.system_prompt
    info["n_users"] = len(users)

    # Persist metadata and generated explanations as separate artifacts.
    output_json = args.outfilename + "_metadata.json"
    save_metadata_json(output_json, info)

    output_csv = args.outfilename + ".csv"
    save_explanations_csv(output_csv, user_explanations)

    print(f"\nExplanations saved to {output_csv}")
    print(f"Metadata saved to {output_json}")