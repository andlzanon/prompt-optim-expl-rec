from src.utils.args import args_prepare_selected_paths
from src.llm.llm_for_explainability import LLM
from src.utils.geral import (
    prepare_selected_paths_users,
    save_metadata_json,
    save_selected_paths_csv,
)

import time

if __name__ == "__main__":
    """
    Precompute and persist the candidate explanation paths used by later runs.

    This script lets the user sample the candidate paths once and reuse the
    resulting CSV in either prompt optimization or explainability generation.
    """

    args, info = args_prepare_selected_paths()

    user_data = prepare_selected_paths_users(args, user_scope=args.user_scope)
    users = user_data["users"]

    llm = LLM(seed=args.seed)

    start_time = time.time()
    selected_paths_df = llm.prepare_selected_paths(
        users=users,
        explanation_paths_prefix=args.explanation_paths_prefix,
        selection_strategy=args.selection_strategy,
        num_recommendations=args.num_recommendations,
        num_paths_per_recommendation=args.num_paths_per_recommendation,
    )
    end_time = time.time()

    save_selected_paths_csv(args.selected_paths_output_path, selected_paths_df)

    info["user_scope"] = user_data["user_scope"]
    info["split_dir"] = user_data["split_dir"]
    info["n_users"] = len(users)
    info["selection_strategy"] = args.selection_strategy
    info["selected_paths_output_path"] = args.selected_paths_output_path
    info["time_to_prepare_selected_paths"] = float(end_time - start_time)

    output_json = args.outfilename + "_metadata.json"
    save_metadata_json(output_json, info)

    print(f"\nSelected candidate paths saved to {args.selected_paths_output_path}")
    print(f"Metadata saved to {output_json}")
