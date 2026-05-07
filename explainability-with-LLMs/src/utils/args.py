from __future__ import annotations

from src.representation import available_representations
from src.utils.geral import check_if_out_file_exists

from datetime import datetime
from typing import Tuple, Dict, Any
import numpy as np
import os, torch, random, socket, argparse

def args_llm() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Build and parse CLI arguments for the LLM explanation-path selection runner.

    This function defines the command-line interface used by the module that
    generates LLM-based explainability outputs from precomputed explanation
    paths. After parsing the raw CLI values, it enriches the resulting
    namespace with derived paths, a timestamp, and normalized boolean flags.

    Parameters
    ----------
    None
        The function does not receive Python-level arguments. It reads command-
        line values from ``sys.argv`` through ``argparse.ArgumentParser``.

    Returns
    -------
    Tuple[argparse.Namespace, Dict[str, Any]]
        A tuple ``(args, info)`` where ``args`` is the parsed namespace with
        additional derived attributes such as ``inputdir``, ``outputdir``,
        ``outfilename``, ``start_time``, ``include_user_history``, and
        ``explanation_paths_prefix``. ``info`` contains ``{"args": vars(args)}``
        so the final configuration can be logged downstream.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` when required arguments are missing or invalid.
    OSError
        Raised if the output directory needs to be created and the filesystem
        operation fails.

    Side Effects
    ------------
    Prints the parsed arguments and status messages, calls
    ``check_if_out_file_exists(args)``, creates the output directory when it
    does not exist yet, and seeds Python's ``random``, NumPy, and PyTorch
    random number generators.

    Notes
    -----
    This function centralizes the runtime configuration for the explanation
    generation flow. The derived ``explanation_paths_prefix`` is the base path
    later used to locate the explanation-path files associated with the chosen
    recommendation algorithm.
    """

    parser = argparse.ArgumentParser(
        description="LLM explainability (explanation-path selection)."
    )

    # Input
    parser.add_argument(
        "--datain",
        type=str,
        required=True,
        help="Base input data directory (e.g., ../datasets).",
    )

    # Explanation paths
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm name used to locate explanation paths (e.g., user_knn, item_knn).",
    )

    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="random",
        help="Strategy used to select explanation paths (default: random).",
    )

    parser.add_argument(
        "--num_recommendations",
        type=int,
        default=3,
        help="Number of recommendations to include per user (default: 3).",
    )

    parser.add_argument(
        "--num_paths_per_recommendation",
        type=int,
        default=3,
        help="Number of candidate explanation paths per recommendation (default: 3).",
    )

    parser.add_argument(
        "--include_user_history",
        type=str,
        default="true",
        help="Whether to include user interaction history in the prompt (true/false).",
    )

    parser.add_argument(
        "--selected_paths_input_path",
        type=str,
        default=None,
        help=(
            "Optional CSV path with precomputed candidate paths. When provided, "
            "the explainability run reuses this file instead of sampling paths "
            "again."
        ),
    )

    # Seed and model
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for reproducibility.",
    )

    parser.add_argument(
        "--llm_method",
        type=str,
        required=True,
        help="LLM method or model name.",
    )

    parser.add_argument(
        "--prompt_source",
        type=str,
        default="default",
        choices=["default", "best_prompt"],
        help=(
            "Source used to populate the system prompt. "
            "'default' keeps the built-in prompt and 'best_prompt' loads the "
            "prompt stored in a best_prompt.json file."
        ),
    )

    parser.add_argument(
        "--best_prompt_path",
        type=str,
        default=None,
        help="Path to the best_prompt.json file used when --prompt_source best_prompt.",
    )

    parser.add_argument(
        "--kg_path",
        type=str,
        default=None,
        help=(
            "Path to the knowledge-graph properties file used when computing "
            "SEP or ETD. Defaults to <datain>/knowledge-graphs/"
            "props_wikidata_movielens_small.csv."
        ),
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["sep", "etd"],
        default="sep",
        help="Metric used to score the generated explanations (default: sep).",
    )

    parser.add_argument(
        "--sep_beta",
        type=float,
        default=0.3,
        help="(SEP only) Exponential decay parameter beta.",
    )

    parser.add_argument(
        "--etd_k",
        type=int,
        default=5,
        help="(ETD only) Number of explanations (k) considered.",
    )

    # Output
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Base output directory.",
    )

    # Machine / environment
    parser.add_argument(
        "--machine",
        type=str,
        default=socket.gethostname(),
        help="Machine hostname.",
    )

    args = parser.parse_args()

    # Post-processing
    args.inputdir = f"{args.datain}"
    args.outputdir = f"{args.out}"
    args.outfilename = f"{args.outputdir}/responses"
    args.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Normalize CLI string values into booleans expected by the downstream flow.
    args.include_user_history = str(args.include_user_history).lower() in ["true", "1", "yes"]
    args.test_users_path = os.path.join(
        args.datain,
        "user_split_train_val_test",
        "test_users.csv",
    )
    if args.kg_path is None:
        args.kg_path = os.path.join(
            args.datain,
            "knowledge-graphs",
            "props_wikidata_movielens_small.csv",
        )

    if args.prompt_source == "best_prompt" and not args.best_prompt_path:
        parser.error("--best_prompt_path is required when --prompt_source best_prompt.")

    if args.metric == "sep":
        args.metric_params = {"beta": float(args.sep_beta)}
    else:
        args.metric_params = {"k": int(args.etd_k)}

    # Build the shared prefix used to locate the explanation-path files.
    args.explanation_paths_prefix = os.path.join(
        args.datain,
        "explanation_paths",
        f"{args.algorithm}-opt",
        args.algorithm,
    )
    args.selected_paths_output_dir = os.path.join(
        args.outputdir,
        "selected_paths",
        args.selection_strategy,
        f"seed_{args.seed}",
    )
    args.selected_paths_output_path = os.path.join(
        args.selected_paths_output_dir,
        "selected_paths.csv",
    )

    print('\n', args)
    check_if_out_file_exists(args)

    if not os.path.exists(args.outputdir):
        print(f"\nCreating output directory at {args.outputdir}\n")
        os.makedirs(args.outputdir, exist_ok=True)
    else: 
        print('\n')

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    info = {"args": vars(args)}

    return args, info

def args_prompt_optimizer() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Build and parse CLI arguments for the prompt-optimization runner.

    This function defines the command-line interface for the prompt
    optimization workflow, which searches for better system instructions for
    the explainability pipeline. Besides parsing the raw CLI values, it derives
    convenience attributes used later in the optimization process, such as the
    explanation-path prefix, output paths, metric-specific parameters, and
    normalized boolean flags.

    Parameters
    ----------
    None
        The function does not receive Python-level arguments. It reads command-
        line values from ``sys.argv`` through ``argparse.ArgumentParser``.

    Returns
    -------
    Tuple[argparse.Namespace, Dict[str, Any]]
        A tuple ``(args, info)`` where ``args`` is the parsed namespace with
        additional derived attributes such as ``inputdir``,
        ``explanation_paths_prefix``, ``metric_params``, ``outputdir``,
        ``outfilename``, ``start_time``, ``include_user_history``, and
        ``early_stopping``. ``info`` contains ``{"args": vars(args)}`` so the
        final configuration can be logged downstream.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` when required arguments are missing or invalid.
    OSError
        Raised if the output directory needs to be created and the filesystem
        operation fails.

    Side Effects
    ------------
    Prints the parsed arguments and status messages, calls
    ``check_if_out_file_exists(args)``, creates the output directory when it
    does not exist yet, and seeds Python's ``random``, NumPy, and PyTorch
    random number generators.

    Notes
    -----
    This function acts as the configuration entry point for prompt
    optimization. The metric-specific branch converts the selected objective
    into the ``metric_params`` structure expected by later evaluation and
    optimization stages.
    """
    
    parser = argparse.ArgumentParser(
        description="Prompt optimization for LLM explainability (path selection)."
    )

    # Input
    parser.add_argument(
        "--datain",
        type=str,
        required=True,
        help="Base input data directory (e.g., ../datasets).",
    )

    # Explanation paths (same as explainability)
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm name used to locate explanation paths (e.g., user_knn, item_knn).",
    )

    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="random",
        help="Strategy used to select explanation paths (default: random).",
    )

    parser.add_argument(
        "--num_recommendations",
        type=int,
        default=3,
        help="Number of recommendations to include per user (default: 3).",
    )

    parser.add_argument(
        "--num_paths_per_recommendation",
        type=int,
        default=3,
        help="Number of candidate explanation paths per recommendation (default: 3).",
    )

    parser.add_argument(
        "--include_user_history",
        type=str,
        default="true",
        help="Whether to include user interaction history in the prompt (true/false).",
    )

    parser.add_argument(
        "--selected_paths_input_path",
        type=str,
        default=None,
        help=(
            "Optional CSV path with precomputed candidate paths. When provided, "
            "prompt optimization reuses this file instead of sampling paths "
            "again."
        ),
    )

    # Knowledge Graph (needed by metrics)
    parser.add_argument(
        "--kg_path",
        type=str,
        required=True,
        help="Path to the Knowledge Graph CSV file.",
    )

    # Seed and model
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for reproducibility.",
    )

    parser.add_argument(
        "--llm_method",
        type=str,
        required=True,
        help="LLM method or model name.",
    )

    # Optimization settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of optimization iterations.",
    )

    parser.add_argument(
        "--total_instructions_per_iteration",
        type=int,
        default=1,
        help="Candidate prompts per iteration.",
    )

    parser.add_argument(
        "--meta_prompt_instruction_quantity",
        type=int,
        default=3,
        help="How many best prompts to show as examples.",
    )

    # Optimization control (validation + early stopping)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run validation every N iterations (default: 1).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in validation checks (default: 3).",
    )

    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-2,
        help="Minimum validation improvement to be considered progress (default: 1e-2).",
    )

    parser.add_argument(
        "--early_stopping",
        type=str,
        default="false",
        help="Whether to enable early stopping during validation checks (true/false).",
    )

    parser.add_argument(
        "--mmr_lambda_quality",
        type=float,
        default=1.0,
        help="MMR balance between relevance and diversity in reference selection (default: 1.0).",
    )

    parser.add_argument(
        "--mmr_pool_multiplier",
        type=int,
        default=10,
        help="MMR candidate pool multiplier used in reference selection (default: 10).",
    )

    parser.add_argument(
        "--representation_model",
        type=str,
        default="llm2vec",
        choices=available_representations(),
        help="Text representation model used to embed candidate system instructions.",
    )

    # Metric (objective)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["sep", "etd"],
        required=True,
        help="Metric used to score explanations during prompt optimization.",
    )

    parser.add_argument(
        "--sep_beta",
        type=float,
        default=0.3,
        help="(SEP only) Exponential decay parameter beta.",
    )

    parser.add_argument(
        "--etd_k",
        type=int,
        default=5,
        help="(ETD only) Number of explanations (k) considered.",
    )

    # Output
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Base output directory.",
    )

    # Machine / environment
    parser.add_argument(
        "--machine",
        type=str,
        default=socket.gethostname(),
        help="Machine hostname.",
    )

    args = parser.parse_args()

    # Post-processing
    args.inputdir = f"{args.datain}"

    # Normalize CLI string values into booleans expected by the downstream flow.
    args.include_user_history = str(args.include_user_history).lower() in ["true", "1", "yes"]
    args.early_stopping = str(args.early_stopping).lower() in ["true", "1", "yes"]

    # Build the shared prefix used to locate the explanation-path files.
    args.explanation_paths_prefix = os.path.join(
        args.datain,
        "explanation_paths",
        f"{args.algorithm}-opt",
        args.algorithm,
    )

    # Convert metric-specific CLI options into the compact structure used later.
    if args.metric == "sep":
        args.metric_params = {"beta": float(args.sep_beta)}
    else:
        args.metric_params = {"k": int(args.etd_k)}

    # Output dirs/files
    args.outputdir = os.path.join(args.out, args.llm_method, "prompt_opt", args.metric)
    args.outfilename = os.path.join(args.outputdir, "optimization_process")
    args.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print("\n", args)
    check_if_out_file_exists(args)

    if not os.path.exists(args.outputdir):
        print(f"\nCreating output directory at {args.outputdir}\n")
        os.makedirs(args.outputdir, exist_ok=True)
    else:
        print("\n")

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    info = {"args": vars(args)}
    return args, info

def args_prepare_selected_paths() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Build and parse CLI arguments for the selected-path precomputation runner.

    This function configures a lightweight preprocessing step that samples the
    candidate explanation paths before the LLM runs. The resulting CSV can be
    reused later by either ``run_prompt_optimizer.py`` or
    ``run_llm_explainability.py``.
    """

    parser = argparse.ArgumentParser(
        description="Precompute and save selected explanation-path candidates."
    )

    parser.add_argument(
        "--datain",
        type=str,
        required=True,
        help="Base input data directory (e.g., ../datasets).",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm name used to locate explanation paths (e.g., user_knn, item_knn).",
    )

    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="random",
        help="Strategy used to select explanation paths (default: random).",
    )

    parser.add_argument(
        "--num_recommendations",
        type=int,
        default=3,
        help="Number of recommendations to include per user (default: 3).",
    )

    parser.add_argument(
        "--num_paths_per_recommendation",
        type=int,
        default=3,
        help="Number of candidate explanation paths per recommendation (default: 3).",
    )

    parser.add_argument(
        "--user_scope",
        type=str,
        default="test",
        choices=["train", "val", "test", "train_val", "all", "optimization", "explainability"],
        help=(
            "Which user split to precompute. "
            "'optimization' is an alias for 'train_val' and "
            "'explainability' is an alias for 'test'."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for reproducibility.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Base output directory used to store the selected-path CSV.",
    )

    parser.add_argument(
        "--machine",
        type=str,
        default=socket.gethostname(),
        help="Machine hostname.",
    )

    args = parser.parse_args()

    args.inputdir = f"{args.datain}"
    args.explanation_paths_prefix = os.path.join(
        args.datain,
        "explanation_paths",
        f"{args.algorithm}-opt",
        args.algorithm,
    )
    args.outputdir = os.path.join(
        args.out,
        args.algorithm,
        args.user_scope,
        args.selection_strategy,
        f"recs_{args.num_recommendations}_paths_{args.num_paths_per_recommendation}",
        f"seed_{args.seed}",
    )
    args.outfilename = os.path.join(args.outputdir, "selected_paths")
    args.selected_paths_output_path = args.outfilename + ".csv"
    args.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print("\n", args)
    check_if_out_file_exists(args)

    if not os.path.exists(args.outputdir):
        print(f"\nCreating output directory at {args.outputdir}\n")
        os.makedirs(args.outputdir, exist_ok=True)
    else:
        print("\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    info = {"args": vars(args)}
    return args, info
