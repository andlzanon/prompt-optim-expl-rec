from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - fallback only
    wilcoxon = None


OBJECTIVE_METRICS = ("sep", "etd", "sep_etd_f1")
METHOD_LABELS = {
    "lod": "LOD",
    "llama_without_optimization": "Llama w/o opt.",
    "llama_with_optimization": "Llama w/ opt.",
}
LATEX_SYMBOLS = {
    "better": r"\greentriangleup",
    "equal": r"\yellowcircle",
    "worse": r"\redtriangledown",
}
ALGORITHM_LABELS = {
    "user_knn": "User-KNN",
    "item_knn": "Item-KNN",
    "bprmf": "BPR-MF",
    "ncf": "NCF",
}
REPRESENTATION_LABELS = {
    "repr_llm2vec": "LLM2Vec",
    "repr_sbert": "SBERT",
}

def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "run_prompt_optimizer.py").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the explainability-with-LLMs root."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.graph_utils import (  # noqa: E402
    combine_sep_etd_f1,
    parse_graph_explanations,
)
from src.metrics.sep import _build_sep_table  # noqa: E402
from src.utils.geral import explanations_df_to_blocks  # noqa: E402


def workspace_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class RunDescriptor:
    run_type: str
    model: str
    algorithm: str
    metric: str
    representation_model: str | None
    early_stopping_tag: str | None
    mmr_lambda_tag: str | None
    mmr_pool_tag: str | None
    responses_path: Path
    metadata_path: Path

    @property
    def baseline_key(self) -> tuple[str, str, str]:
        return (self.model, self.algorithm, self.metric)

    @property
    def run_label(self) -> str:
        parts = [self.run_type, self.model, self.algorithm, self.metric]
        for extra in (
            self.representation_model,
            self.early_stopping_tag,
            self.mmr_lambda_tag,
            self.mmr_pool_tag,
        ):
            if extra:
                parts.append(extra)
        return "/".join(parts)

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def resolve_project_relative(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()

def parse_run_descriptor(responses_path: Path, test_root: Path) -> RunDescriptor:
    rel_parts = responses_path.relative_to(test_root).parts
    run_type = rel_parts[0]

    if run_type == "without_optimization":
        if len(rel_parts) != 5:
            raise ValueError(f"Unexpected layout for without_optimization: {responses_path}")
        _, model, algorithm, metric, _ = rel_parts
        return RunDescriptor(
            run_type=run_type,
            model=model,
            algorithm=algorithm,
            metric=metric,
            representation_model=None,
            early_stopping_tag=None,
            mmr_lambda_tag=None,
            mmr_pool_tag=None,
            responses_path=responses_path,
            metadata_path=responses_path.with_name("responses_metadata.json"),
        )

    if run_type == "with_optimization":
        if len(rel_parts) != 9:
            raise ValueError(f"Unexpected layout for with_optimization: {responses_path}")
        (
            _,
            model,
            algorithm,
            metric,
            representation_model,
            early_stopping_tag,
            mmr_lambda_tag,
            mmr_pool_tag,
            _,
        ) = rel_parts
        return RunDescriptor(
            run_type=run_type,
            model=model,
            algorithm=algorithm,
            metric=metric,
            representation_model=representation_model,
            early_stopping_tag=early_stopping_tag,
            mmr_lambda_tag=mmr_lambda_tag,
            mmr_pool_tag=mmr_pool_tag,
            responses_path=responses_path,
            metadata_path=responses_path.with_name("responses_metadata.json"),
        )

    raise ValueError(f"Unsupported run type: {responses_path}")

def discover_runs(
    test_root: Path,
    model_filter: str | None = None,
    algorithm_filter: str | None = None,
) -> list[RunDescriptor]:
    descriptors: list[RunDescriptor] = []
    for response_path in sorted(test_root.rglob("responses.csv")):
        descriptor = parse_run_descriptor(response_path, test_root)
        if model_filter and descriptor.model != model_filter:
            continue
        if algorithm_filter and descriptor.algorithm != algorithm_filter:
            continue
        descriptors.append(descriptor)
    return descriptors


PROPS_DF_CACHE: dict[Path, pd.DataFrame] = {}
TOTAL_TYPES_CACHE: dict[Path, int] = {}
SEP_MEMO_CACHE: dict[tuple[Path, float], dict[Any, Any]] = {}
PROP_LINKS_KEY_CACHE: dict[Path, dict[str, tuple[str, ...]]] = {}

def extract_sep_beta(metadata: dict[str, Any]) -> float:
    metric_params = metadata.get("metric_params") or {}
    if "beta" in metric_params:
        return float(metric_params["beta"])

    args_metric_params = metadata.get("args", {}).get("metric_params", {})
    if "beta" in args_metric_params:
        return float(args_metric_params["beta"])

    args_sep_beta = metadata.get("args", {}).get("sep_beta")
    if args_sep_beta is not None:
        return float(args_sep_beta)

    return 0.3

def count_valid_explanation_lines(block: str) -> int:
    if not isinstance(block, str):
        return 0
    return sum(
        1 for line in block.splitlines() if line.strip() and "|" in line and "->" in line
    )

def build_per_user_cache_path(cache_dir: Path, descriptor: RunDescriptor) -> Path:
    safe_label = descriptor.run_label.replace("/", "__")
    digest = hashlib.md5(str(descriptor.responses_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{safe_label}__{digest}.csv"

def build_prop_links_key_map(props_df: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    tmp = props_df[["obj", "prop"]].dropna().copy()
    tmp["obj"] = tmp["obj"].astype(str)
    tmp["prop"] = tmp["prop"].astype(str)
    grouped = tmp.groupby("obj")["prop"].agg(lambda s: tuple(sorted(set(s.tolist()))))
    return grouped.to_dict()

def compute_fast_sep_score(
    middle_entities: list[str],
    props_df: pd.DataFrame,
    prop_links_key_map: dict[str, tuple[str, ...]],
    memo_sep: dict[Any, Any],
    beta: float,
) -> float:
    if not middle_entities:
        return 0.0

    values: list[float] = []
    for entity in middle_entities:
        links_key = prop_links_key_map.get(str(entity), ())
        if not links_key:
            continue

        memo_df = memo_sep.get(links_key)
        if memo_df is None:
            memo_df = _build_sep_table(
                beta=beta,
                prop_set=props_df,
                links_key=links_key,
            )
            if memo_df.empty:
                continue
            memo_sep[links_key] = memo_df

        if entity not in memo_df.index:
            continue
        values.append(float(memo_df.loc[entity, "normalized"]))

    if not values:
        return 0.0
    return float(sum(values) / len(values))

def compute_fast_etd_score(middle_entities: list[str], total_types: int) -> float:
    max_distinct = min(len(middle_entities), total_types)
    if max_distinct <= 0:
        return 0.0
    return float(len(set(middle_entities)) / max_distinct)

def compute_metrics_per_user(
    descriptor: RunDescriptor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = load_json(descriptor.metadata_path)
    kg_path = resolve_project_relative(
        metadata.get("kg_path") or metadata.get("args", {}).get("kg_path")
    )
    if kg_path is None or not kg_path.exists():
        raise FileNotFoundError(
            f"Could not resolve kg_path for {descriptor.metadata_path}: {kg_path}"
        )

    beta = extract_sep_beta(metadata)
    props_df = PROPS_DF_CACHE.setdefault(kg_path, pd.read_csv(kg_path))
    total_types = TOTAL_TYPES_CACHE.setdefault(kg_path, int(props_df["obj"].dropna().nunique()))
    prop_links_key_map = PROP_LINKS_KEY_CACHE.setdefault(kg_path, build_prop_links_key_map(props_df))
    memo_sep = SEP_MEMO_CACHE.setdefault((kg_path, beta), {})

    responses_df = pd.read_csv(descriptor.responses_path)
    explanation_blocks = explanations_df_to_blocks(responses_df)

    rows = []
    ordered_user_ids = responses_df["userId"].dropna().astype(int).drop_duplicates().tolist()
    for user_id in ordered_user_ids:
        block = explanation_blocks.get(user_id, "")
        parsed = parse_graph_explanations([block])
        middle_entities = [entity for entity in parsed["middle_entities"] if entity]

        sep_score = compute_fast_sep_score(
            middle_entities=middle_entities,
            props_df=props_df,
            prop_links_key_map=prop_links_key_map,
            memo_sep=memo_sep,
            beta=beta,
        )
        etd_score = compute_fast_etd_score(
            middle_entities=middle_entities,
            total_types=total_types,
        )
        sep_etd_f1_score = float(combine_sep_etd_f1(sep_value=sep_score, etd_value=etd_score))

        user_rows = responses_df[responses_df["userId"] == user_id]
        valid_line_count = count_valid_explanation_lines(block)
        rows.append(
            {
                "userId": int(user_id),
                "sep": sep_score,
                "etd": etd_score,
                "sep_etd_f1": sep_etd_f1_score,
                "n_response_rows": int(len(user_rows)),
                "n_valid_explanations": int(valid_line_count),
            }
        )

    per_user_df = pd.DataFrame(rows).sort_values("userId").reset_index(drop=True)
    run_summary = {
        "run_label": descriptor.run_label,
        "responses_path": workspace_relative_path(descriptor.responses_path),
        "metadata_path": workspace_relative_path(descriptor.metadata_path),
        "kg_path": workspace_relative_path(kg_path),
        "beta": beta,
        "n_users": int(len(per_user_df)),
        "mean_sep_per_user": float(per_user_df["sep"].mean()) if not per_user_df.empty else 0.0,
        "mean_etd_per_user": float(per_user_df["etd"].mean()) if not per_user_df.empty else 0.0,
        "mean_sep_etd_f1_per_user": (
            float(per_user_df["sep_etd_f1"].mean()) if not per_user_df.empty else 0.0
        ),
        "metadata_sep": metadata.get("sep_value"),
        "metadata_etd": metadata.get("etd_value"),
        "metadata_sep_etd_f1": metadata.get("sep_etd_f1_value"),
    }
    return per_user_df, run_summary

def load_lod_scores(lod_results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(lod_results_dir.glob("indiv_metrics_explanations_optimized_*_K=*_recs.csv")):
        algorithm = parse_lod_algorithm(path)
        df = pd.read_csv(path)
        required_cols = {"userId", "sep", "etd"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"LOD file missing required columns {missing}: {path}")

        lod_df = df[["userId", "sep", "etd"]].copy()
        lod_df["userId"] = lod_df["userId"].astype(int)
        lod_df["sep"] = lod_df["sep"].astype(float)
        lod_df["etd"] = lod_df["etd"].astype(float)
        lod_df["sep_etd_f1"] = lod_df.apply(
            lambda row: combine_sep_etd_f1(row["sep"], row["etd"]), axis=1
        )
        lod_df["algorithm"] = algorithm
        lod_df["lod_path"] = str(path.resolve())
        rows.append(lod_df[["algorithm", "userId", "sep", "etd", "sep_etd_f1", "lod_path"]])

    if not rows:
        raise FileNotFoundError(f"No LOD files found in {lod_results_dir}")

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["algorithm", "userId"])
        .reset_index(drop=True)
    )

def parse_lod_algorithm(path: Path) -> str:
    name = path.name
    for algorithm in ALGORITHM_LABELS:
        if f"_{algorithm}_" in name:
            return algorithm
    raise ValueError(f"Could not identify the algorithm from LOD file: {path.name}")

def method_score_column(method: str, objective_metric: str) -> str:
    return f"{objective_metric}_{method}"

def compare_symbol(left_mean: float, right_mean: float, p_value: float | None, alpha: float) -> str:
    if p_value is None or math.isnan(p_value) or p_value >= alpha:
        return LATEX_SYMBOLS["equal"]
    if left_mean > right_mean:
        return LATEX_SYMBOLS["better"]
    return LATEX_SYMBOLS["worse"]

def wilcoxon_pair_summary(
    paired_df: pd.DataFrame,
    objective_metric: str,
    left_method: str,
    right_method: str,
    alpha: float,
) -> dict[str, Any]:
    left_col = method_score_column(left_method, objective_metric)
    right_col = method_score_column(right_method, objective_metric)
    diff = paired_df[left_col] - paired_df[right_col]

    summary = {
        "objective_metric": objective_metric,
        "left_method": left_method,
        "right_method": right_method,
        "left_label": METHOD_LABELS[left_method],
        "right_label": METHOD_LABELS[right_method],
        "n_users_paired": int(len(paired_df)),
        "mean_left": float(paired_df[left_col].mean()),
        "mean_right": float(paired_df[right_col].mean()),
        "mean_diff_left_minus_right": float(diff.mean()),
        "median_diff_left_minus_right": float(diff.median()),
        "n_users_left_higher": int((diff > 0).sum()),
        "n_users_right_higher": int((diff < 0).sum()),
        "n_users_tied": int((diff == 0).sum()),
        "alpha": float(alpha),
    }

    if wilcoxon is None:
        summary["wilcoxon_statistic"] = None
        summary["wilcoxon_pvalue"] = None
        summary["wilcoxon_error"] = "scipy is not installed"
    elif len(paired_df) == 0:
        summary["wilcoxon_statistic"] = None
        summary["wilcoxon_pvalue"] = None
        summary["wilcoxon_error"] = "no paired users"
    else:
        try:
            result = wilcoxon(
                paired_df[left_col],
                paired_df[right_col],
                alternative="two-sided",
                zero_method="wilcox",
            )
            summary["wilcoxon_statistic"] = float(result.statistic)
            summary["wilcoxon_pvalue"] = float(result.pvalue)
            summary["wilcoxon_error"] = None
        except ValueError as exc:
            summary["wilcoxon_statistic"] = None
            summary["wilcoxon_pvalue"] = None
            summary["wilcoxon_error"] = str(exc)

    p_value = summary["wilcoxon_pvalue"]
    if p_value is None or pd.isna(p_value):
        summary["statistical_conclusion"] = "sem conclusao estatistica"
    elif p_value < alpha:
        summary["statistical_conclusion"] = "significant difference"
    else:
        summary["statistical_conclusion"] = "no significant difference"

    summary["left_vs_right_symbol"] = compare_symbol(
        left_mean=summary["mean_left"],
        right_mean=summary["mean_right"],
        p_value=summary["wilcoxon_pvalue"],
        alpha=alpha,
    )
    return summary

def format_float_ptbr(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")

def format_algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm.replace("_", r"\_"))

def format_representation_label(representation_model: str | None) -> str:
    if not representation_model:
        return "-"
    return REPRESENTATION_LABELS.get(representation_model, representation_model.replace("_", r"\_"))

def format_lambda_value(mmr_lambda_tag: str | None) -> str:
    if not mmr_lambda_tag:
        return "-"
    return mmr_lambda_tag.replace("mmr_lambda_", "").replace("_", ".")

def bold_if_max(value: float, all_values: list[float], digits: int = 4) -> str:
    formatted = format_float_ptbr(value, digits=digits)
    if value == max(all_values):
        return rf"\textbf{{{formatted}}}"
    return formatted

def build_article_table_tex(
    objective_metric: str,
    article_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_label = objective_metric.upper()
    lines = [
        r"\begin{table*}[!t]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        \textbf{Algorithm} & \textbf{LOD} & \textbf{Llama w/o opt.} & \textbf{Best optimized Llama} & \textbf{Opt. vs w/o opt.} & \textbf{Opt. vs LOD} & \textbf{Best config.} \\",
        r"        \midrule",
    ]

    for _, row in article_df.iterrows():
        values = [row["mean_lod"], row["mean_llama_without_optimization"], row["mean_llama_with_optimization"]]
        config = f"{format_representation_label(row['representation_model'])}, " + r"$\lambda$" + f"={format_lambda_value(row['mmr_lambda_tag'])}"
        lines.append(
            "        "
            + " & ".join(
                [
                    format_algorithm_label(row["algorithm"]),
                    bold_if_max(row["mean_lod"], values),
                    bold_if_max(row["mean_llama_without_optimization"], values),
                    bold_if_max(row["mean_llama_with_optimization"], values),
                    row["optimized_vs_baseline_symbol"],
                    row["optimized_vs_lod_symbol"],
                    config,
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            (
                r"    \caption{Paired comparison on "
                + metric_label
                + r" between LOD, Llama without optimization, and the best optimized Llama for each algorithm. "
                + r"Bold values indicate the highest mean per-user value in each row. "
                + r"The symbol \greentriangleup indicates that the best optimized Llama was statistically superior in the corresponding comparison "
                + r"(\textit{p-value} $< 0.05$, Wilcoxon test); \yellowcircle indicates no significant difference; "
                + r"and \redtriangledown indicates that the best optimized Llama was statistically inferior.}"
            ),
            rf"    \label{{tab:stat_{objective_metric}}}",
            r"\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")

def write_metric_report(
    objective_metric: str,
    metric_dir: Path,
    article_tables_dir: Path,
    candidate_df: pd.DataFrame,
    best_summary_df: pd.DataFrame,
    best_wide_df: pd.DataFrame,
    winner_pairwise_df: pd.DataFrame,
    all_pairwise_df: pd.DataFrame,
    article_df: pd.DataFrame,
) -> None:
    results_dir = metric_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_df.to_csv(results_dir / f"{objective_metric}_all_optimized_candidates_summary.csv", index=False)
    best_wide_df.to_csv(results_dir / f"{objective_metric}_best_method_by_algorithm_per_user_wide.csv", index=False)
    best_summary_df.to_csv(results_dir / f"{objective_metric}_best_method_by_algorithm_summary.csv", index=False)
    winner_pairwise_df.to_csv(results_dir / f"{objective_metric}_best_method_pairwise_wilcoxon.csv", index=False)
    all_pairwise_df.to_csv(results_dir / f"{objective_metric}_all_method_pairwise_wilcoxon.csv", index=False)
    article_df.to_csv(results_dir / f"{objective_metric}_article_compact_table.csv", index=False)

    article_tables_dir.mkdir(parents=True, exist_ok=True)
    build_article_table_tex(
        objective_metric=objective_metric,
        article_df=article_df,
        output_path=article_tables_dir / f"stat_table_{objective_metric}.tex",
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate statistical analysis artifacts for explainability runs.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to the explainability-with-LLMs project root.",
    )
    parser.add_argument(
        "--llm-method",
        default="Llama3.1-I",
        help="LLM method under out/test_explainability.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        help="Optional algorithm filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Wilcoxon tests.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Base output directory for statistical analysis. Default: <project-root>/analyse_results/statistical_analysis",
    )
    parser.add_argument(
        "--article-dir",
        type=Path,
        default=WORKSPACE_ROOT / "webmedia2026_old" / "webmedia2025",
        help="Article directory where compact LaTeX tables will be written under generated_tables/.",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    out_dir = (args.out_dir or (project_root / "analyse_results" / "statistical_analysis")).resolve()
    test_root = project_root / "out" / "test_explainability"
    lod_results_dir = project_root.parent / "datasets" / "lod_results" / "individual_metrics"
    article_tables_dir = args.article_dir.resolve() / "generated_tables"
    per_user_cache_dir = out_dir / "common" / "per_user_cache"
    per_user_cache_dir.mkdir(parents=True, exist_ok=True)

    descriptors = discover_runs(
        test_root=test_root,
        model_filter=args.llm_method,
        algorithm_filter=args.algorithms[0] if args.algorithms and len(args.algorithms) == 1 else None,
    )
    if args.algorithms and len(args.algorithms) > 1:
        allowed_algorithms = set(args.algorithms)
        descriptors = [descriptor for descriptor in descriptors if descriptor.algorithm in allowed_algorithms]

    if not descriptors:
        raise FileNotFoundError("No responses.csv file was found for the provided filters.")

    lod_scores_df = load_lod_scores(lod_results_dir)
    if args.algorithms:
        lod_scores_df = lod_scores_df[lod_scores_df["algorithm"].isin(set(args.algorithms))].copy()

    per_run_scores: dict[RunDescriptor, pd.DataFrame] = {}
    run_summary_rows: list[dict[str, Any]] = []

    for descriptor in descriptors:
        cache_path = build_per_user_cache_path(per_user_cache_dir, descriptor)
        if cache_path.exists():
            per_user_df = pd.read_csv(cache_path)
            cache_status = "cache_hit"
            run_summary = {
                "mean_sep_per_user": float(per_user_df["sep"].mean()) if not per_user_df.empty else 0.0,
                "mean_etd_per_user": float(per_user_df["etd"].mean()) if not per_user_df.empty else 0.0,
                "mean_sep_etd_f1_per_user": (
                    float(per_user_df["sep_etd_f1"].mean()) if not per_user_df.empty else 0.0
                ),
                "n_users": int(len(per_user_df)),
            }
        else:
            per_user_df, run_summary = compute_metrics_per_user(descriptor)
            per_user_df.to_csv(cache_path, index=False)
            cache_status = "computed"

        per_run_scores[descriptor] = per_user_df
        run_summary_rows.append(
            {
                "run_type": descriptor.run_type,
                "model": descriptor.model,
                "algorithm": descriptor.algorithm,
                "metric": descriptor.metric,
                "representation_model": descriptor.representation_model,
                "early_stopping_tag": descriptor.early_stopping_tag,
                "mmr_lambda_tag": descriptor.mmr_lambda_tag,
                "mmr_pool_tag": descriptor.mmr_pool_tag,
                "cache_status": cache_status,
                "per_user_csv": workspace_relative_path(cache_path),
                "run_label": descriptor.run_label,
                **run_summary,
            }
        )

    run_summary_df = pd.DataFrame(run_summary_rows).sort_values(
        [
            "run_type",
            "algorithm",
            "metric",
            "representation_model",
            "mmr_lambda_tag",
        ],
        na_position="last",
    )
    run_summary_output = out_dir / "common" / "per_user_run_summary.csv"
    run_summary_output.parent.mkdir(parents=True, exist_ok=True)
    run_summary_df.to_csv(run_summary_output, index=False)

    baseline_runs = {
        descriptor.baseline_key: descriptor
        for descriptor in descriptors
        if descriptor.run_type == "without_optimization"
    }

    generated_paths: list[Path] = [run_summary_output]

    for objective_metric in OBJECTIVE_METRICS:
        metric_dir = out_dir / objective_metric
        candidate_rows: list[dict[str, Any]] = []
        candidate_paired_frames: dict[str, pd.DataFrame] = {}

        for descriptor in descriptors:
            if descriptor.run_type != "with_optimization" or descriptor.metric != objective_metric:
                continue

            baseline_descriptor = baseline_runs.get(descriptor.baseline_key)
            if baseline_descriptor is None:
                continue

            lod_df = (
                lod_scores_df[lod_scores_df["algorithm"] == descriptor.algorithm][["userId", "sep", "etd", "sep_etd_f1"]]
                .rename(
                    columns={
                        "sep": "sep_lod",
                        "etd": "etd_lod",
                        "sep_etd_f1": "sep_etd_f1_lod",
                    }
                )
                .copy()
            )
            baseline_df = (
                per_run_scores[baseline_descriptor][["userId", "sep", "etd", "sep_etd_f1"]]
                .rename(
                    columns={
                        "sep": "sep_llama_without_optimization",
                        "etd": "etd_llama_without_optimization",
                        "sep_etd_f1": "sep_etd_f1_llama_without_optimization",
                    }
                )
                .copy()
            )
            optimized_df = (
                per_run_scores[descriptor][["userId", "sep", "etd", "sep_etd_f1"]]
                .rename(
                    columns={
                        "sep": "sep_llama_with_optimization",
                        "etd": "etd_llama_with_optimization",
                        "sep_etd_f1": "sep_etd_f1_llama_with_optimization",
                    }
                )
                .copy()
            )

            paired_df = (
                lod_df.merge(baseline_df, on="userId", how="inner")
                .merge(optimized_df, on="userId", how="inner")
                .sort_values("userId")
                .reset_index(drop=True)
            )
            if paired_df.empty:
                continue

            candidate_paired_frames[descriptor.run_label] = paired_df
            candidate_rows.append(
                {
                    "model": descriptor.model,
                    "algorithm": descriptor.algorithm,
                    "metric": objective_metric,
                    "representation_model": descriptor.representation_model,
                    "early_stopping_tag": descriptor.early_stopping_tag,
                    "mmr_lambda_tag": descriptor.mmr_lambda_tag,
                    "mmr_pool_tag": descriptor.mmr_pool_tag,
                    "baseline_run_label": baseline_descriptor.run_label,
                    "optimized_run_label": descriptor.run_label,
                    "n_users_paired": int(len(paired_df)),
                    "mean_sep_lod": float(paired_df["sep_lod"].mean()),
                    "mean_sep_llama_without_optimization": float(paired_df["sep_llama_without_optimization"].mean()),
                    "mean_sep_llama_with_optimization": float(paired_df["sep_llama_with_optimization"].mean()),
                    "mean_etd_lod": float(paired_df["etd_lod"].mean()),
                    "mean_etd_llama_without_optimization": float(paired_df["etd_llama_without_optimization"].mean()),
                    "mean_etd_llama_with_optimization": float(paired_df["etd_llama_with_optimization"].mean()),
                    "mean_sep_etd_f1_lod": float(paired_df["sep_etd_f1_lod"].mean()),
                    "mean_sep_etd_f1_llama_without_optimization": float(paired_df["sep_etd_f1_llama_without_optimization"].mean()),
                    "mean_sep_etd_f1_llama_with_optimization": float(paired_df["sep_etd_f1_llama_with_optimization"].mean()),
                }
            )

        candidate_df = pd.DataFrame(candidate_rows)
        if candidate_df.empty:
            continue

        sort_cols = [f"mean_{objective_metric}_llama_with_optimization"]
        best_candidate_df = (
            candidate_df.sort_values(sort_cols, ascending=False)
            .groupby("algorithm", as_index=False)
            .first()
        )

        best_summary_rows: list[dict[str, Any]] = []
        best_wide_rows: list[pd.DataFrame] = []
        winner_pairwise_rows: list[dict[str, Any]] = []
        all_pairwise_rows: list[dict[str, Any]] = []
        article_rows: list[dict[str, Any]] = []

        for _, best_row in best_candidate_df.iterrows():
            paired_df = candidate_paired_frames[best_row["optimized_run_label"]].copy()
            for column in (
                "model",
                "algorithm",
                "metric",
                "representation_model",
                "early_stopping_tag",
                "mmr_lambda_tag",
                "mmr_pool_tag",
                "baseline_run_label",
                "optimized_run_label",
            ):
                paired_df[column] = best_row[column]
            best_wide_rows.append(paired_df)

            method_means = {
                "lod": float(best_row[f"mean_{objective_metric}_lod"]),
                "llama_without_optimization": float(
                    best_row[f"mean_{objective_metric}_llama_without_optimization"]
                ),
                "llama_with_optimization": float(
                    best_row[f"mean_{objective_metric}_llama_with_optimization"]
                ),
            }
            winner_method = max(method_means, key=method_means.get)
            winner_label = METHOD_LABELS[winner_method]

            best_row = best_row.copy()
            best_row["winner_method"] = winner_method
            best_row["winner_label"] = winner_label
            best_row["winner_mean_objective"] = method_means[winner_method]
            best_summary_rows.append(best_row.to_dict())

            for left_method, right_method in combinations(
                ("lod", "llama_without_optimization", "llama_with_optimization"), 2
            ):
                pair_summary = wilcoxon_pair_summary(
                    paired_df=paired_df,
                    objective_metric=objective_metric,
                    left_method=left_method,
                    right_method=right_method,
                    alpha=args.alpha,
                )
                pair_summary.update(
                    {
                        "model": best_row["model"],
                        "algorithm": best_row["algorithm"],
                        "metric": best_row["metric"],
                        "representation_model": best_row["representation_model"],
                        "early_stopping_tag": best_row["early_stopping_tag"],
                        "mmr_lambda_tag": best_row["mmr_lambda_tag"],
                        "mmr_pool_tag": best_row["mmr_pool_tag"],
                        "baseline_run_label": best_row["baseline_run_label"],
                        "optimized_run_label": best_row["optimized_run_label"],
                    }
                )
                all_pairwise_rows.append(pair_summary)

            for compared_method in ("lod", "llama_without_optimization", "llama_with_optimization"):
                if compared_method == winner_method:
                    continue
                pair_summary = wilcoxon_pair_summary(
                    paired_df=paired_df,
                    objective_metric=objective_metric,
                    left_method=winner_method,
                    right_method=compared_method,
                    alpha=args.alpha,
                )
                pair_summary.update(
                    {
                        "model": best_row["model"],
                        "algorithm": best_row["algorithm"],
                        "metric": best_row["metric"],
                        "representation_model": best_row["representation_model"],
                        "early_stopping_tag": best_row["early_stopping_tag"],
                        "mmr_lambda_tag": best_row["mmr_lambda_tag"],
                        "mmr_pool_tag": best_row["mmr_pool_tag"],
                        "baseline_run_label": best_row["baseline_run_label"],
                        "optimized_run_label": best_row["optimized_run_label"],
                        "winner_method": winner_method,
                        "winner_label": winner_label,
                    }
                )
                winner_pairwise_rows.append(pair_summary)

            opt_vs_base = next(
                row
                for row in all_pairwise_rows
                if row["algorithm"] == best_row["algorithm"]
                and row["left_method"] == "llama_without_optimization"
                and row["right_method"] == "llama_with_optimization"
            )
            opt_vs_lod = next(
                row
                for row in all_pairwise_rows
                if row["algorithm"] == best_row["algorithm"]
                and row["left_method"] == "lod"
                and row["right_method"] == "llama_with_optimization"
            )

            optimized_mean = float(best_row[f"mean_{objective_metric}_llama_with_optimization"])
            baseline_mean = float(best_row[f"mean_{objective_metric}_llama_without_optimization"])
            lod_mean = float(best_row[f"mean_{objective_metric}_lod"])
            article_rows.append(
                {
                    "algorithm": best_row["algorithm"],
                    "objective_metric": objective_metric,
                    "mean_lod": lod_mean,
                    "mean_llama_without_optimization": baseline_mean,
                    "mean_llama_with_optimization": optimized_mean,
                    "representation_model": best_row["representation_model"],
                    "mmr_lambda_tag": best_row["mmr_lambda_tag"],
                    "optimized_vs_baseline_pvalue": opt_vs_base["wilcoxon_pvalue"],
                    "optimized_vs_lod_pvalue": opt_vs_lod["wilcoxon_pvalue"],
                    "optimized_vs_baseline_symbol": compare_symbol(
                        left_mean=optimized_mean,
                        right_mean=baseline_mean,
                        p_value=opt_vs_base["wilcoxon_pvalue"],
                        alpha=args.alpha,
                    ),
                    "optimized_vs_lod_symbol": compare_symbol(
                        left_mean=optimized_mean,
                        right_mean=lod_mean,
                        p_value=opt_vs_lod["wilcoxon_pvalue"],
                        alpha=args.alpha,
                    ),
                    "winner_method": winner_method,
                }
            )

        best_summary_df = pd.DataFrame(best_summary_rows)
        best_wide_df = (
            pd.concat(best_wide_rows, ignore_index=True)
            if best_wide_rows
            else pd.DataFrame()
        )
        winner_pairwise_df = pd.DataFrame(winner_pairwise_rows)
        all_pairwise_df = pd.DataFrame(all_pairwise_rows)
        article_df = pd.DataFrame(article_rows).sort_values("algorithm").reset_index(drop=True)

        write_metric_report(
            objective_metric=objective_metric,
            metric_dir=metric_dir,
            article_tables_dir=article_tables_dir,
            candidate_df=candidate_df.sort_values(
                ["algorithm", f"mean_{objective_metric}_llama_with_optimization"],
                ascending=[True, False],
            ).reset_index(drop=True),
            best_summary_df=best_summary_df.sort_values("algorithm").reset_index(drop=True),
            best_wide_df=best_wide_df,
            winner_pairwise_df=winner_pairwise_df.sort_values(
                ["algorithm", "winner_method", "right_method"]
            ).reset_index(drop=True),
            all_pairwise_df=all_pairwise_df.sort_values(
                ["algorithm", "left_method", "right_method"]
            ).reset_index(drop=True),
            article_df=article_df,
        )

        generated_paths.extend(
            [
                metric_dir / "results" / f"{objective_metric}_all_optimized_candidates_summary.csv",
                metric_dir / "results" / f"{objective_metric}_best_method_by_algorithm_per_user_wide.csv",
                metric_dir / "results" / f"{objective_metric}_best_method_by_algorithm_summary.csv",
                metric_dir / "results" / f"{objective_metric}_best_method_pairwise_wilcoxon.csv",
                metric_dir / "results" / f"{objective_metric}_all_method_pairwise_wilcoxon.csv",
                metric_dir / "results" / f"{objective_metric}_article_compact_table.csv",
                article_tables_dir / f"stat_table_{objective_metric}.tex",
            ]
        )

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
