from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Iterable

def _cell_id(label: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in label)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")[:48] or "cell"

def _to_source_lines(text: str) -> list[str]:
    stripped = textwrap.dedent(text).strip("\n")
    if not stripped:
        return []
    return [line + "\n" for line in stripped.splitlines()]

def markdown_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(cell_id),
        "metadata": {},
        "source": _to_source_lines(text),
    }

def code_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(cell_id),
        "metadata": {},
        "outputs": [],
        "source": _to_source_lines(text),
    }

def render_template(text: str, **replacements: str) -> str:
    rendered = text
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered

def join_code_blocks(*blocks: str) -> str:
    normalized_blocks = []
    for block in blocks:
        normalized = textwrap.dedent(block).strip("\n")
        if normalized:
            normalized_blocks.append(normalized)
    return "\n\n".join(normalized_blocks)

def discover_algorithms(prompt_opt_root: Path) -> list[str]:
    algorithms = {
        metadata_path.relative_to(prompt_opt_root).parts[0]
        for metadata_path in prompt_opt_root.rglob("optimization_process_metadata.json")
    }
    return sorted(algorithms)


NOTEBOOK_METRIC_COLOR_HELPERS = textwrap.dedent(
    """
    METRIC_COLOR_MAP = {
        "etd": "#1f77b4",
        "sep": "#ff7f0e",
        "sep_etd_f1": "#2ca02c",
        "geom_mean": "#d62728",
        "mean_balance": "#8c564b",
    }


    def metric_color(metric_name: str | None, default: str = "#4C78A8") -> str:
        return METRIC_COLOR_MAP.get(str(metric_name), default)


    METRIC_LABEL_MAP = {
        "etd": "ETD",
        "sep": "SEP",
        "sep_etd_f1": "SEP_ETD_F1",
        "geom_mean": "Geometric Mean",
        "mean_balance": "Mean Balance",
    }


    def metric_label(metric_name: str | None) -> str:
        return METRIC_LABEL_MAP.get(str(metric_name), str(metric_name))


    def blend_with_white(color: str, blend: float) -> str:
        red, green, blue = mcolors.to_rgb(color)
        return mcolors.to_hex(
            tuple((1 - blend) * channel + blend for channel in (red, green, blue))
        )


    def metric_shades(metric_name: str | None, size: int) -> list[str]:
        base_color = metric_color(metric_name)
        if size <= 1:
            return [base_color]

        max_blend = 0.45
        return [
            blend_with_white(base_color, max_blend * (index / max(1, size - 1)))
            for index in range(size)
        ]


    def ensure_balance_metrics(epochs_df: pd.DataFrame) -> pd.DataFrame:
        epochs_df = epochs_df.copy()

        for split in ("train", "val"):
            sep_col = f"{split}_score_sep"
            etd_col = f"{split}_score_etd"

            if sep_col not in epochs_df.columns or etd_col not in epochs_df.columns:
                continue

            sep_series = pd.to_numeric(epochs_df[sep_col], errors="coerce")
            etd_series = pd.to_numeric(epochs_df[etd_col], errors="coerce")

            epochs_df[f"{split}_score_geom_mean"] = (
                sep_series.clip(lower=0) * etd_series.clip(lower=0)
            ).pow(0.5)
            epochs_df[f"{split}_score_mean_balance"] = (
                ((sep_series + etd_series) / 2.0)
                * (1 - (sep_series - etd_series).abs())
            )

        return epochs_df


    def preferred_metric_order(metric_names: list[str]) -> list[str]:
        preferred = ["sep", "etd", "sep_etd_f1", "geom_mean", "mean_balance"]
        ordered = [metric_name for metric_name in preferred if metric_name in metric_names]
        ordered.extend(metric_name for metric_name in metric_names if metric_name not in ordered)
        return ordered
    """
).strip()


NOTEBOOK_PROJECT_ROOT_HELPERS = textwrap.dedent(
    """
    def _is_project_root(candidate: Path) -> bool:
        return (candidate / "run_prompt_optimizer.py").exists() and (candidate / "out").exists()


    def find_local_project_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if _is_project_root(candidate):
                return candidate

        descendant_hints = []
        for base in (start, *start.parents):
            descendant_hints.extend(
                [
                    base / "prompt-optim-expl-rec" / "explainability-with-LLMs",
                    base / "explainability-with-LLMs",
                ]
            )

        for candidate in descendant_hints:
            if _is_project_root(candidate):
                return candidate

        raise FileNotFoundError(
            "Could not locate the explainability-with-LLMs root. "
            "Run the notebook inside the project, from one of its subdirectories, or from the workspace root."
        )


    def display_workspace_relative(path: Path, workspace_root: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(workspace_root.resolve()))
        except ValueError:
            return str(resolved)


    def display_path_string(path_str: str | None, workspace_root: Path) -> str | None:
        if not path_str:
            return path_str

        path = Path(path_str)
        if not path.is_absolute():
            return path_str

        return display_workspace_relative(path, workspace_root)
    """
).strip()

def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }

def build_notebook(algorithm: str, search_root_rel: str) -> dict:
    replacements = {
        "ALGORITHM": algorithm,
        "SEARCH_ROOT_REL": search_root_rel,
    }

    intro = render_template(
        """
        # Aggregate analysis of the prompt optimization processes for `__ALGORITHM__`

        This notebook scans the completed processes in `__SEARCH_ROOT_REL__` and helps identify which configurations performed best.

        The focus here is to compare complete processes within the algorithm:
        - highlight the best process for each objective metric using `best_train_metric`;
        - show the full ranking of configurations;
        - plot training and validation progress by epoch;
        - indicate whether there is already any test result linked to `best_prompt.json`.

        Notes:
        - The best process is defined by the highest `best_train_metric`, which represents the best training value reached during optimization.
        - When `best_train_metric` ties, tie-breaking uses `best_val_metric` and then the lower `time_prompt_optimization`.
        - Different objective metrics should not be compared directly; the notebook keeps rankings separated by metric.
        - If `out/test_explainability` does not exist yet, the test-results section will remain empty without breaking the analysis.
        """,
        **replacements,
    )

    imports = join_code_blocks(
        """
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import Markdown, display
        """,
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        """
        PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.utils.optimization_process_analysis import (
            discover_optimization_processes,
            load_process_bundle,
            summarize_processes,
        )
        """,
        NOTEBOOK_METRIC_COLOR_HELPERS,
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", 180)
        pd.set_option("display.max_columns", 50)

        PROJECT_ROOT
        """,
    )

    config = render_template(
        """
        ALGORITHM_NAME = "__ALGORITHM__"
        SEARCH_ROOT_REL = "__SEARCH_ROOT_REL__"
        SEARCH_ROOT = PROJECT_ROOT / SEARCH_ROOT_REL
        TEST_ROOT = PROJECT_ROOT / "out" / "test_explainability"

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("ALGORITHM_NAME:", ALGORITHM_NAME)
        print("SEARCH_ROOT:", SEARCH_ROOT)
        print("TEST_ROOT_EXISTS:", TEST_ROOT.exists())
        """,
        **replacements,
    )

    catalog = """
        process_catalog = discover_optimization_processes(PROJECT_ROOT, search_root=SEARCH_ROOT)
        if process_catalog.empty:
            raise FileNotFoundError("No optimization_process_metadata.json was found in " + str(SEARCH_ROOT))

        process_catalog = process_catalog.copy()
        process_catalog["early_label"] = "early_" + process_catalog["early_stopping"].astype(str).str.lower()
        process_catalog["process_label"] = (
            process_catalog["objective_metric"].astype(str)
            + " | "
            + process_catalog["representation_model"].astype(str)
            + " | "
            + process_catalog["early_label"].astype(str)
            + " | lambda="
            + process_catalog["mmr_lambda_quality"].astype(str)
            + " | pool="
            + process_catalog["mmr_pool_multiplier"].astype(str)
        )
        process_catalog["process_notebook_path_rel"] = process_catalog["process_dir_rel"] + "/plot_optimization_process.ipynb"

        display(Markdown("## Catalog of processes found in `" + SEARCH_ROOT_REL + "`"))
        display(
            summarize_processes(
                process_catalog,
                columns=[
                    "objective_metric",
                    "representation_model",
                    "early_stopping",
                    "mmr_lambda_quality",
                    "mmr_pool_multiplier",
                    "epochs_completed",
                    "best_train_metric",
                    "best_val_metric",
                    "saved_best_origin",
                    "process_dir_rel",
                ],
            )
        )
        print("Number of completed processes:", len(process_catalog))
    """

    ranking = """
        ranking_view = process_catalog.sort_values(
            by=["objective_metric", "best_train_metric", "best_val_metric", "time_prompt_optimization"],
            ascending=[True, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

        best_by_metric = (
            ranking_view.groupby("objective_metric", as_index=False)
            .first()
            .loc[
                :,
                [
                    "objective_metric",
                    "metric_name",
                    "representation_model",
                    "early_stopping",
                    "mmr_lambda_quality",
                    "mmr_pool_multiplier",
                    "best_train_metric",
                    "best_val_metric",
                    "epochs_completed",
                    "time_prompt_optimization",
                    "process_dir_rel",
                    "process_notebook_path_rel",
                ],
            ]
        )

        display(Markdown("## Best process by objective metric"))
        display(best_by_metric)
        display(
            Markdown(
                "> Ranking principal: `best_train_metric` decrescente. Desempates usam `best_val_metric` e depois menor `time_prompt_optimization`."
            )
        )
    """

    plots = """
        display(Markdown("## Full ranking by metric"))
        ranking_columns = [
            "process_label",
            "representation_model",
            "early_stopping",
            "mmr_lambda_quality",
            "mmr_pool_multiplier",
            "best_train_metric",
            "best_val_metric",
            "epochs_completed",
            "time_prompt_optimization",
            "process_notebook_path_rel",
        ]

        for objective_metric, metric_df in ranking_view.groupby("objective_metric", sort=True):
            ordered = metric_df.reset_index(drop=True)
            display(Markdown("### " + str(objective_metric)))
            display(ordered.loc[:, ranking_columns])

            chart_df = ordered.dropna(subset=["best_train_metric"]).copy()
            if chart_df.empty:
                display(Markdown("> No `best_train_metric` is available to plot in this group."))
                continue

            chart_colors = metric_shades(objective_metric, len(chart_df))
            fig_height = max(4.0, 0.65 * len(chart_df))
            fig, ax = plt.subplots(figsize=(13, fig_height))
            ax.barh(chart_df["process_label"], chart_df["best_train_metric"], color=chart_colors)
            ax.invert_yaxis()
            ax.set_xlabel("best_train_metric")
            ax.set_ylabel("process")
            ax.set_title("Best training values")

            max_value = chart_df["best_train_metric"].max()
            offset = max_value * 0.01 if pd.notna(max_value) and max_value != 0 else 0.01
            for index, value in enumerate(chart_df["best_train_metric"]):
                ax.text(value + offset, index, f"{value:.4f}", va="center")

            plt.tight_layout()
            plt.show()
    """

    linked_tests = """
        bundles = {}
        linked_rows = []

        for _, row in process_catalog.iterrows():
            bundle = load_process_bundle(Path(row["process_dir"]), PROJECT_ROOT)
            bundles[row["process_dir"]] = bundle
            linked_payload = bundle["linked_test_metadata"] or {}
            linked_rows.append(
                {
                    "objective_metric": row["objective_metric"],
                    "process_label": row["process_label"],
                    "linked_test_metadata_path": bundle["summary"]["linked_test_metadata_path"],
                    "test_metric_value": linked_payload.get("metric_value"),
                    "n_users": linked_payload.get("n_users"),
                    "time_to_explain": linked_payload.get("time_to_explain"),
                    "prompt_source": linked_payload.get("prompt_source"),
                }
            )

        linked_tests_df = pd.DataFrame(linked_rows)

        display(Markdown("## Link to test results"))
        if linked_tests_df["linked_test_metadata_path"].notna().any():
            display(linked_tests_df)
        else:
            display(
                Markdown(
                    "> No `responses_metadata.json` linked to `best_prompt.json` was found in `out/test_explainability`."
                )
            )
    """

    epoch_curves = """
        display(Markdown("## Per-epoch curves for the processes"))

        for objective_metric, metric_df in ranking_view.groupby("objective_metric", sort=True):
            display(Markdown("### " + str(objective_metric)))
            fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
            train_ax, val_ax = axes
            plotted_train = False
            plotted_val = False

            curve_colors = metric_shades(objective_metric, len(metric_df))

            for color, (_, row) in zip(curve_colors, metric_df.iterrows()):
                epochs_df = bundles[row["process_dir"]]["epochs_df"]
                if epochs_df.empty:
                    continue

                label = row["process_label"]
                if "train_metric" in epochs_df and epochs_df["train_metric"].notna().any():
                    train_ax.plot(
                        epochs_df["epoch"],
                        epochs_df["train_metric"],
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=label,
                    )
                    plotted_train = True

                if "val_metric" in epochs_df and epochs_df["val_metric"].notna().any():
                    val_ax.plot(
                        epochs_df["epoch"],
                        epochs_df["val_metric"],
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=label,
                    )
                    plotted_val = True

            train_ax.set_title("Treino")
            train_ax.set_xlabel("epoch")
            train_ax.set_ylabel("train_metric")

            val_ax.set_title("Validação")
            val_ax.set_xlabel("epoch")
            val_ax.set_ylabel("val_metric")

            if plotted_train:
                train_ax.legend(loc="best", fontsize=8)
            else:
                train_ax.text(0.5, 0.5, "No training data", ha="center", va="center", transform=train_ax.transAxes)

            if plotted_val:
                val_ax.legend(loc="best", fontsize=8)
            else:
                val_ax.text(0.5, 0.5, "No validation data", ha="center", va="center", transform=val_ax.transAxes)

            plt.tight_layout()
            plt.show()
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("config", config),
            code_cell("catalog", catalog),
            code_cell("ranking", ranking),
            code_cell("plots", plots),
            code_cell("linked-tests", linked_tests),
            code_cell("epoch-curves", epoch_curves),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_metric_notebook(algorithm: str, llm_method: str, objective_metric: str, search_root_rel: str) -> dict:
    replacements = {
        "ALGORITHM": algorithm,
        "LLM_METHOD": llm_method,
        "OBJECTIVE_METRIC": objective_metric,
        "SEARCH_ROOT_REL": search_root_rel,
    }

    intro = render_template(
        """
        # Analysis of prompt optimization metrics for `__ALGORITHM__` on `__OBJECTIVE_METRIC__`

        This notebook scans the completed processes in `__SEARCH_ROOT_REL__` and helps identify which configurations performed best within the objective metric `__OBJECTIVE_METRIC__`.

        The focus here is:
        - highlight the best overall combination;
        - show the best combination by representation, when more than one exists;
        - show the full configuration ranking;
        - link processes to test results when a matching `responses_metadata.json` exists;
        - plot aggregated training and validation curves to compare processes;
        - and open a detailed per-process section with per-epoch curves, main prompts, and an epoch table.

        Notes:
        - The best process is defined by the highest `best_train_metric`.
        - On ties, tie-breaking uses `best_val_metric` and then the lower `time_prompt_optimization`.
        - This notebook is restricted to the `__OBJECTIVE_METRIC__` metric, so comparisons already happen within the same objective.
        - The color scheme is fixed across all plots: `etd` in blue, `sep` in orange, and `sep_etd_f1` in green.
        """,
        **replacements,
    )

    imports = join_code_blocks(
        """
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import Markdown, display
        """,
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        """
        PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.utils.optimization_process_analysis import (
            discover_optimization_processes,
            load_process_bundle,
            summarize_processes,
        )
        """,
        NOTEBOOK_METRIC_COLOR_HELPERS,
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", 180)
        pd.set_option("display.max_columns", 60)

        PROJECT_ROOT
        """,
    )

    config = render_template(
        """
        ALGORITHM_NAME = "__ALGORITHM__"
        OBJECTIVE_METRIC = "__OBJECTIVE_METRIC__"
        SEARCH_ROOT_REL = "__SEARCH_ROOT_REL__"
        SEARCH_ROOT = PROJECT_ROOT / SEARCH_ROOT_REL
        TEST_ROOT = PROJECT_ROOT / "out" / "test_explainability"

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("ALGORITHM_NAME:", ALGORITHM_NAME)
        print("OBJECTIVE_METRIC:", OBJECTIVE_METRIC)
        print("SEARCH_ROOT:", SEARCH_ROOT)
        print("TEST_ROOT_EXISTS:", TEST_ROOT.exists())
        """,
        **replacements,
    )

    catalog = """
        process_catalog = discover_optimization_processes(PROJECT_ROOT, search_root=SEARCH_ROOT)
        if process_catalog.empty:
            raise FileNotFoundError("No optimization_process_metadata.json was found in " + str(SEARCH_ROOT))

        process_catalog = process_catalog.copy()
        process_catalog["early_label"] = "early_" + process_catalog["early_stopping"].astype(str).str.lower()
        process_catalog["process_label"] = (
            process_catalog["representation_model"].astype(str)
            + " | "
            + process_catalog["early_label"].astype(str)
            + " | lambda="
            + process_catalog["mmr_lambda_quality"].astype(str)
            + " | pool="
            + process_catalog["mmr_pool_multiplier"].astype(str)
        )
        process_catalog["process_notebook_path_rel"] = process_catalog["process_dir_rel"] + "/plot_optimization_process.ipynb"

        display(Markdown("## Catalog of processes found in `" + SEARCH_ROOT_REL + "`"))
        display(
            summarize_processes(
                process_catalog,
                columns=[
                    "objective_metric",
                    "representation_model",
                    "early_stopping",
                    "mmr_lambda_quality",
                    "mmr_pool_multiplier",
                    "epochs_completed",
                    "best_train_epoch",
                    "best_train_metric",
                    "best_val_epoch",
                    "best_val_metric",
                    "saved_best_origin",
                    "process_dir_rel",
                ],
            )
        )
        print("Number of completed processes:", len(process_catalog))
    """

    ranking = """
        ranking_view = process_catalog.sort_values(
            by=["best_train_metric", "best_val_metric", "time_prompt_optimization"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranking_view.insert(0, "rank", range(1, len(ranking_view) + 1))
        best_row = ranking_view.iloc[0] if not ranking_view.empty else None

        display(Markdown("## Best overall configuration"))
        display(
            ranking_view.head(1).loc[
                :,
                [
                    "rank",
                    "process_label",
                    "representation_model",
                    "mmr_lambda_quality",
                    "mmr_pool_multiplier",
                    "best_train_epoch",
                    "best_train_metric",
                    "best_val_epoch",
                    "best_val_metric",
                    "time_prompt_optimization",
                    "process_notebook_path_rel",
                ],
            ]
        )
        display(
            Markdown(
                "> Criterion: descending `best_train_metric`. Ties use `best_val_metric` and then the lower `time_prompt_optimization`."
            )
        )

        display(Markdown("## Best configuration by representation"))
        best_by_repr = (
            ranking_view.groupby("representation_model", as_index=False)
            .first()
            .loc[
                :,
                [
                    "representation_model",
                    "process_label",
                    "mmr_lambda_quality",
                    "mmr_pool_multiplier",
                    "best_train_epoch",
                    "best_train_metric",
                    "best_val_epoch",
                    "best_val_metric",
                    "process_notebook_path_rel",
                ],
            ]
        )
        display(best_by_repr)

        display(Markdown("## Ranking completo"))
        ranking_columns = [
            "rank",
            "process_label",
            "representation_model",
            "early_stopping",
            "mmr_lambda_quality",
            "mmr_pool_multiplier",
            "best_train_epoch",
            "best_train_metric",
            "best_val_epoch",
            "best_val_metric",
            "epochs_completed",
            "time_prompt_optimization",
            "saved_best_origin",
            "process_notebook_path_rel",
        ]
        display(ranking_view.loc[:, ranking_columns])

        chart_df = ranking_view.dropna(subset=["best_train_metric"]).copy()
        chart_colors = metric_shades(OBJECTIVE_METRIC, len(chart_df))
        fig_height = max(4.0, 0.65 * len(chart_df))
        fig, ax = plt.subplots(figsize=(13, fig_height))
        ax.barh(chart_df["process_label"], chart_df["best_train_metric"], color=chart_colors)
        ax.invert_yaxis()
        ax.set_xlabel("best_train_metric")
        ax.set_ylabel("process")
        ax.set_title("Best training values")

        max_value = chart_df["best_train_metric"].max()
        offset = max_value * 0.01 if pd.notna(max_value) and max_value != 0 else 0.01
        for index, value in enumerate(chart_df["best_train_metric"]):
            ax.text(value + offset, index, f"{value:.4f}", va="center")

        plt.tight_layout()
        plt.show()
    """

    linked_tests = """
        bundles = {}
        linked_rows = []

        for _, row in process_catalog.iterrows():
            bundle = load_process_bundle(Path(row["process_dir"]), PROJECT_ROOT)
            bundles[row["process_dir"]] = bundle
            linked_payload = bundle["linked_test_metadata"] or {}
            linked_rows.append(
                {
                    "process_label": row["process_label"],
                    "linked_test_metadata_path": bundle["summary"]["linked_test_metadata_path"],
                    "test_metric_value": linked_payload.get("metric_value"),
                    "n_users": linked_payload.get("n_users"),
                    "time_to_explain": linked_payload.get("time_to_explain"),
                    "prompt_source": linked_payload.get("prompt_source"),
                }
            )

        linked_tests_df = pd.DataFrame(linked_rows)

        display(Markdown("## Link to test results"))
        if linked_tests_df["linked_test_metadata_path"].notna().any():
            display(linked_tests_df)
        else:
            display(
                Markdown(
                    "> No `responses_metadata.json` linked to `best_prompt.json` was found in `out/test_explainability`."
                )
            )
    """

    aggregate_curves = """
        if OBJECTIVE_METRIC == "sep_etd_f1":
            display(Markdown("## Curves for the best overall configuration"))
            if best_row is None:
                display(Markdown("> No process was found for the `sep_etd_f1` metric."))
            else:
                display(
                    best_row.loc[
                        [
                            "rank",
                            "process_label",
                            "representation_model",
                            "mmr_lambda_quality",
                            "mmr_pool_multiplier",
                            "best_train_epoch",
                            "best_train_metric",
                            "best_val_epoch",
                            "best_val_metric",
                            "time_prompt_optimization",
                            "process_notebook_path_rel",
                        ]
                    ].to_frame(name="value")
                )

                epochs_df = ensure_balance_metrics(bundles[best_row["process_dir"]]["epochs_df"].copy())
                if epochs_df.empty:
                    display(Markdown("> The best process has no epoch history to plot."))
                else:
                    epoch_positions = epochs_df["epoch"] + 1
                    combined_metric_names = preferred_metric_order(
                        [
                            metric_name
                            for metric_name in ["sep", "etd", "sep_etd_f1", "geom_mean", "mean_balance"]
                            if (
                                f"train_score_{metric_name}" in epochs_df.columns
                                or f"val_score_{metric_name}" in epochs_df.columns
                            )
                        ]
                    )

                    display(Markdown(
                        "**Métricas derivadas neste notebook**: "
                        "`geom_mean = sqrt(SEP * ETD)` e "
                        "`mean_balance = ((SEP + ETD) / 2) * (1 - abs(SEP - ETD))`."
                    ))

                    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)
                    for ax, split_name, title in zip(
                        axes,
                        ["train", "val"],
                        ["Treino", "Validação"],
                    ):
                        plotted_any = False
                        for metric_name in combined_metric_names:
                            score_col = f"{split_name}_score_{metric_name}"
                            if score_col not in epochs_df.columns or not epochs_df[score_col].notna().any():
                                continue

                            ax.plot(
                                epoch_positions,
                                epochs_df[score_col],
                                marker="o",
                                linewidth=2,
                                color=metric_color(metric_name),
                                label=metric_label(metric_name),
                            )
                            plotted_any = True

                        ax.set_title(f"{title}: metric comparison")
                        ax.set_xlabel("epoch")
                        ax.set_ylabel("score")
                        ax.set_xticks(epoch_positions.tolist())
                        if plotted_any:
                            ax.legend(loc="best", ncol=2)
                        else:
                            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)

                    plt.tight_layout()
                    plt.show()

                    metric_names = []
                    for metric_name in [OBJECTIVE_METRIC, "sep", "etd"]:
                        if metric_name not in metric_names:
                            metric_names.append(metric_name)

                    for metric_name in metric_names:
                        train_col = f"train_score_{metric_name}"
                        val_col = f"val_score_{metric_name}"
                        if train_col not in epochs_df.columns and val_col not in epochs_df.columns:
                            continue

                        line_color = metric_color(metric_name)
                        fig, ax = plt.subplots(figsize=(14, 6))

                        if train_col in epochs_df.columns and epochs_df[train_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[train_col],
                                marker="o",
                                linewidth=2,
                                linestyle="-",
                                color=line_color,
                                label=f"train::{metric_name}",
                            )

                        if val_col in epochs_df.columns and epochs_df[val_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[val_col],
                                marker="o",
                                linewidth=2,
                                linestyle="--",
                                color=line_color,
                                label=f"val::{metric_name}",
                            )

                        ax.set_title(f"Progression of {metric_name} in the best configuration")
                        ax.set_xlabel("epoch")
                        ax.set_ylabel("score")
                        ax.set_xticks(epoch_positions.tolist())
                        ax.legend(loc="best")
                        plt.tight_layout()
                        plt.show()
        else:
            display(Markdown("## Aggregated curves by process"))
            fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
            train_ax, val_ax = axes
            plotted_train = False
            plotted_val = False

            curve_colors = metric_shades(OBJECTIVE_METRIC, len(ranking_view))

            for color, (_, row) in zip(curve_colors, ranking_view.iterrows()):
                epochs_df = bundles[row["process_dir"]]["epochs_df"]
                if epochs_df.empty:
                    continue

                label = row["process_label"]
                if "train_metric" in epochs_df and epochs_df["train_metric"].notna().any():
                    train_ax.plot(
                        epochs_df["epoch"] + 1,
                        epochs_df["train_metric"],
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=label,
                    )
                    plotted_train = True

                if "val_metric" in epochs_df and epochs_df["val_metric"].notna().any():
                    val_ax.plot(
                        epochs_df["epoch"] + 1,
                        epochs_df["val_metric"],
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=label,
                    )
                    plotted_val = True

            train_ax.set_title("Treino")
            train_ax.set_xlabel("epoch")
            train_ax.set_ylabel("train_metric")

            val_ax.set_title("Validação")
            val_ax.set_xlabel("epoch")
            val_ax.set_ylabel("val_metric")

            if plotted_train:
                train_ax.legend(loc="best", fontsize=8)
            else:
                train_ax.text(0.5, 0.5, "No training data", ha="center", va="center", transform=train_ax.transAxes)

            if plotted_val:
                val_ax.legend(loc="best", fontsize=8)
            else:
                val_ax.text(0.5, 0.5, "No validation data", ha="center", va="center", transform=val_ax.transAxes)

            plt.tight_layout()
            plt.show()
    """

    per_process_detail = """
        if OBJECTIVE_METRIC == "sep_etd_f1":
            display(Markdown("## Details of the best configuration"))
            detail_iterator = ranking_view.head(1).iterrows()
        else:
            display(Markdown("## Per-process details"))
            detail_iterator = ranking_view.iterrows()

        for _, row in detail_iterator:
            bundle = bundles[row["process_dir"]]
            summary = bundle["summary"]
            epochs_df = bundle["epochs_df"].copy()
            prompt_df = bundle["prompt_df"].copy()
            prompt_display_df = prompt_df.reset_index(drop=True).copy()
            prompt_display_df.insert(0, "prompt_no", range(1, len(prompt_display_df) + 1))

            if OBJECTIVE_METRIC == "sep_etd_f1":
                epochs_df = ensure_balance_metrics(epochs_df)

            display(Markdown("### " + str(row["process_label"])))
            display(
                pd.DataFrame(
                    [
                        {
                            "process_dir_rel": summary["process_dir_rel"],
                            "representation_model": summary["representation_model"],
                            "early_stopping": summary["early_stopping"],
                            "mmr_lambda_quality": summary["mmr_lambda_quality"],
                            "mmr_pool_multiplier": summary["mmr_pool_multiplier"],
                            "best_train_epoch": summary["best_train_epoch"],
                            "best_train_metric": summary["best_train_metric"],
                            "best_val_epoch": summary["best_val_epoch"],
                            "best_val_metric": summary["best_val_metric"],
                            "saved_best_origin": summary["saved_best_origin"],
                            "linked_test_metadata_path": summary["linked_test_metadata_path"],
                        }
                    ]
                )
            )

            display(Markdown("**Prompts principais**"))
            display(
                prompt_display_df[
                    ["prompt_no", "prompt_role", "epoch", "source_metric", "score", "prompt_preview"]
                ]
            )

            if not epochs_df.empty:
                epoch_positions = epochs_df["epoch"] + 1
                if OBJECTIVE_METRIC == "sep_etd_f1":
                    metric_names = []
                    for metric_name in [OBJECTIVE_METRIC, "sep", "etd"]:
                        if metric_name not in metric_names:
                            metric_names.append(metric_name)

                    for metric_name in metric_names:
                        train_col = f"train_score_{metric_name}"
                        val_col = f"val_score_{metric_name}"
                        if train_col not in epochs_df.columns and val_col not in epochs_df.columns:
                            continue

                        line_color = metric_color(metric_name)
                        fig, ax = plt.subplots(figsize=(14, 6))
                        if train_col in epochs_df.columns and epochs_df[train_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[train_col],
                                marker="o",
                                linestyle="-",
                                color=line_color,
                                label=f"train:{metric_name}",
                            )

                        if val_col in epochs_df.columns and epochs_df[val_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[val_col],
                                marker="o",
                                linestyle="--",
                                color=line_color,
                                label=f"val:{metric_name}",
                            )

                        ax.set_title(f"Progression of {metric_name} by epoch")
                        ax.set_xlabel("epoch")
                        ax.set_ylabel("score")
                        ax.set_xticks(epoch_positions.tolist())
                        ax.legend(loc="best")
                        plt.tight_layout()
                        plt.show()
                else:
                    metric_columns = sorted(
                        {
                            column.replace("train_score_", "")
                            for column in epochs_df.columns
                            if column.startswith("train_score_")
                        }
                    )
                    all_metric_names = []
                    for metric_name in [summary["objective_metric"], *metric_columns]:
                        if metric_name not in all_metric_names:
                            all_metric_names.append(metric_name)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    for metric_name in all_metric_names:
                        train_col = f"train_score_{metric_name}"
                        val_col = f"val_score_{metric_name}"
                        line_color = metric_color(metric_name)

                        if train_col in epochs_df.columns and epochs_df[train_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[train_col],
                                marker="o",
                                linestyle="-",
                                color=line_color,
                                label=f"train:{metric_name}",
                            )

                        if val_col in epochs_df.columns and epochs_df[val_col].notna().any():
                            ax.plot(
                                epoch_positions,
                                epochs_df[val_col],
                                marker="o",
                                linestyle="--",
                                color=line_color,
                                label=f"val:{metric_name}",
                            )

                    ax.set_title("Metric progression by epoch")
                    ax.set_xlabel("epoch")
                    ax.set_ylabel("score")
                    ax.set_xticks(epoch_positions.tolist())
                    ax.legend(loc="best")
                    plt.tight_layout()
                    plt.show()

                epoch_columns = [
                    "epoch",
                    "generated_new_prompt",
                    "train_metric",
                    "val_metric",
                    "val_improvement_vs_prev",
                    "is_best_train_epoch",
                    "is_best_val_epoch",
                    "is_saved_best_epoch",
                    "time_spent_instruction",
                    "time_spent_train_eval",
                    "time_spent_val_eval",
                    "mmr_selected_reference_epochs",
                    "prompt_preview",
                ]
                if OBJECTIVE_METRIC == "sep_etd_f1":
                    score_columns = []
                    for metric_name in [OBJECTIVE_METRIC, "sep", "etd", "geom_mean", "mean_balance"]:
                        for column in (f"train_score_{metric_name}", f"val_score_{metric_name}"):
                            if column in epochs_df.columns and column not in score_columns:
                                score_columns.append(column)
                else:
                    score_columns = sorted(
                        [column for column in epochs_df.columns if column.startswith("train_score_") or column.startswith("val_score_")]
                    )
                display(Markdown("**Epoch table**"))
                display(epochs_df[epoch_columns + score_columns])
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("config", config),
            code_cell("catalog", catalog),
            code_cell("ranking", ranking),
            code_cell("linked-tests", linked_tests),
            code_cell("aggregate-curves", aggregate_curves),
            code_cell("per-process-detail", per_process_detail),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_process_notebook(process_dir_rel: str) -> dict:
    replacements = {
        "PROCESS_DIR_REL": process_dir_rel,
    }

    intro = render_template(
        """
        # Per-process analysis of prompt optimization

        This notebook analyzes only the process in `__PROCESS_DIR_REL__`.

        Suggested workflow:
        1. inspect the summary and the artifacts loaded from this process directory;
        2. inspect the base prompt, the best saved prompt, the best validation prompt, and the metric behavior by epoch;
        3. if needed, adjust `PROCESS_DIR` to another compatible process directory.

        Important notes:
        - `best_prompt.json` currently stores the **best training prompt** (`best_on_train`).
        - When the best validation result occurs at a different epoch, the notebook highlights that divergence.
        - The color scheme is fixed across all plots: `etd` in blue, `sep` in orange, and `sep_etd_f1` in green.
        """,
        **replacements,
    )

    imports = join_code_blocks(
        """
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import Markdown, display
        """,
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        """
        PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.utils.optimization_process_analysis import (
            discover_optimization_processes,
            load_process_bundle,
            summarize_processes,
        )
        """,
        NOTEBOOK_METRIC_COLOR_HELPERS,
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.max_columns", 50)

        PROJECT_ROOT
        """,
    )

    config = render_template(
        """
        ANALYSIS_ROOT_REL = "__PROCESS_DIR_REL__"
        ANALYSIS_ROOT = PROJECT_ROOT / ANALYSIS_ROOT_REL
        PROCESS_INDEX = 0
        PROCESS_DIR = str(ANALYSIS_ROOT)

        process_catalog = discover_optimization_processes(PROJECT_ROOT, search_root=ANALYSIS_ROOT)
        if process_catalog.empty:
            raise FileNotFoundError(f"No optimization_process_metadata.json was found in {ANALYSIS_ROOT}.")

        display(Markdown(f"## Process at `{ANALYSIS_ROOT_REL}`"))
        display(summarize_processes(process_catalog))

        selected_process_path = Path(PROCESS_DIR) if PROCESS_DIR is not None else Path(process_catalog.loc[PROCESS_INDEX, "process_dir"])
        print(f"Analysis root directory: {ANALYSIS_ROOT}")
        print(f"Selected process: {selected_process_path}")
        """,
        **replacements,
    )

    summary = """
        bundle = load_process_bundle(selected_process_path, PROJECT_ROOT)
        summary_df = pd.DataFrame(bundle["summary"].items(), columns=["field", "value"])

        display(Markdown("## Summary of the selected process"))
        display(summary_df)

        saved_best_origin = bundle["summary"]["saved_best_origin"]
        if saved_best_origin == "train":
            display(Markdown(
                "> `best_prompt.json` corresponds to the best **training** prompt. The best **validation** prompt occurred at a different epoch."
            ))
        elif saved_best_origin == "train_and_validation":
            display(Markdown(
                "> `best_prompt.json` matches both the best **training** prompt and the best **validation** prompt."
            ))
        elif saved_best_origin == "validation":
            display(Markdown(
                "> `best_prompt.json` matches the best **validation** prompt."
            ))
        else:
            display(Markdown(
                "> Could not determine whether `best_prompt.json` matches the best training prompt or the best validation prompt."
            ))

        display(Markdown("## Comparison between the base prompt, best saved prompt, best training prompt, and best validation prompt"))
        prompt_display_df = bundle["prompt_df"].reset_index(drop=True).copy()
        prompt_display_df.insert(0, "prompt_no", range(1, len(prompt_display_df) + 1))
        display(
            prompt_display_df[
                ["prompt_no", "prompt_role", "epoch", "source_metric", "score", "prompt_preview"]
            ]
        )
    """

    full_prompts = """
        display(Markdown("## Full text of the main prompts"))
        prompt_text_df = bundle["prompt_df"].reset_index(drop=True)
        for prompt_number, row in enumerate(prompt_text_df.itertuples(index=False), start=1):
            if not isinstance(row.prompt_text, str) or not row.prompt_text.strip():
                continue
            score_text = "-" if row.score is None or pd.isna(row.score) else f"{row.score:.6f}"
            epoch_text = "-" if row.epoch is None or pd.isna(row.epoch) else int(row.epoch)
            display(Markdown(f"### Prompt {prompt_number} | {row.prompt_role} | epoch={epoch_text} | score={score_text}"))
            print(row.prompt_text)
            print()
    """

    charts = """
        epochs_df = bundle["epochs_df"].copy()
        objective_metric = bundle["summary"]["objective_metric"]

        epochs_df = ensure_balance_metrics(epochs_df)

        if epochs_df.empty:
            display(Markdown("## Gráficos"))
            display(Markdown("> No epoch was recorded for this process."))
        else:
            metric_columns = preferred_metric_order(
                sorted(
                    {
                        column.replace("train_score_", "")
                        for column in epochs_df.columns
                        if column.startswith("train_score_")
                    }
                )
            )
            all_metric_names = []
            for metric_name in [objective_metric, *metric_columns]:
                if metric_name not in all_metric_names:
                    all_metric_names.append(metric_name)

            metric_colors = {
                metric_name: metric_color(metric_name)
                for metric_name in all_metric_names
            }
            epoch_positions = epochs_df["epoch"].tolist()
            epoch_labels = [epoch + 1 for epoch in epoch_positions]
            best_train_epochs = epochs_df.loc[epochs_df["is_best_train_epoch"], "epoch"].tolist()
            best_val_epochs = epochs_df.loc[epochs_df["is_best_val_epoch"], "epoch"].tolist()
            shared_best_epochs = sorted(set(best_train_epochs) & set(best_val_epochs))
            train_only_epochs = [epoch for epoch in best_train_epochs if epoch not in shared_best_epochs]
            val_only_epochs = [epoch for epoch in best_val_epochs if epoch not in shared_best_epochs]

            display(Markdown(f"## Plot focused on the objective metric ({objective_metric})"))
            best_train_text = ", ".join(str(epoch + 1) for epoch in train_only_epochs) if train_only_epochs else "none"
            best_val_text = ", ".join(str(epoch + 1) for epoch in val_only_epochs) if val_only_epochs else "none"
            shared_best_text = ", ".join(str(epoch + 1) for epoch in shared_best_epochs) if shared_best_epochs else "none"
            display(Markdown(
                f"Vertical lines: purple = best epoch only in training ({best_train_text}); pink = best epoch only in validation ({best_val_text}); gray = the same epoch was best in both training and validation ({shared_best_text})."
            ))
            fig, ax = plt.subplots(figsize=(14, 5))

            ax.plot(
                epochs_df["epoch"],
                epochs_df["train_metric"],
                marker="o",
                linewidth=2,
                linestyle="-",
                color=metric_colors[objective_metric],
                label=f"train::{objective_metric}",
            )
            if epochs_df["val_metric"].notna().any():
                ax.plot(
                    epochs_df["epoch"],
                    epochs_df["val_metric"],
                    marker="o",
                    linewidth=2,
                    linestyle="--",
                    color=metric_colors[objective_metric],
                    label=f"val::{objective_metric}",
                )

            for epoch in train_only_epochs:
                ax.axvline(epoch, color="#7b2cbf", linestyle=":", linewidth=2, alpha=0.9)
            for epoch in val_only_epochs:
                ax.axvline(epoch, color="#e75480", linestyle=":", linewidth=2, alpha=0.9)
            for epoch in shared_best_epochs:
                ax.axvline(epoch, color="#6c757d", linestyle="-", linewidth=2.2, alpha=0.95)

            ax.set_title(f"Objective metric progression ({objective_metric})")
            ax.set_xlabel("epoch")
            ax.set_ylabel("score")
            ax.set_xticks(epoch_positions)
            ax.set_xticklabels(epoch_labels)
            ax.legend(loc="best")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

            display(Markdown("## Plot with all metrics"))
            if metric_columns:
                if "geom_mean" in metric_columns or "mean_balance" in metric_columns:
                    display(Markdown(
                        "`geom_mean = sqrt(SEP * ETD)` e "
                        "`mean_balance = ((SEP + ETD) / 2) * (1 - abs(SEP - ETD))`."
                    ))
                fig, ax = plt.subplots(figsize=(14, 6))
                for metric_name in metric_columns:
                    train_col = f"train_score_{metric_name}"
                    val_col = f"val_score_{metric_name}"
                    line_color = metric_colors[metric_name]
                    ax.plot(
                        epochs_df["epoch"],
                        epochs_df[train_col],
                        marker="o",
                        linewidth=1.8,
                        linestyle="-",
                        color=line_color,
                        label=f"train::{metric_name}",
                    )
                    if val_col in epochs_df.columns and epochs_df[val_col].notna().any():
                        ax.plot(
                            epochs_df["epoch"],
                            epochs_df[val_col],
                            marker="o",
                            linewidth=1.8,
                            linestyle="--",
                            color=line_color,
                            label=f"val::{metric_name}",
                        )
                ax.set_title("Progression of all metrics by epoch")
                ax.set_xlabel("epoch")
                ax.set_ylabel("score")
                ax.set_xticks(epoch_positions)
                ax.set_xticklabels(epoch_labels)
                ax.legend(loc="best", ncol=2)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            else:
                display(Markdown("No detailed metrics were found for this process."))
    """

    epoch_table = """
        display(Markdown("## Epoch table"))
        if epochs_df.empty:
            display(Markdown("> No epoch was recorded for this process."))
        else:
            epoch_columns = [
                "epoch",
                "generated_new_prompt",
                "train_metric",
                "val_metric",
                "val_improvement_vs_prev",
                "is_best_train_epoch",
                "is_best_val_epoch",
                "is_saved_best_epoch",
                "time_spent_instruction",
                "time_spent_train_eval",
                "time_spent_val_eval",
                "mmr_selected_reference_epochs",
                "prompt_preview",
            ]
            score_columns = []
            for metric_name in preferred_metric_order(
                sorted(
                    {
                        column.replace("train_score_", "").replace("val_score_", "")
                        for column in epochs_df.columns
                        if column.startswith("train_score_") or column.startswith("val_score_")
                    }
                )
            ):
                for column in (f"train_score_{metric_name}", f"val_score_{metric_name}"):
                    if column in epochs_df.columns:
                        score_columns.append(column)
            display(epochs_df[epoch_columns + score_columns])
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("config", config),
            code_cell("summary", summary),
            code_cell("full-prompts", full_prompts),
            code_cell("charts", charts),
            code_cell("epoch-table", epoch_table),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_global_test_results_notebook(llm_method: str) -> dict:
    replacements = {"LLM_METHOD": llm_method}

    intro = render_template(
        """
        # Global table of test results

        This notebook builds tables with the test results of model `__LLM_METHOD__`, split by recommendation method, combining:
        - the `without_optimization` result, shown in the first row of each method;
        - and all `with_optimization` results found for the configurations in `out/prompt_optimization/__LLM_METHOD__`.

        The goal here is to provide a consolidated view by algorithm, covering all methods found in the current `out` directory.
        """,
        **replacements,
    )

    imports = """
        import json
        import warnings
        from pathlib import Path

        import pandas as pd
        from IPython.display import display
    """

    paths = join_code_blocks(
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        render_template(
            """
            PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
            WORKSPACE_ROOT = PROJECT_ROOT.parent
            MODEL_NAME = "__LLM_METHOD__"
            PROMPT_OPT_ROOT = PROJECT_ROOT / "out" / "prompt_optimization" / MODEL_NAME
            TEST_ROOT = PROJECT_ROOT / "out" / "test_explainability"
            WITHOUT_OPT_TEST_ROOT = TEST_ROOT / "without_optimization" / MODEL_NAME
            WITH_OPT_TEST_ROOT = TEST_ROOT / "with_optimization" / MODEL_NAME

            print(f"PROJECT_ROOT: {display_workspace_relative(PROJECT_ROOT, WORKSPACE_ROOT)}")
            print(f"PROMPT_OPT_ROOT: {display_workspace_relative(PROMPT_OPT_ROOT, WORKSPACE_ROOT)}")
            print(
                "WITHOUT_OPT_TEST_ROOT: "
                f"{display_workspace_relative(WITHOUT_OPT_TEST_ROOT, WORKSPACE_ROOT)}"
            )
            print(
                "WITH_OPT_TEST_ROOT: "
                f"{display_workspace_relative(WITH_OPT_TEST_ROOT, WORKSPACE_ROOT)}"
            )
            """,
            **replacements,
        ),
    )

    helpers = """
        def lambda_to_float(lambda_name: str | None) -> float | None:
            if not lambda_name or not lambda_name.startswith("mmr_lambda_"):
                return None
            value = lambda_name.replace("mmr_lambda_", "")
            return float(value.replace("_", "."))


        def find_named_parent(path: Path, prefix: str, default: str | None = None) -> str | None:
            for parent in path.parents:
                if parent.name.startswith(prefix):
                    return parent.name
            return default


        def discover_prompt_optimization_algorithms(prompt_opt_root: Path) -> list[str]:
            if not prompt_opt_root.exists():
                return []
            return sorted(path.name for path in prompt_opt_root.iterdir() if path.is_dir())


        def load_test_metadata(metadata_path: Path) -> dict:
            return json.loads(metadata_path.read_text(encoding="utf-8"))


        def discover_test_algorithms(test_root: Path) -> list[str]:
            if not test_root.exists():
                return []
            return sorted(path.name for path in test_root.iterdir() if path.is_dir())


        def discover_without_optimization_results(test_root: Path, valid_algorithms: list[str]) -> pd.DataFrame:
            rows = []

            for algorithm in valid_algorithms:
                algorithm_dir = test_root / algorithm
                if not algorithm_dir.exists():
                    continue

                for metadata_path in sorted(algorithm_dir.rglob("responses_metadata.json")):
                    payload = load_test_metadata(metadata_path)
                    args = payload.get("args", {})
                    rows.append(
                        {
                            "optimization_mode": "without_optimization",
                            "algorithm": algorithm,
                            "metric": payload.get("metric", args.get("metric", "metric")),
                            "metric_name": payload.get("metric_name", "METRIC"),
                            "metric_value": payload.get("metric_value"),
                            "repr_model": pd.NA,
                            "early_profile": pd.NA,
                            "mmr_lambda": pd.NA,
                            "lambda_value": pd.NA,
                            "mmr_pool": pd.NA,
                            "prompt_source": payload.get("prompt_source", "desconhecido"),
                            "llm_method": args.get("llm_method", "desconhecido"),
                            "n_users": payload.get("n_users"),
                            "time_to_explain": payload.get("time_to_explain"),
                            "best_prompt_path": display_path_string(payload.get("best_prompt_path"), WORKSPACE_ROOT),
                            "responses_metadata_path": display_workspace_relative(metadata_path, WORKSPACE_ROOT),
                        }
                    )

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame(rows)


        def discover_with_optimization_results(test_root: Path, valid_algorithms: list[str]) -> pd.DataFrame:
            rows = []

            for algorithm in valid_algorithms:
                algorithm_dir = test_root / algorithm
                if not algorithm_dir.exists():
                    continue

                for metadata_path in sorted(algorithm_dir.rglob("responses_metadata.json")):
                    payload = load_test_metadata(metadata_path)
                    args = payload.get("args", {})
                    mmr_lambda = find_named_parent(metadata_path, "mmr_lambda_", None)

                    rows.append(
                        {
                            "optimization_mode": "with_optimization",
                            "algorithm": algorithm,
                            "metric": payload.get("metric", args.get("metric", "metric")),
                            "metric_name": payload.get("metric_name", "METRIC"),
                            "metric_value": payload.get("metric_value"),
                            "repr_model": find_named_parent(metadata_path, "repr_", pd.NA),
                            "early_profile": find_named_parent(metadata_path, "early_", pd.NA),
                            "mmr_lambda": mmr_lambda or pd.NA,
                            "lambda_value": lambda_to_float(mmr_lambda),
                            "mmr_pool": find_named_parent(metadata_path, "mmr_pool_", pd.NA),
                            "prompt_source": payload.get("prompt_source", "desconhecido"),
                            "llm_method": args.get("llm_method", payload.get("best_prompt_model", "desconhecido")),
                            "n_users": payload.get("n_users"),
                            "time_to_explain": payload.get("time_to_explain"),
                            "best_prompt_path": display_path_string(payload.get("best_prompt_path"), WORKSPACE_ROOT),
                            "responses_metadata_path": display_workspace_relative(metadata_path, WORKSPACE_ROOT),
                        }
                    )

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame(rows)


        def build_consolidated_table(
            prompt_opt_root: Path,
            without_opt_test_root: Path,
            with_opt_test_root: Path,
        ) -> pd.DataFrame:
            preferred_columns = [
                "optimization_mode",
                "algorithm",
                "llm_method",
                "metric",
                "metric_name",
                "metric_value",
                "repr_model",
                "early_profile",
                "mmr_lambda",
                "lambda_value",
                "mmr_pool",
                "prompt_source",
                "n_users",
                "time_to_explain",
                "best_prompt_path",
                "responses_metadata_path",
            ]

            algorithms = sorted(
                set(discover_prompt_optimization_algorithms(prompt_opt_root))
                | set(discover_test_algorithms(without_opt_test_root))
                | set(discover_test_algorithms(with_opt_test_root))
            )
            without_opt = discover_without_optimization_results(without_opt_test_root, algorithms)
            with_opt = discover_with_optimization_results(with_opt_test_root, algorithms)

            rows = []
            for frame in (without_opt, with_opt):
                if frame.empty:
                    continue
                rows.extend(frame.reindex(columns=preferred_columns).to_dict(orient="records"))

            if not rows:
                return pd.DataFrame(columns=preferred_columns)

            consolidated = pd.DataFrame(rows, columns=preferred_columns)
            consolidated["optimization_order"] = consolidated["optimization_mode"].map(
                {"without_optimization": 0, "with_optimization": 1}
            )
            consolidated = consolidated.sort_values(
                by=["algorithm", "metric", "optimization_order", "repr_model", "lambda_value", "mmr_pool"],
                na_position="last",
            ).reset_index(drop=True)

            return consolidated[preferred_columns]
    """

    table = """
        consolidated_table = build_consolidated_table(
            prompt_opt_root=PROMPT_OPT_ROOT,
            without_opt_test_root=WITHOUT_OPT_TEST_ROOT,
            with_opt_test_root=WITH_OPT_TEST_ROOT,
        )

        if consolidated_table.empty:
            warnings.warn("No test result was found to build the global table.")
        else:
            print(f"Rows in the consolidated table: {len(consolidated_table)}")
            for algorithm, algorithm_table in consolidated_table.groupby("algorithm", sort=False):
                print(f"\\nAlgoritmo: {algorithm}")
                display(algorithm_table.reset_index(drop=True))
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("paths", paths),
            code_cell("helpers", helpers),
            code_cell("table", table),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_global_objective_comparison_notebook(llm_method: str) -> dict:
    replacements = {"LLM_METHOD": llm_method}

    intro = render_template(
        """
        # Global tables by optimization objective

        This notebook consolidates, for model `__LLM_METHOD__`, comparisons between:
        - `lod`;
        - `llama / initial_system_prompt` (`without_optimization` result);
        - `llama / best_system_prompt` (best `with_optimization` configuration for the analyzed objective).

        The output is organized by optimization objective:
        - `sep`
        - `etd`
        - `sep_etd_f1`

        For each objective, the notebook:
        1. chooses the best optimized process per algorithm based on that objective's own target metric;
        2. shows a summary table of the selected best configuration;
        3. displays three tables with the `SEP`, `ETD`, and `SEP_ETD_F1` metrics for `lod`, `initial_system_prompt`, and `best_system_prompt`;
        4. annotates the `lod` and `initial_system_prompt` values directly with one symbol per cell, always compared to `best_system_prompt`, using a paired Wilcoxon test:
           - `▲` for better and significant,
           - `●` for no significant difference,
           - `▼` for worse and significant.
        """,
        **replacements,
    )

    imports = """
        import warnings
        from pathlib import Path

        import pandas as pd
        from IPython.display import Markdown, display

        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
    """

    paths = join_code_blocks(
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        render_template(
            """
            PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
            WORKSPACE_ROOT = PROJECT_ROOT.parent
            MODEL_NAME = "__LLM_METHOD__"
            SUMMARY_CSV = PROJECT_ROOT / "analyse_results" / "statistical_analysis" / "common" / "per_user_run_summary.csv"
            STATISTICAL_ANALYSIS_ROOT = PROJECT_ROOT / "analyse_results" / "statistical_analysis"
            LOD_ROOT = PROJECT_ROOT.parent / "datasets" / "lod_results" / "individual_metrics"

            print(f"PROJECT_ROOT: {display_workspace_relative(PROJECT_ROOT, WORKSPACE_ROOT)}")
            print(f"SUMMARY_CSV: {display_workspace_relative(SUMMARY_CSV, WORKSPACE_ROOT)}")
            print(
                "STATISTICAL_ANALYSIS_ROOT: "
                f"{display_workspace_relative(STATISTICAL_ANALYSIS_ROOT, WORKSPACE_ROOT)}"
            )
            print(f"LOD_ROOT: {display_workspace_relative(LOD_ROOT, WORKSPACE_ROOT)}")
            """,
            **replacements,
        ),
    )

    helpers = """
        OBJECTIVE_ORDER = ["sep", "etd", "sep_etd_f1"]
        ALGORITHM_ORDER = ["user_knn", "item_knn", "bprmf", "ncf"]
        TARGET_COLUMN_BY_OBJECTIVE = {
            "sep": "mean_sep_per_user",
            "etd": "mean_etd_per_user",
            "sep_etd_f1": "mean_sep_etd_f1_per_user",
        }
        DISPLAY_METRICS = [
            ("mean_sep_per_user", "sep_value", "SEP", "sep"),
            ("mean_etd_per_user", "etd_value", "ETD", "etd"),
            ("mean_sep_etd_f1_per_user", "sep_etd_f1_value", "SEP_ETD_F1", "sep_etd_f1"),
        ]
        ALPHA = 0.05
        SYMBOLS = {
            "gain": "▲",
            "tie": "●",
            "loss": "▼",
            "missing": "-",
        }
        ROW_INDEX = [
            ("lod", ""),
            ("llama", "initial_system_prompt"),
            ("llama", "best_system_prompt"),
        ]


        def ordered_algorithms(values: list[str]) -> list[str]:
            ordered = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in values]
            ordered.extend(sorted(algorithm for algorithm in values if algorithm not in ordered))
            return ordered


        def harmonic_mean(left: float, right: float) -> float:
            if pd.isna(left) or pd.isna(right) or (left + right) == 0:
                return 0.0
            return float((2.0 * left * right) / (left + right))


        def load_best_wide_df(statistical_analysis_root: Path, objective_metric: str) -> pd.DataFrame:
            path = (
                statistical_analysis_root
                / objective_metric
                / "results"
                / f"{objective_metric}_best_method_by_algorithm_per_user_wide.csv"
            )
            if not path.exists():
                return pd.DataFrame()

            df = pd.read_csv(path)
            score_columns = [
                "sep_lod",
                "etd_lod",
                "sep_etd_f1_lod",
                "sep_llama_without_optimization",
                "etd_llama_without_optimization",
                "sep_etd_f1_llama_without_optimization",
                "sep_llama_with_optimization",
                "etd_llama_with_optimization",
                "sep_etd_f1_llama_with_optimization",
            ]
            for column in score_columns:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            return df


        def build_best_wide_by_objective(statistical_analysis_root: Path) -> dict[str, pd.DataFrame]:
            return {
                objective_metric: load_best_wide_df(statistical_analysis_root, objective_metric)
                for objective_metric in OBJECTIVE_ORDER
            }


        def wilcoxon_symbol_from_series(
            best_series: pd.Series,
            reference_series: pd.Series,
            alpha: float = ALPHA,
        ) -> str:
            paired = pd.DataFrame(
                {
                    "best": pd.to_numeric(best_series, errors="coerce"),
                    "reference": pd.to_numeric(reference_series, errors="coerce"),
                }
            ).dropna()

            if paired.empty or wilcoxon is None:
                return SYMBOLS["missing"]

            try:
                result = wilcoxon(
                    paired["best"],
                    paired["reference"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                p_value = float(result.pvalue)
            except ValueError:
                return SYMBOLS["missing"]

            if pd.isna(p_value) or p_value >= alpha:
                return SYMBOLS["tie"]

            mean_best = float(paired["best"].mean())
            mean_reference = float(paired["reference"].mean())
            if mean_best > mean_reference:
                return SYMBOLS["gain"]
            if mean_best < mean_reference:
                return SYMBOLS["loss"]
            return SYMBOLS["tie"]


        def load_summary(summary_csv: Path) -> pd.DataFrame:
            if not summary_csv.exists():
                raise FileNotFoundError(f"Statistical summary not found at {summary_csv}")

            df = pd.read_csv(summary_csv)
            numeric_cols = [
                "mean_sep_per_user",
                "mean_etd_per_user",
                "mean_sep_etd_f1_per_user",
                "n_users",
            ]
            for column in numeric_cols:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            return df


        def load_lod_summary(lod_root: Path) -> pd.DataFrame:
            rows = []
            for csv_path in sorted(lod_root.glob("indiv_metrics_explanations_optimized_*_K=20_recs.csv")):
                algorithm = (
                    csv_path.name
                    .replace("indiv_metrics_explanations_optimized_", "")
                    .replace("_K=20_recs.csv", "")
                )
                frame = pd.read_csv(csv_path)
                frame["sep"] = pd.to_numeric(frame["sep"], errors="coerce")
                frame["etd"] = pd.to_numeric(frame["etd"], errors="coerce")
                frame["sep_etd_f1"] = frame.apply(
                    lambda row: harmonic_mean(row["sep"], row["etd"]),
                    axis=1,
                )
                rows.append(
                    {
                        "algorithm": algorithm,
                        "mean_sep_per_user": frame["sep"].mean(),
                        "mean_etd_per_user": frame["etd"].mean(),
                        "mean_sep_etd_f1_per_user": frame["sep_etd_f1"].mean(),
                        "n_users": int(frame["userId"].nunique()),
                        "source_group": "lod",
                        "source_label": "",
                        "representation_model": pd.NA,
                        "mmr_lambda_tag": pd.NA,
                        "run_label": f"lod/{algorithm}",
                    }
                )

            if not rows:
                return pd.DataFrame(
                    columns=[
                        "algorithm",
                        "mean_sep_per_user",
                        "mean_etd_per_user",
                        "mean_sep_etd_f1_per_user",
                        "n_users",
                        "source_group",
                        "source_label",
                        "representation_model",
                        "mmr_lambda_tag",
                        "run_label",
                    ]
                )

            return pd.DataFrame(rows)


        def format_repr_model(value: str | float | None) -> str:
            if value is None or pd.isna(value):
                return "-"
            return str(value).replace("repr_", "")


        def format_lambda(value: str | float | None) -> str:
            if value is None or pd.isna(value):
                return "-"
            return str(value).replace("mmr_lambda_", "").replace("_", ".")


        def pick_initial_rows(summary_df: pd.DataFrame, objective_metric: str) -> pd.DataFrame:
            frame = summary_df[
                (summary_df["run_type"] == "without_optimization")
                & (summary_df["metric"] == objective_metric)
            ].copy()
            frame["source_group"] = "llama"
            frame["source_label"] = "initial_system_prompt"
            return frame


        def pick_best_optimized_rows(summary_df: pd.DataFrame, objective_metric: str) -> pd.DataFrame:
            target_column = TARGET_COLUMN_BY_OBJECTIVE[objective_metric]
            frame = summary_df[
                (summary_df["run_type"] == "with_optimization")
                & (summary_df["metric"] == objective_metric)
            ].copy()

            if frame.empty:
                return frame

            frame = frame.sort_values(
                by=[
                    "algorithm",
                    target_column,
                    "mean_sep_etd_f1_per_user",
                    "mean_sep_per_user",
                    "mean_etd_per_user",
                    "run_label",
                ],
                ascending=[True, False, False, False, False, True],
                na_position="last",
            )
            frame = frame.drop_duplicates(subset=["algorithm"], keep="first")
            frame["source_group"] = "llama"
            frame["source_label"] = "best_system_prompt"
            return frame


        def build_objective_comparison_df(
            summary_df: pd.DataFrame,
            lod_df: pd.DataFrame,
            objective_metric: str,
        ) -> pd.DataFrame:
            frames = []

            initial_df = pick_initial_rows(summary_df, objective_metric)
            if not initial_df.empty:
                frames.append(initial_df)

            best_df = pick_best_optimized_rows(summary_df, objective_metric)
            if not best_df.empty:
                frames.append(best_df)

            if not lod_df.empty:
                lod_view = lod_df.copy()
                lod_view["metric"] = objective_metric
                frames.append(lod_view)

            if not frames:
                return pd.DataFrame()

            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined["objective_metric"] = objective_metric
            combined["algorithm"] = pd.Categorical(
                combined["algorithm"],
                categories=ordered_algorithms(combined["algorithm"].dropna().astype(str).unique().tolist()),
                ordered=True,
            )
            return combined.sort_values(by=["algorithm", "source_group", "source_label"]).reset_index(drop=True)


        def build_metric_pivot(
            comparison_df: pd.DataFrame,
            metric_column: str,
            top_label: str,
        ) -> pd.DataFrame:
            pivot = comparison_df.pivot_table(
                index=["source_group", "source_label"],
                columns="algorithm",
                values=metric_column,
                aggfunc="first",
                observed=False,
            )
            pivot = pivot.reindex(
                index=pd.MultiIndex.from_tuples(ROW_INDEX, names=[None, None])
            )
            algorithms = ordered_algorithms([str(column) for column in pivot.columns.tolist()])
            pivot = pivot.reindex(columns=algorithms)
            pivot.columns = pd.MultiIndex.from_product([[top_label], pivot.columns])
            return pivot


        def build_significance_symbol_map(
            best_wide_df: pd.DataFrame,
            metric_key: str,
        ) -> dict[str, dict[str, str]]:
            if best_wide_df.empty:
                return {}

            best_column = f"{metric_key}_llama_with_optimization"
            initial_column = f"{metric_key}_llama_without_optimization"
            lod_column = f"{metric_key}_lod"
            required_columns = {"algorithm", best_column, initial_column, lod_column}
            if not required_columns.issubset(best_wide_df.columns):
                return {}

            available_algorithms = ordered_algorithms(
                best_wide_df["algorithm"].dropna().astype(str).unique().tolist()
            )
            symbols_by_algorithm = {}

            for algorithm in available_algorithms:
                symbols_by_algorithm[algorithm] = {
                    "initial_vs_best": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, initial_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                    ),
                    "lod_vs_best": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, lod_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                    )
                }
            return symbols_by_algorithm


        def build_annotated_metric_table(
            metric_table: pd.DataFrame,
            best_wide_df: pd.DataFrame,
            metric_key: str,
            top_label: str,
        ) -> pd.DataFrame:
            if metric_table.empty:
                return pd.DataFrame()

            values_df = metric_table[top_label].copy()
            annotated_df = values_df.copy().astype(object)
            for row_index in annotated_df.index:
                for algorithm in annotated_df.columns:
                    value = values_df.loc[row_index, algorithm]
                    annotated_df.loc[row_index, algorithm] = "-" if pd.isna(value) else f"{float(value):.6f}"

            symbols_by_algorithm = build_significance_symbol_map(
                best_wide_df=best_wide_df,
                metric_key=metric_key,
            )

            row_symbol_key_map = {
                ("lod", ""): "lod_vs_best",
                ("llama", "initial_system_prompt"): "initial_vs_best",
            }

            for row_index, symbol_key in row_symbol_key_map.items():
                if row_index not in annotated_df.index:
                    continue

                for algorithm in annotated_df.columns:
                    base_value = annotated_df.loc[row_index, algorithm]
                    if base_value == "-":
                        continue

                    symbol_info = symbols_by_algorithm.get(str(algorithm), {})
                    symbol = symbol_info.get(symbol_key, SYMBOLS["missing"])
                    if symbol == SYMBOLS["missing"]:
                        continue

                    annotated_df.loc[row_index, algorithm] = f"{symbol} {base_value}"

            annotated_df.columns = pd.MultiIndex.from_product([[top_label], annotated_df.columns])
            return annotated_df


        def build_best_config_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
            best_df = comparison_df[comparison_df["source_label"] == "best_system_prompt"].copy()
            if best_df.empty:
                return pd.DataFrame()

            best_df["representation_model"] = best_df["representation_model"].apply(format_repr_model)
            best_df["mmr_lambda_quality"] = best_df["mmr_lambda_tag"].apply(format_lambda)

            columns = [
                "algorithm",
                "representation_model",
                "mmr_lambda_quality",
                "mean_sep_per_user",
                "mean_etd_per_user",
                "mean_sep_etd_f1_per_user",
                "run_label",
            ]
            return best_df[columns].rename(
                columns={
                    "algorithm": "algorithm",
                    "representation_model": "representation_model",
                    "mmr_lambda_quality": "mmr_lambda_quality",
                    "mean_sep_per_user": "sep_value",
                    "mean_etd_per_user": "etd_value",
                    "mean_sep_etd_f1_per_user": "sep_etd_f1_value",
                    "run_label": "optimized_run_label",
                }
            ).reset_index(drop=True)
    """

    outputs = """
        summary_df = load_summary(SUMMARY_CSV)
        lod_df = load_lod_summary(LOD_ROOT)
        best_wide_by_objective = build_best_wide_by_objective(STATISTICAL_ANALYSIS_ROOT)

        if summary_df.empty:
            warnings.warn("The per-user summary is empty. No table was built.")
        else:
            display(Markdown(
                "The tables below use, for each objective, the best optimized process by algorithm "
                "according to that objective's own target metric."
            ))
            display(Markdown(
                "The symbols appear next to the `lod` and `initial_system_prompt` values, always compared to `best_system_prompt`. "
                "Wilcoxon legend: `▲` = better and significant, "
                "`●` = no significant difference and `▼` = worse and significant."
            ))

            for objective_metric in OBJECTIVE_ORDER:
                comparison_df = build_objective_comparison_df(
                    summary_df=summary_df,
                    lod_df=lod_df,
                    objective_metric=objective_metric,
                )

                if comparison_df.empty:
                    display(Markdown(f"## Objective `{objective_metric}`"))
                    display(Markdown("> No data found for this objective."))
                    continue

                display(Markdown(f"## Objective `{objective_metric}`"))
                display(Markdown("### Best optimized configuration by algorithm"))
                best_config_df = build_best_config_table(comparison_df)
                best_wide_df = best_wide_by_objective.get(objective_metric, pd.DataFrame())
                if best_config_df.empty:
                    display(Markdown("> No optimized configuration was found."))
                else:
                    display(best_config_df.style.format(
                        {
                            "sep_value": "{:.6f}",
                            "etd_value": "{:.6f}",
                            "sep_etd_f1_value": "{:.6f}",
                        },
                        na_rep="-",
                    ))

                for metric_column, top_label, metric_label, metric_key in DISPLAY_METRICS:
                    display(Markdown(
                        f"### Table of `{metric_label}` for processes optimized for `{objective_metric}`"
                    ))
                    metric_table = build_metric_pivot(
                        comparison_df=comparison_df,
                        metric_column=metric_column,
                        top_label=top_label,
                    )
                    annotated_metric_table = build_annotated_metric_table(
                        metric_table=metric_table,
                        best_wide_df=best_wide_df,
                        metric_key=metric_key,
                        top_label=top_label,
                    )
                    display(annotated_metric_table)
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("paths", paths),
            code_cell("helpers", helpers),
            code_cell("outputs", outputs),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_global_best_process_curves_notebook(llm_method: str) -> dict:
    replacements = {"LLM_METHOD": llm_method}

    intro = render_template(
        """
        # Curves of the best processes by objective

        This notebook retrieves exactly the `best_system_prompt` values used in the global objective tables
        and shows, for each of them, the progression of the metrics during the optimization process.

        A seleção segue o mesmo critério do notebook `plot_optimization_metric_tables_by_objective.ipynb`:
        - for each objective (`sep`, `etd`, `sep_etd_f1`);
        - and for each algorithm;
        - the `with_optimization` configuration with the highest average per-user value on the target metric itself is selected.

        For each selected process, the notebook displays:
        1. a summary of the winning configuration;
        2. a plot focused on the target metric;
        3. a plot with the target metric and the other main metrics recorded by epoch (`SEP`, `ETD`, and `SEP_ETD_F1`).
        """,
        **replacements,
    )

    imports = join_code_blocks(
        """
        import os
        import sys
        import warnings
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import Markdown, display
        """,
        NOTEBOOK_PROJECT_ROOT_HELPERS,
        """
        PROJECT_ROOT = find_local_project_root(Path.cwd().resolve())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.utils.optimization_process_analysis import load_process_bundle
        """,
        NOTEBOOK_METRIC_COLOR_HELPERS,
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.max_columns", 50)
        """,
    )

    paths = render_template(
        """
        MODEL_NAME = "__LLM_METHOD__"
        WORKSPACE_ROOT = PROJECT_ROOT.parent
        SUMMARY_CSV = PROJECT_ROOT / "analyse_results" / "statistical_analysis" / "common" / "per_user_run_summary.csv"

        print(f"PROJECT_ROOT: {display_workspace_relative(PROJECT_ROOT, WORKSPACE_ROOT)}")
        print(f"SUMMARY_CSV: {display_workspace_relative(SUMMARY_CSV, WORKSPACE_ROOT)}")
        """,
        **replacements,
    )

    helpers = """
        OBJECTIVE_ORDER = ["sep", "etd", "sep_etd_f1"]
        ALGORITHM_ORDER = ["user_knn", "item_knn", "bprmf", "ncf"]
        TARGET_COLUMN_BY_OBJECTIVE = {
            "sep": "mean_sep_per_user",
            "etd": "mean_etd_per_user",
            "sep_etd_f1": "mean_sep_etd_f1_per_user",
        }


        def ordered_algorithms(values: list[str]) -> list[str]:
            ordered = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in values]
            ordered.extend(sorted(algorithm for algorithm in values if algorithm not in ordered))
            return ordered


        def load_summary(summary_csv: Path) -> pd.DataFrame:
            if not summary_csv.exists():
                raise FileNotFoundError(f"Statistical summary not found at {summary_csv}")

            df = pd.read_csv(summary_csv)
            numeric_cols = [
                "mean_sep_per_user",
                "mean_etd_per_user",
                "mean_sep_etd_f1_per_user",
                "n_users",
            ]
            for column in numeric_cols:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            return df


        def format_repr_model(value: str | float | None) -> str:
            if value is None or pd.isna(value):
                return "-"
            return str(value).replace("repr_", "")


        def format_lambda(value: str | float | None) -> str:
            if value is None or pd.isna(value):
                return "-"
            return str(value).replace("mmr_lambda_", "").replace("_", ".")


        def build_process_dir_rel(row: pd.Series) -> str:
            return (
                "out/prompt_optimization/"
                + str(row["model"])
                + "/"
                + str(row["algorithm"])
                + "/"
                + str(row["metric"])
                + "/"
                + str(row["representation_model"])
                + "/"
                + str(row["early_stopping_tag"])
                + "/"
                + str(row["mmr_lambda_tag"])
                + "/"
                + str(row["mmr_pool_tag"])
                + "/"
                + str(row["model"])
                + "/prompt_opt/"
                + str(row["metric"])
            )


        def pick_best_optimized_rows(summary_df: pd.DataFrame, objective_metric: str) -> pd.DataFrame:
            target_column = TARGET_COLUMN_BY_OBJECTIVE[objective_metric]
            frame = summary_df[
                (summary_df["run_type"] == "with_optimization")
                & (summary_df["metric"] == objective_metric)
            ].copy()

            if frame.empty:
                return frame

            frame = frame.sort_values(
                by=[
                    "algorithm",
                    target_column,
                    "mean_sep_etd_f1_per_user",
                    "mean_sep_per_user",
                    "mean_etd_per_user",
                    "run_label",
                ],
                ascending=[True, False, False, False, False, True],
                na_position="last",
            )
            frame = frame.drop_duplicates(subset=["algorithm"], keep="first").reset_index(drop=True)
            frame["process_dir_rel"] = frame.apply(build_process_dir_rel, axis=1)
            frame["representation_model_label"] = frame["representation_model"].apply(format_repr_model)
            frame["mmr_lambda_quality"] = frame["mmr_lambda_tag"].apply(format_lambda)
            frame["algorithm"] = pd.Categorical(
                frame["algorithm"],
                categories=ordered_algorithms(frame["algorithm"].dropna().astype(str).unique().tolist()),
                ordered=True,
            )
            return frame.sort_values("algorithm").reset_index(drop=True)


        def draw_process_charts(bundle: dict, title_prefix: str = "") -> None:
            plt.close("all")
            epochs_df = ensure_balance_metrics(bundle["epochs_df"].copy())
            objective_metric = bundle["summary"]["objective_metric"]

            if epochs_df.empty:
                display(Markdown("> No epoch was recorded for this process."))
                return

            available_primary_metrics = [
                metric_name
                for metric_name in ["sep", "etd", "sep_etd_f1"]
                if (
                    f"train_score_{metric_name}" in epochs_df.columns
                    or f"val_score_{metric_name}" in epochs_df.columns
                )
            ]
            all_metric_names = []
            for metric_name in [objective_metric, *available_primary_metrics]:
                if metric_name not in all_metric_names:
                    all_metric_names.append(metric_name)

            metric_colors = {
                metric_name: metric_color(metric_name)
                for metric_name in all_metric_names
            }
            epoch_positions = epochs_df["epoch"].tolist()
            epoch_labels = [epoch + 1 for epoch in epoch_positions]
            best_train_epochs = epochs_df.loc[epochs_df["is_best_train_epoch"], "epoch"].tolist()
            best_val_epochs = epochs_df.loc[epochs_df["is_best_val_epoch"], "epoch"].tolist()
            shared_best_epochs = sorted(set(best_train_epochs) & set(best_val_epochs))
            train_only_epochs = [epoch for epoch in best_train_epochs if epoch not in shared_best_epochs]
            val_only_epochs = [epoch for epoch in best_val_epochs if epoch not in shared_best_epochs]

            display(Markdown(
                "Vertical lines: purple = best epoch only in training; pink = best epoch only in validation; "
                "gray = the same epoch was best in both training and validation."
            ))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(
                epochs_df["epoch"],
                epochs_df["train_metric"],
                marker="o",
                linewidth=2,
                linestyle="-",
                color=metric_colors[objective_metric],
                label=f"train::{objective_metric}",
            )
            if epochs_df["val_metric"].notna().any():
                ax.plot(
                    epochs_df["epoch"],
                    epochs_df["val_metric"],
                    marker="o",
                    linewidth=2,
                    linestyle="--",
                    color=metric_colors[objective_metric],
                    label=f"val::{objective_metric}",
                )

            for epoch in train_only_epochs:
                ax.axvline(epoch, color="#7b2cbf", linestyle=":", linewidth=2, alpha=0.9)
            for epoch in val_only_epochs:
                ax.axvline(epoch, color="#e75480", linestyle=":", linewidth=2, alpha=0.9)
            for epoch in shared_best_epochs:
                ax.axvline(epoch, color="#6c757d", linestyle="-", linewidth=2.2, alpha=0.95)

            chart_title = f"{title_prefix} - target metric ({objective_metric})" if title_prefix else f"Objective metric ({objective_metric})"
            ax.set_title(chart_title)
            ax.set_xlabel("epoch")
            ax.set_ylabel("score")
            ax.set_xticks(epoch_positions)
            ax.set_xticklabels(epoch_labels)
            ax.legend(loc="best")
            plt.tight_layout()
            plt.show()

            if all_metric_names:
                fig, ax = plt.subplots(figsize=(14, 6))
                for metric_name in all_metric_names:
                    train_col = f"train_score_{metric_name}"
                    val_col = f"val_score_{metric_name}"
                    line_color = metric_colors[metric_name]

                    if train_col in epochs_df.columns and epochs_df[train_col].notna().any():
                        ax.plot(
                            epochs_df["epoch"],
                            epochs_df[train_col],
                            marker="o",
                            linewidth=1.8,
                            linestyle="-",
                            color=line_color,
                            label=f"train::{metric_name}",
                        )
                    if val_col in epochs_df.columns and epochs_df[val_col].notna().any():
                        ax.plot(
                            epochs_df["epoch"],
                            epochs_df[val_col],
                            marker="o",
                            linewidth=1.8,
                            linestyle="--",
                            color=line_color,
                            label=f"val::{metric_name}",
                        )

                chart_title = f"{title_prefix} - all metrics" if title_prefix else "All metrics"
                ax.set_title(chart_title)
                ax.set_xlabel("epoch")
                ax.set_ylabel("score")
                ax.set_xticks(epoch_positions)
                ax.set_xticklabels(epoch_labels)
                ax.legend(loc="best", ncol=2)
                plt.tight_layout()
                plt.show()
            else:
                display(Markdown("No detailed metrics were found for this process."))
            plt.close("all")
    """

    outputs = """
        summary_df = load_summary(SUMMARY_CSV)

        if summary_df.empty:
            warnings.warn("The per-user summary is empty. No plot was generated.")
        else:
            display(Markdown(
                "This notebook uses the same `best_system_prompt` values selected in the global objective tables."
            ))

            for objective_metric in OBJECTIVE_ORDER:
                best_df = pick_best_optimized_rows(summary_df, objective_metric)

                if best_df.empty:
                    display(Markdown(f"## Objective `{objective_metric}`"))
                    display(Markdown("> No optimized process was found for this objective."))
                    continue

                display(Markdown(f"## Objective `{objective_metric}`"))
                display(Markdown("### Selected configurations"))
                display(
                    best_df[
                        [
                            "algorithm",
                            "representation_model_label",
                            "mmr_lambda_quality",
                            "mean_sep_per_user",
                            "mean_etd_per_user",
                            "mean_sep_etd_f1_per_user",
                            "process_dir_rel",
                        ]
                    ].rename(
                        columns={
                            "representation_model_label": "representation_model",
                            "mean_sep_per_user": "sep_value",
                            "mean_etd_per_user": "etd_value",
                            "mean_sep_etd_f1_per_user": "sep_etd_f1_value",
                        }
                    ).style.format(
                        {
                            "sep_value": "{:.6f}",
                            "etd_value": "{:.6f}",
                            "sep_etd_f1_value": "{:.6f}",
                        },
                        na_rep="-",
                    )
                )

                for row in best_df.itertuples(index=False):
                    process_path = PROJECT_ROOT / row.process_dir_rel
                    display(Markdown(
                        "### "
                        + str(row.algorithm)
                        + " | repr="
                        + str(row.representation_model_label)
                        + " | lambda="
                        + str(row.mmr_lambda_quality)
                    ))

                    if not process_path.exists():
                        display(Markdown(f"> Process not found at `{row.process_dir_rel}`."))
                        continue

                    bundle = load_process_bundle(process_path, PROJECT_ROOT)
                    display(
                        pd.DataFrame(
                            [
                                {
                                    "process_dir_rel": row.process_dir_rel,
                                    "objective_metric": objective_metric,
                                    "algorithm": row.algorithm,
                                    "representation_model": row.representation_model_label,
                                    "mmr_lambda_quality": row.mmr_lambda_quality,
                                    "best_train_epoch": bundle["summary"]["best_train_epoch"],
                                    "best_train_metric": bundle["summary"]["best_train_metric"],
                                    "best_val_epoch": bundle["summary"]["best_val_epoch"],
                                    "best_val_metric": bundle["summary"]["best_val_metric"],
                                    "saved_best_origin": bundle["summary"]["saved_best_origin"],
                                }
                            ]
                        )
                    )

                    draw_process_charts(
                        bundle=bundle,
                        title_prefix=f"{objective_metric} / {row.algorithm}",
                    )
    """

    return {
        "cells": [
            markdown_cell("overview", intro),
            code_cell("imports", imports),
            code_cell("paths", paths),
            code_cell("helpers", helpers),
            code_cell("outputs", outputs),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def build_global_objective_comparison_best_prompt_symbols_notebook(llm_method: str) -> dict:
    notebook = build_global_objective_comparison_notebook(llm_method=llm_method)

    def replace_first_available(
        text: str,
        replacements: list[tuple[str, str]],
        label: str,
    ) -> str:
        for old, new in replacements:
            if old in text:
                return text.replace(old, new, 1)
        raise ValueError(f"Could not locate the expected block for {label}.")

    overview = "".join(notebook["cells"][0]["source"])
    overview = replace_first_available(
        overview,
        [
            (
                "4. annotates the `lod` and `initial_system_prompt` values directly with one symbol per cell, "
                "always compared to `best_system_prompt`, using a paired Wilcoxon test:\n"
                "   - `▲` for better and significant,\n"
                "   - `●` for no significant difference,\n"
                "   - `▼` for worse and significant.",
                "4. annotates the `best_system_prompt` values directly with two symbols per cell, "
                "in the order `vs initial` and then `vs lod`, using a paired Wilcoxon test:\n"
                "   - `▲` for better and significant,\n"
                "   - `●` for no significant difference,\n"
                "   - `▼` for worse and significant.",
            ),
            (
                "4. anota diretamente os valores de `lod` e `initial_system_prompt` com um símbolo por célula, "
                "sempre em comparação com `best_system_prompt`, obtido por teste de Wilcoxon pareado:\n"
                "   - `▲` for better and significant,\n"
                "   - `●` for no significant difference,\n"
                "   - `▼` for worse and significant.",
                "4. annotates the `best_system_prompt` values directly with two symbols per cell, "
                "in the order `vs initial` and then `vs lod`, using a paired Wilcoxon test:\n"
                "   - `▲` for better and significant,\n"
                "   - `●` for no significant difference,\n"
                "   - `▼` for worse and significant.",
            ),
        ],
        "overview-best-prompt",
    )
    notebook["cells"][0]["source"] = _to_source_lines(overview)

    helpers = "".join(notebook["cells"][3]["source"])
    old_significance_block = textwrap.dedent(
        """
        def build_significance_symbol_map(
            best_wide_df: pd.DataFrame,
            metric_key: str,
        ) -> dict[str, dict[str, str]]:
            if best_wide_df.empty:
                return {}

            best_column = f"{metric_key}_llama_with_optimization"
            initial_column = f"{metric_key}_llama_without_optimization"
            lod_column = f"{metric_key}_lod"
            required_columns = {"algorithm", best_column, initial_column, lod_column}
            if not required_columns.issubset(best_wide_df.columns):
                return {}

            available_algorithms = ordered_algorithms(
                best_wide_df["algorithm"].dropna().astype(str).unique().tolist()
            )
            symbols_by_algorithm = {}

            for algorithm in available_algorithms:
                symbols_by_algorithm[algorithm] = {
                    "initial_vs_best": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, initial_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                    ),
                    "lod_vs_best": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, lod_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                    )
                }
            return symbols_by_algorithm
        """
    ).strip("\n")
    new_significance_block = textwrap.dedent(
        """
        def build_significance_symbol_map(
            best_wide_df: pd.DataFrame,
            metric_key: str,
        ) -> dict[str, dict[str, str]]:
            if best_wide_df.empty:
                return {}

            best_column = f"{metric_key}_llama_with_optimization"
            initial_column = f"{metric_key}_llama_without_optimization"
            lod_column = f"{metric_key}_lod"
            required_columns = {"algorithm", best_column, initial_column, lod_column}
            if not required_columns.issubset(best_wide_df.columns):
                return {}

            available_algorithms = ordered_algorithms(
                best_wide_df["algorithm"].dropna().astype(str).unique().tolist()
            )
            symbols_by_algorithm = {}

            for algorithm in available_algorithms:
                symbols_by_algorithm[algorithm] = {
                    "best_vs_initial": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, initial_column],
                    ),
                    "best_vs_lod": wilcoxon_symbol_from_series(
                        best_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, best_column],
                        reference_series=best_wide_df.loc[best_wide_df["algorithm"] == algorithm, lod_column],
                    )
                }
            return symbols_by_algorithm
        """
    ).strip("\n")
    helpers = replace_first_available(
        helpers,
        [(old_significance_block, new_significance_block)],
        "significance-map-best-prompt",
    )

    old_annotated_block = textwrap.dedent(
        """
        def build_annotated_metric_table(
            metric_table: pd.DataFrame,
            best_wide_df: pd.DataFrame,
            metric_key: str,
            top_label: str,
        ) -> pd.DataFrame:
            if metric_table.empty:
                return pd.DataFrame()

            values_df = metric_table[top_label].copy()
            annotated_df = values_df.copy().astype(object)
            for row_index in annotated_df.index:
                for algorithm in annotated_df.columns:
                    value = values_df.loc[row_index, algorithm]
                    annotated_df.loc[row_index, algorithm] = "-" if pd.isna(value) else f"{float(value):.6f}"

            symbols_by_algorithm = build_significance_symbol_map(
                best_wide_df=best_wide_df,
                metric_key=metric_key,
            )

            row_symbol_key_map = {
                ("lod", ""): "lod_vs_best",
                ("llama", "initial_system_prompt"): "initial_vs_best",
            }

            for row_index, symbol_key in row_symbol_key_map.items():
                if row_index not in annotated_df.index:
                    continue

                for algorithm in annotated_df.columns:
                    base_value = annotated_df.loc[row_index, algorithm]
                    if base_value == "-":
                        continue

                    symbol_info = symbols_by_algorithm.get(str(algorithm), {})
                    symbol = symbol_info.get(symbol_key, SYMBOLS["missing"])
                    if symbol == SYMBOLS["missing"]:
                        continue

                    annotated_df.loc[row_index, algorithm] = f"{symbol} {base_value}"

            annotated_df.columns = pd.MultiIndex.from_product([[top_label], annotated_df.columns])
            return annotated_df
        """
    ).strip("\n")
    new_annotated_block = textwrap.dedent(
        """
        def build_annotated_metric_table(
            metric_table: pd.DataFrame,
            best_wide_df: pd.DataFrame,
            metric_key: str,
            top_label: str,
        ) -> pd.DataFrame:
            if metric_table.empty:
                return pd.DataFrame()

            values_df = metric_table[top_label].copy()
            annotated_df = values_df.copy().astype(object)
            for row_index in annotated_df.index:
                for algorithm in annotated_df.columns:
                    value = values_df.loc[row_index, algorithm]
                    annotated_df.loc[row_index, algorithm] = "-" if pd.isna(value) else f"{float(value):.6f}"

            symbols_by_algorithm = build_significance_symbol_map(
                best_wide_df=best_wide_df,
                metric_key=metric_key,
            )

            best_index = ("llama", "best_system_prompt")
            if best_index in annotated_df.index:
                for algorithm in annotated_df.columns:
                    base_value = annotated_df.loc[best_index, algorithm]
                    if base_value == "-":
                        continue

                    symbol_info = symbols_by_algorithm.get(
                        str(algorithm),
                        {"best_vs_initial": SYMBOLS["missing"], "best_vs_lod": SYMBOLS["missing"]},
                    )
                    best_vs_initial = symbol_info["best_vs_initial"]
                    best_vs_lod = symbol_info["best_vs_lod"]
                    if best_vs_initial == SYMBOLS["missing"] and best_vs_lod == SYMBOLS["missing"]:
                        continue

                    annotated_df.loc[best_index, algorithm] = (
                        f"{best_vs_initial}{best_vs_lod} {base_value}"
                    )

            annotated_df.columns = pd.MultiIndex.from_product([[top_label], annotated_df.columns])
            return annotated_df
        """
    ).strip("\n")
    helpers = replace_first_available(
        helpers,
        [(old_annotated_block, new_annotated_block)],
        "annotated-table-best-prompt",
    )
    notebook["cells"][3]["source"] = _to_source_lines(helpers)

    outputs = "".join(notebook["cells"][4]["source"])
    outputs = replace_first_available(
        outputs,
        [
            (
                "The symbols appear next to the `lod` and `initial_system_prompt` values, always compared to `best_system_prompt`. ",
                "The symbols appear on the `best_system_prompt` row, in the order `vs initial` and then `vs lod`. ",
            ),
            (
                "Os símbolos aparecem na linha `best_system_prompt`, na ordem `vs initial` e depois `vs lod`. ",
                "The symbols appear on the `best_system_prompt` row, in the order `vs initial` and then `vs lod`. ",
            ),
        ],
        "outputs-best-prompt-legend",
    )
    notebook["cells"][4]["source"] = _to_source_lines(outputs)
    return notebook

def write_notebook(project_root: Path, llm_method: str, algorithm: str) -> Path:
    search_root_rel = f"out/prompt_optimization/{llm_method}/{algorithm}"
    notebook = build_notebook(algorithm=algorithm, search_root_rel=search_root_rel)
    output_dir = project_root / search_root_rel / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"plot_optimization_metrics_{algorithm}.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def discover_metrics(prompt_opt_root: Path, algorithm: str) -> list[str]:
    algorithm_root = prompt_opt_root / algorithm
    if not algorithm_root.exists():
        return []

    return sorted(
        path.name
        for path in algorithm_root.iterdir()
        if path.is_dir()
        and path.name != "analysis"
        and any(path.rglob("optimization_process_metadata.json"))
    )

def write_metric_notebook(project_root: Path, llm_method: str, algorithm: str, objective_metric: str) -> Path:
    search_root_rel = f"out/prompt_optimization/{llm_method}/{algorithm}/{objective_metric}"
    notebook = build_metric_notebook(
        algorithm=algorithm,
        llm_method=llm_method,
        objective_metric=objective_metric,
        search_root_rel=search_root_rel,
    )
    output_dir = project_root / search_root_rel
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"plot_optimization_metrics_{algorithm}_{objective_metric}.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def discover_process_dirs(prompt_opt_root: Path, algorithm: str | None = None) -> list[Path]:
    base_root = prompt_opt_root / algorithm if algorithm else prompt_opt_root
    if not base_root.exists():
        return []

    return sorted(metadata_path.parent for metadata_path in base_root.rglob("optimization_process_metadata.json"))

def write_process_notebook(project_root: Path, process_dir: Path) -> Path:
    process_dir = process_dir.resolve()
    process_dir_rel = str(process_dir.relative_to(project_root))
    notebook = build_process_notebook(process_dir_rel=process_dir_rel)
    output_path = process_dir / "plot_optimization_process.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def write_global_test_results_notebook(project_root: Path, llm_method: str) -> Path:
    notebook = build_global_test_results_notebook(llm_method=llm_method)
    output_dir = project_root / "analyse_results" / "analysis_global"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plot_optimization_metrics.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def write_global_objective_comparison_notebook(project_root: Path, llm_method: str) -> Path:
    notebook = build_global_objective_comparison_best_prompt_symbols_notebook(llm_method=llm_method)
    output_dir = project_root / "analyse_results" / "analysis_global"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plot_optimization_metric_tables_by_objective.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def write_global_best_process_curves_notebook(project_root: Path, llm_method: str) -> Path:
    notebook = build_global_best_process_curves_notebook(llm_method=llm_method)
    output_dir = project_root / "analyse_results" / "analysis_global"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plot_best_optimization_processes_by_objective.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path

def generate(project_root: Path, llm_method: str, algorithms: Iterable[str] | None = None) -> list[Path]:
    prompt_opt_root = project_root / "out" / "prompt_optimization" / llm_method
    discovered = discover_algorithms(prompt_opt_root)
    selected = list(algorithms) if algorithms else discovered
    written_paths = []
    for algorithm in selected:
        algorithm_root = prompt_opt_root / algorithm
        if not algorithm_root.exists():
            continue
        if not any(algorithm_root.rglob("optimization_process_metadata.json")):
            continue
        written_paths.append(write_notebook(project_root, llm_method, algorithm))
    return written_paths

def generate_metric_notebooks(
    project_root: Path,
    llm_method: str,
    algorithms: Iterable[str] | None = None,
) -> list[Path]:
    prompt_opt_root = project_root / "out" / "prompt_optimization" / llm_method
    discovered = discover_algorithms(prompt_opt_root)
    selected = list(algorithms) if algorithms else discovered
    written_paths = []

    for algorithm in selected:
        for objective_metric in discover_metrics(prompt_opt_root, algorithm):
            written_paths.append(
                write_metric_notebook(
                    project_root=project_root,
                    llm_method=llm_method,
                    algorithm=algorithm,
                    objective_metric=objective_metric,
                )
            )

    return written_paths

def generate_process_notebooks(
    project_root: Path,
    llm_method: str,
    algorithms: Iterable[str] | None = None,
) -> list[Path]:
    prompt_opt_root = project_root / "out" / "prompt_optimization" / llm_method
    discovered = discover_algorithms(prompt_opt_root)
    selected = list(algorithms) if algorithms else discovered
    written_paths = []

    for algorithm in selected:
        for process_dir in discover_process_dirs(prompt_opt_root, algorithm):
            written_paths.append(write_process_notebook(project_root=project_root, process_dir=process_dir))

    return written_paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate aggregate optimization analysis notebooks.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the explainability-with-LLMs project root.",
    )
    parser.add_argument(
        "--llm-method",
        default="Llama3.1-I",
        help="LLM method subdirectory inside out/prompt_optimization.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        help="Optional algorithm(s) to generate explicitly. Repeat the flag to pass more than one.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    written_paths = []
    written_paths.append(
        write_global_test_results_notebook(
            project_root=project_root,
            llm_method=args.llm_method,
        )
    )
    written_paths.append(
        write_global_objective_comparison_notebook(
            project_root=project_root,
            llm_method=args.llm_method,
        )
    )
    written_paths.append(
        write_global_best_process_curves_notebook(
            project_root=project_root,
            llm_method=args.llm_method,
        )
    )
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
