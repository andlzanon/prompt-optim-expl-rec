from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "run_prompt_optimizer.py").exists() and (candidate / "out").exists():
            return candidate
    raise FileNotFoundError(
        "Nao foi possivel localizar a raiz de explainability-with-LLMs a partir do caminho informado."
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _shorten(text: Optional[str], width: int = 120) -> Optional[str]:
    if not text:
        return text
    normalized = " ".join(str(text).split())
    return textwrap.shorten(normalized, width=width, placeholder="...")


def _normalize_process_dir(process_path: Path) -> Path:
    if process_path.is_file():
        if process_path.name == "optimization_process_metadata.json":
            return process_path.parent
        if process_path.name == "best_prompt.json":
            return process_path.parent
        raise FileNotFoundError(
            f"O arquivo {process_path} nao corresponde a um artefato conhecido do processo."
        )

    metadata_path = process_path / "optimization_process_metadata.json"
    if metadata_path.exists():
        return process_path

    raise FileNotFoundError(
        "Informe o diretorio do processo ou o caminho para optimization_process_metadata.json."
    )


def _resolve_path(project_root: Path, raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _saved_best_origin(metadata: Dict[str, Any], best_prompt_payload: Dict[str, Any]) -> str:
    saved_best_prompt = best_prompt_payload.get("best_prompt")
    best_train_prompt = metadata.get("best_on_train", {}).get("best_train_prompt")
    best_val_prompt = metadata.get("best_on_validation", {}).get("best_val_prompt")

    if saved_best_prompt and saved_best_prompt == best_train_prompt == best_val_prompt:
        return "train_and_validation"
    if saved_best_prompt and saved_best_prompt == best_train_prompt:
        return "train"
    if saved_best_prompt and saved_best_prompt == best_val_prompt:
        return "validation"
    return "unknown"


def _flatten_metric_scores(
    row: Dict[str, Any],
    scores: Optional[Dict[str, Any]],
    prefix: str,
) -> None:
    if not scores:
        return
    for metric_name, metric_value in scores.items():
        row[f"{prefix}_{metric_name}"] = metric_value


def discover_optimization_processes(
    project_root: Path,
    search_root: Optional[Path] = None,
) -> pd.DataFrame:
    process_rows: List[Dict[str, Any]] = []
    base_dir = (search_root or (project_root / "out" / "prompt_optimization")).resolve()
    metadata_paths = sorted(base_dir.rglob("optimization_process_metadata.json"))

    for metadata_path in metadata_paths:
        process_dir = metadata_path.parent
        metadata = _load_json(metadata_path)
        best_prompt_path = process_dir / "best_prompt.json"
        best_prompt_payload = _load_json(best_prompt_path) if best_prompt_path.exists() else {}
        settings = metadata.get("settings", {})
        args = metadata.get("args", {})
        run_summary = metadata.get("run_summary", {})

        row = {
            "process_dir": str(process_dir),
            "process_dir_rel": str(process_dir.relative_to(project_root)),
            "metadata_path": str(metadata_path),
            "metadata_path_rel": str(metadata_path.relative_to(project_root)),
            "best_prompt_path": str(best_prompt_path) if best_prompt_path.exists() else None,
            "best_prompt_path_rel": (
                str(best_prompt_path.relative_to(project_root)) if best_prompt_path.exists() else None
            ),
            "algorithm": args.get("algorithm"),
            "objective_metric": settings.get("objective_metric", metadata.get("metric")),
            "metric_name": metadata.get("metric_name"),
            "llm_method": args.get("llm_method"),
            "representation_model": settings.get("representation_model"),
            "early_stopping": settings.get("early_stopping"),
            "mmr_lambda_quality": settings.get("mmr_lambda_quality"),
            "mmr_pool_multiplier": settings.get("mmr_pool_multiplier"),
            "epochs_completed": run_summary.get("epochs_completed"),
            "time_prompt_optimization": metadata.get("time_prompt_optimization"),
            "best_train_epoch": metadata.get("best_on_train", {}).get("best_train_epoch"),
            "best_train_metric": metadata.get("best_on_train", {}).get("best_train_metric"),
            "best_val_epoch": metadata.get("best_on_validation", {}).get("best_val_epoch"),
            "best_val_metric": metadata.get("best_on_validation", {}).get("best_val_metric"),
            "saved_best_origin": _saved_best_origin(metadata, best_prompt_payload),
            "saved_best_prompt_preview": _shorten(best_prompt_payload.get("best_prompt")),
        }
        process_rows.append(row)

    catalog = pd.DataFrame(process_rows)
    if catalog.empty:
        return catalog

    return catalog.sort_values(
        by=[
            "algorithm",
            "objective_metric",
            "representation_model",
            "early_stopping",
            "mmr_lambda_quality",
            "mmr_pool_multiplier",
        ]
    ).reset_index(drop=True)


def build_epochs_dataframe(
    metadata: Dict[str, Any],
    project_root: Path,
    saved_best_prompt: Optional[str] = None,
) -> pd.DataFrame:
    epoch_rows: List[Dict[str, Any]] = []
    best_train_epoch = metadata.get("best_on_train", {}).get("best_train_epoch")
    best_val_epoch = metadata.get("best_on_validation", {}).get("best_val_epoch")

    for epoch_payload in metadata.get("epochs_history", []):
        row: Dict[str, Any] = {
            "epoch": epoch_payload.get("epoch"),
            "generated_new_prompt": epoch_payload.get("generated_new_prompt"),
            "time_spent_instruction": epoch_payload.get("time_spent_instruction"),
            "train_metric": epoch_payload.get("train_metric"),
            "time_spent_train_eval": epoch_payload.get("time_spent_train_eval"),
            "train_rows": epoch_payload.get("train_rows"),
            "train_valid_rate": epoch_payload.get("train_valid_rate"),
            "val_ran_this_epoch": epoch_payload.get("val_ran_this_epoch"),
            "val_metric": epoch_payload.get("val_metric"),
            "time_spent_val_eval": epoch_payload.get("time_spent_val_eval"),
            "prev_val_metric": epoch_payload.get("prev_val_metric"),
            "val_improvement_vs_prev": epoch_payload.get("val_improvement_vs_prev"),
            "no_improve_streak": epoch_payload.get("no_improve_streak"),
            "early_stopping_triggered_here": epoch_payload.get("early_stopping_triggered_here"),
            "prompt_this_epoch": epoch_payload.get("prompt_this_epoch"),
            "prompt_preview": _shorten(epoch_payload.get("prompt_this_epoch"), width=110),
            "meta_prompt_preview": _shorten(epoch_payload.get("meta_prompt_used"), width=110),
            "mmr_selected_reference_epochs": ",".join(
                str(item.get("epoch"))
                for item in (epoch_payload.get("mmr_selected_reference_epochs") or [])
            ),
            "is_best_train_epoch": epoch_payload.get("epoch") == best_train_epoch,
            "is_best_val_epoch": epoch_payload.get("epoch") == best_val_epoch,
            "is_saved_best_epoch": saved_best_prompt is not None
            and epoch_payload.get("prompt_this_epoch") == saved_best_prompt,
        }

        artifacts = epoch_payload.get("artifacts", {})
        row["epoch_json_path"] = str(
            _resolve_path(project_root, artifacts.get("epoch_json_path"))
        ) if artifacts.get("epoch_json_path") else None
        row["train_csv_path"] = str(
            _resolve_path(project_root, artifacts.get("train_csv_path"))
        ) if artifacts.get("train_csv_path") else None
        row["val_csv_path"] = str(
            _resolve_path(project_root, artifacts.get("val_csv_path"))
        ) if artifacts.get("val_csv_path") else None

        _flatten_metric_scores(row, epoch_payload.get("train_metric_scores"), "train_score")
        _flatten_metric_scores(row, epoch_payload.get("val_metric_scores"), "val_score")
        epoch_rows.append(row)

    epochs_df = pd.DataFrame(epoch_rows)
    if epochs_df.empty:
        return epochs_df
    return epochs_df.sort_values("epoch").reset_index(drop=True)


def build_prompt_comparison_dataframe(
    metadata: Dict[str, Any],
    best_prompt_payload: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    best_prompt_payload = best_prompt_payload or {}
    best_train = metadata.get("best_on_train", {})
    best_val = metadata.get("best_on_validation", {})
    rows = [
        {
            "prompt_role": "baseline",
            "epoch": 0,
            "source_metric": None,
            "score": None,
            "prompt_preview": _shorten(metadata.get("baseline_prompt"), width=140),
            "prompt_text": metadata.get("baseline_prompt"),
        },
        {
            "prompt_role": "best_saved_json",
            "epoch": best_train.get("best_train_epoch"),
            "source_metric": "train",
            "score": best_train.get("best_train_metric"),
            "prompt_preview": _shorten(best_prompt_payload.get("best_prompt"), width=140),
            "prompt_text": best_prompt_payload.get("best_prompt"),
        },
        {
            "prompt_role": "best_train",
            "epoch": best_train.get("best_train_epoch"),
            "source_metric": "train",
            "score": best_train.get("best_train_metric"),
            "prompt_preview": _shorten(best_train.get("best_train_prompt"), width=140),
            "prompt_text": best_train.get("best_train_prompt"),
        },
        {
            "prompt_role": "best_validation",
            "epoch": best_val.get("best_val_epoch"),
            "source_metric": "validation",
            "score": best_val.get("best_val_metric"),
            "prompt_preview": _shorten(best_val.get("best_val_prompt"), width=140),
            "prompt_text": best_val.get("best_val_prompt"),
        },
    ]
    return pd.DataFrame(rows)


def find_linked_test_metadata(project_root: Path, best_prompt_path: Optional[Path]) -> Optional[Path]:
    if best_prompt_path is None:
        return None

    test_root = project_root / "out" / "test_explainability"
    if not test_root.exists():
        return None

    best_prompt_path_rel = str(best_prompt_path.relative_to(project_root))
    for metadata_path in sorted(test_root.rglob("responses_metadata.json")):
        payload = _load_json(metadata_path)
        raw_best_prompt_path = payload.get("best_prompt_path")
        if not raw_best_prompt_path:
            continue
        candidate = str(_resolve_path(project_root, raw_best_prompt_path).relative_to(project_root))
        if candidate == best_prompt_path_rel:
            return metadata_path
    return None


def load_process_bundle(process_path: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    project_root = project_root or find_project_root(process_path.resolve())
    process_dir = _normalize_process_dir(process_path.resolve())
    metadata_path = process_dir / "optimization_process_metadata.json"
    best_prompt_path = process_dir / "best_prompt.json"

    metadata = _load_json(metadata_path)
    best_prompt_payload = _load_json(best_prompt_path) if best_prompt_path.exists() else {}
    epochs_df = build_epochs_dataframe(
        metadata=metadata,
        project_root=project_root,
        saved_best_prompt=best_prompt_payload.get("best_prompt"),
    )
    prompt_df = build_prompt_comparison_dataframe(metadata, best_prompt_payload)
    linked_test_metadata_path = find_linked_test_metadata(
        project_root=project_root,
        best_prompt_path=best_prompt_path if best_prompt_path.exists() else None,
    )
    linked_test_metadata = (
        _load_json(linked_test_metadata_path) if linked_test_metadata_path is not None else None
    )

    summary = {
        "process_dir": str(process_dir),
        "process_dir_rel": str(process_dir.relative_to(project_root)),
        "metadata_path": str(metadata_path),
        "best_prompt_path": str(best_prompt_path) if best_prompt_path.exists() else None,
        "algorithm": metadata.get("args", {}).get("algorithm"),
        "llm_method": metadata.get("args", {}).get("llm_method"),
        "objective_metric": metadata.get("settings", {}).get("objective_metric", metadata.get("metric")),
        "metric_name": metadata.get("metric_name"),
        "representation_model": metadata.get("settings", {}).get("representation_model"),
        "early_stopping": metadata.get("settings", {}).get("early_stopping"),
        "mmr_lambda_quality": metadata.get("settings", {}).get("mmr_lambda_quality"),
        "mmr_pool_multiplier": metadata.get("settings", {}).get("mmr_pool_multiplier"),
        "epochs_completed": metadata.get("run_summary", {}).get("epochs_completed"),
        "time_prompt_optimization": metadata.get("time_prompt_optimization"),
        "best_train_epoch": metadata.get("best_on_train", {}).get("best_train_epoch"),
        "best_train_metric": metadata.get("best_on_train", {}).get("best_train_metric"),
        "best_val_epoch": metadata.get("best_on_validation", {}).get("best_val_epoch"),
        "best_val_metric": metadata.get("best_on_validation", {}).get("best_val_metric"),
        "saved_best_origin": _saved_best_origin(metadata, best_prompt_payload),
        "linked_test_metadata_path": (
            str(linked_test_metadata_path.relative_to(project_root))
            if linked_test_metadata_path is not None
            else None
        ),
    }

    return {
        "project_root": project_root,
        "process_dir": process_dir,
        "metadata_path": metadata_path,
        "best_prompt_path": best_prompt_path if best_prompt_path.exists() else None,
        "metadata": metadata,
        "best_prompt_payload": best_prompt_payload,
        "epochs_df": epochs_df,
        "prompt_df": prompt_df,
        "summary": summary,
        "linked_test_metadata": linked_test_metadata,
    }


def summarize_processes(process_catalog: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if process_catalog.empty:
        return process_catalog

    if columns is None:
        columns = [
            "algorithm",
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
        ]
    return process_catalog.loc[:, list(columns)].copy()
