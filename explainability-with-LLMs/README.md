# explainability-with-LLMs

## Overview

This directory contains the LLM-based explainability pipeline used in the
repository. It covers three main stages:

1. preparing candidate explanation paths for each recommender
2. optimizing the system prompt used to select explanation paths
3. generating and evaluating explanations with either the default or the
   optimized prompt

The directory also includes a separate analysis area with notebooks, statistical
summaries, and article-oriented tables derived from the generated outputs.

## Current layout

```text
explainability-with-LLMs/
├── analyse_results/
│   ├── analysis_global/
│   ├── generated_tables/
│   ├── scripts/
│   ├── statistical_analysis/
│   └── run_statistical_analysis.sh
├── bash/
│   ├── run.sh
│   ├── run_prepare_selected_paths.sh
│   ├── run_llm_for_optimization.sh
│   ├── run_llm_for_explainability.sh
│   └── shared_llm_batch_config.sh
├── out/
│   ├── prompt_optimization/
│   └── test_explainability/
├── settings/
│   ├── .python-version
│   ├── llm_requirements.txt
│   └── setup-llm.sh
├── src/
│   ├── llm/
│   ├── metrics/
│   ├── representation/
│   └── utils/
├── run_prepare_selected_paths.py
├── run_prompt_optimizer.py
├── run_llm_explainability.py
└── README.md
```

## What each area does

### `src/`

Core implementation for:

- LLM prompting and generation
- graph-based metrics (`SEP`, `ETD`, `SEP_ETD_F1`)
- text representations (`llm2vec`, `sbert`)
- optimization-analysis helpers

### `bash/`

Batch runners for the main workflow:

- `run_prepare_selected_paths.sh`: creates reusable candidate-path files for
  optimization users and explainability users
- `run_llm_for_optimization.sh`: runs prompt optimization across the configured
  experiment grid
- `run_llm_for_explainability.sh`: runs explainability evaluation with the
  default prompt and, when available, the optimized prompt
- `run.sh`: executes the full pipeline in sequence for `sep`, `etd`, and
  `sep_etd_f1`

Shared defaults live in `bash/shared_llm_batch_config.sh`. At the moment, the
batch scripts are configured for:

- algorithms: `user_knn`, `item_knn`, `ncf`, `bprmf`
- selection strategy: `random`
- recommendations per user: `10`
- candidate paths per recommendation: `10`
- seed: `2026`

### `out/`

Runtime outputs from the pipeline:

- `out/prompt_optimization/`: optimization runs and `best_prompt.json` files
- `out/test_explainability/without_optimization/`: explainability runs using
  the default prompt
- `out/test_explainability/with_optimization/`: explainability runs using
  optimized prompts discovered under `out/prompt_optimization/`

### `analyse_results/`

Post-processing and reporting artifacts:

- `analysis_global/`: notebooks for global result inspection and figure/table
  generation
- `statistical_analysis/`: per-metric statistical summaries, pairwise Wilcoxon
  results, compact article tables, and cached per-user metric files
- `generated_tables/`: LaTeX tables derived from the statistical outputs
- `scripts/`: Python generators for the notebooks and statistical summaries
- `run_statistical_analysis.sh`: wrapper to rebuild the statistical analysis

## Main workflow

### 1. Prepare selected paths

Run:

```bash
bash bash/run_prepare_selected_paths.sh
```

This creates `selected_paths.csv` and `selected_paths_metadata.json` under:

```text
../datasets/preselected_explanation_paths/<algorithm>/<user_scope>/random/recs_10_paths_10/seed_2026/
```

The current user scopes are:

- `optimization`: train + validation users
- `explainability`: test users

### 2. Optimize prompts

Run:

```bash
bash bash/run_llm_for_optimization.sh
```

The current optimization grid includes:

- model: `Llama3.1-I`
- metrics: `sep`, `etd`, `sep_etd_f1`
- representation models: `llm2vec`, `sbert`
- early stopping flag: `false`
- MMR lambda: `0.0`, `0.5`, `1.0`
- MMR pool multiplier: `10`

Outputs are written under:

```text
out/prompt_optimization/<model>/<algorithm>/<metric>/repr_<representation>/early_<flag>/mmr_lambda_<value>/mmr_pool_<value>/<model>/prompt_opt/<metric>/
```

Each completed run may contain:

- `best_prompt.json`
- `optimization_process_metadata.json`
- `epoch_<nnn>/epoch.json`
- `epoch_<nnn>/train_explanations.csv`
- `epoch_<nnn>/val_explanations.csv`

### 3. Run explainability

Run:

```bash
bash bash/run_llm_for_explainability.sh
```

The script executes two modes:

- `default`: uses the built-in prompt
- `best_prompt`: uses `best_prompt.json` files found under
  `out/prompt_optimization/`

Outputs are written under:

```text
out/test_explainability/without_optimization/
out/test_explainability/with_optimization/
```

Each explainability run writes:

- `responses.csv`
- `responses_metadata.json`

### 4. Run the full pipeline

Run:

```bash
bash bash/run.sh
```

This executes:

1. selected-path preparation
2. prompt optimization
3. explainability evaluation

for each metric in:

- `sep`
- `etd`
- `sep_etd_f1`

## Statistical analysis and reporting

To rebuild the statistical summaries under `analyse_results/statistical_analysis`,
run:

```bash
bash analyse_results/run_statistical_analysis.sh
```

By default, this script expects:

- a Python executable at `../.venv/bin/python`, relative to this directory's
  parent project root
- the packages required by `requirements.txt`

You can override the interpreter if needed:

```bash
PYTHON_BIN=/path/to/python bash analyse_results/run_statistical_analysis.sh
```

The generated statistical outputs are organized by metric:

- `analyse_results/statistical_analysis/sep/results/`
- `analyse_results/statistical_analysis/etd/results/`
- `analyse_results/statistical_analysis/sep_etd_f1/results/`

The notebooks in `analyse_results/analysis_global/` consume these outputs to
produce cross-run comparisons and publication-oriented summaries.

## Environment setup

The LLM pipeline is configured for Python `3.10.12` in
`settings/.python-version`.

Typical setup:

```bash
cd explainability-with-LLMs/settings
bash setup-llm.sh
source .venv/bin/activate
cd ..
```

`settings/setup-llm.sh` creates a virtual environment in `settings/.venv` and
installs the packages listed in `settings/llm_requirements.txt`.

The batch runners in `bash/` call `python3.10` directly, so `python3.10` must
be available in the shell environment used to launch them.

## Expected input data

The runners assume `DATA_DIR="../datasets"` and currently expect, at minimum:

- `../datasets/ml-latest-small/movies.csv`
- `../datasets/ml-latest-small/ratings.csv`
- `../datasets/recommender_train_test_oficial/train.csv`
- `../datasets/recommender_train_test_oficial/test.csv`
- `../datasets/knowledge-graphs/props_wikidata_movielens_small.csv`

The user split is created automatically when missing under:

- `../datasets/user_split_train_val_test/train_users.csv`
- `../datasets/user_split_train_val_test/val_users.csv`
- `../datasets/user_split_train_val_test/test_users.csv`

## Important CLI entry points

### `run_prepare_selected_paths.py`

Builds reusable candidate-path files per algorithm and user scope.

### `run_prompt_optimizer.py`

Runs prompt optimization and writes `best_prompt.json` plus per-epoch artifacts.

Relevant arguments include:

- `--datain`
- `--kg_path`
- `--algorithm`
- `--llm_method`
- `--representation_model`
- `--epochs`
- `--eval_every`
- `--patience`
- `--min_delta`
- `--early_stopping`
- `--mmr_lambda_quality`
- `--mmr_pool_multiplier`
- `--metric`
- `--sep_beta`
- `--out`

### `run_llm_explainability.py`

Generates explanations for held-out test users and evaluates them with the
selected metric.

Relevant arguments include:

- `--datain`
- `--kg_path`
- `--algorithm`
- `--llm_method`
- `--prompt_source`
- `--best_prompt_path`
- `--metric`
- `--sep_beta`
- `--num_recommendations`
- `--num_paths_per_recommendation`
- `--include_user_history`
- `--selected_paths_input_path`
- `--out`