# explainability-with-LLMs

## Overview

This module contains the LLM layer used in the recommender-explainability pipeline. It currently supports two connected workflows:

1. `run_prompt_optimizer.py`
Searches for a better system prompt for explanation-path selection, evaluates prompt candidates on user-disjoint train/validation splits, and saves the best prompt plus per-epoch artifacts.

2. `run_llm_explainability.py`
Runs explanation-path selection for held-out test users, optionally using the optimized prompt found in the previous workflow, and saves both the explanations and their evaluation metadata.

Today the project is organized around:

- LLM-based path selection with Meta Llama models
- prompt optimization with MMR-based reference selection
- text representations via `llm2vec` or `sbert`
- graph-based evaluation with `SEP` and `ETD`
- automatic train/validation/test user splitting under `../datasets/user_split_train_val_test`

The shell scripts in [`bash/`](./bash) are the recommended entry points because they already encode the experiment matrix currently used in the project.

## Current workflow

### Prompt optimization

`bash/run_llm_for_optimization.sh` loops over the configured combinations of:

- model
- recommender algorithm
- representation model
- metric
- early-stopping flag
- MMR hyperparameters

For each combination it runs `run_prompt_optimizer.py`, which:

- ensures the user split exists in `../datasets/user_split_train_val_test`
- uses train users to score prompts
- optionally evaluates on validation users
- stores per-epoch explanations and metrics
- saves in `best_prompt.json` the prompt with the best training score

Epoch `0` evaluates the default built-in prompt. Later epochs generate new candidate prompts from the best previously ranked prompts.

### Explainability generation

`bash/run_llm_for_explainability.sh` now supports two modes in the same batch:

- `default`: uses the built-in prompt from `src/llm/llm_for_explainability.py`
- `best_prompt`: loads a prompt from a discovered `best_prompt.json`

The script always runs the default explainability pass for the configured algorithms. If optimized prompts already exist under `out/prompt_optimization`, it also runs a second pass using those prompts and writes the results under a separate output tree.

The Python entry point `run_llm_explainability.py`:

- loads the held-out test users from `../datasets/user_split_train_val_test/test_users.csv`
- generates explanations for the configured algorithm
- computes the selected objective (`SEP`, `ETD`, or `SEP_ETD_F1`) and stores
  the full SEP/ETD breakdown in the metadata
- writes the explanations CSV and a metadata JSON

### Orchestration scripts

The main orchestration entry point is:

- `bash/run.sh`

It runs, in sequence:

1. `bash/run_prepare_selected_paths.sh`
2. `bash/run_llm_for_optimization.sh`
3. `bash/run_llm_for_explainability.sh`

## Environment setup

The project is configured for Python `3.10.12` in [`settings/.python-version`](./settings/.python-version).

From the `settings` directory:

```bash
cd explainability-with-LLMs/settings
bash setup-llm.sh
source .venv/bin/activate
cd ..
```

What `settings/setup-llm.sh` does:

- installs `python3.10-venv` with `apt`
- creates the virtual environment in `settings/.venv`
- appends `LLMWORKDIR` to `.venv/bin/activate`
- upgrades `pip`, `wheel`, and `setuptools`
- installs the packages from `settings/llm_requirements.txt`

Current Python dependencies are:

- `transformers`
- `torch`
- `accelerate`
- `bitsandbytes`
- `datasets`
- `sentence-transformers`
- `llm2vec`
- `pandas`
- `scikit-learn`
- `pyarrow`
- `fastparquet`

Notes:

- `setup-llm.sh` uses `sudo apt install`, so it expects a machine where you can install system packages.
- The configured LLMs are gated Hugging Face models, so model access must be available for the token resolution used by the code in [`src/llm/token_id.py`](./src/llm/token_id.py).
- The code falls back to CPU if CUDA is unavailable, but the intended setup is a machine capable of running the quantized model stack.

## Expected data layout

The shell scripts assume `--datain ../datasets`. With that layout, the current code expects these inputs:

- `../datasets/ml-latest-small/movies.csv`
- `../datasets/ml-latest-small/ratings.csv`
- `../datasets/recommender_train_test_oficial/train.csv`
- `../datasets/recommender_train_test_oficial/test.csv`
- `../datasets/knowledge-graphs/props_wikidata_movielens_small.csv`
- `../datasets/explanation_paths/<algorithm>-opt/...`

The user split directory is managed automatically:

- `../datasets/user_split_train_val_test/train_users.csv`
- `../datasets/user_split_train_val_test/val_users.csv`
- `../datasets/user_split_train_val_test/test_users.csv`

If this split directory does not exist, it is created automatically from `ml-latest-small/ratings.csv` and then reused by both workflows.

## Main scripts

### `bash/run_llm_for_optimization.sh`

Main batch runner for prompt optimization.

Current defaults in the script:

- models: `Llama3.1-I`
- algorithms: configured inside `bash/run_llm_for_optimization.sh`
- representation models: `llm2vec`, `sbert`
- metrics/objectives: configured inside `bash/run_llm_for_optimization.sh`
- supported metrics/objectives: `sep`, `etd`, `sep_etd_f1`
- explainability settings: `10` recommendations and `10` candidate paths per recommendation

Important parameters already exposed near the top of the script:

- `EPOCHS`
- `META_PROMPT_INSTRUCTION_QUANTITY`
- `EVAL_EVERY`
- `PATIENCE`
- `MIN_DELTA`
- `EARLY_STOPPING_VALUES`
- `MMR_LAMBDA_QUALITIES`
- `MMR_POOL_MULTIPLIERS`

The script builds an experiment root like:

```text
out/prompt_optimization/<model>/<algorithm>/<metric>/repr_<representation>/early_<flag>/mmr_lambda_<value>/mmr_pool_<value>
```

Then `run_prompt_optimizer.py` appends its own run directory below that root:

```text
.../<llm_method>/prompt_opt/<metric>/
```

Inside the final run directory you should expect artifacts such as:

- `best_prompt.json`
- `optimization_process_metadata.json`
- `epoch_000/epoch.json`
- `epoch_000/train_explanations.csv`
- `epoch_000/val_explanations.csv`

### `bash/run_llm_for_explainability.sh`

Main batch runner for explainability generation.

Current defaults in the script:

- algorithms: configured inside `bash/run_llm_for_explainability.sh`
- models: configured inside `bash/run_llm_for_explainability.sh`
- metrics/objectives: configured inside `bash/run_llm_for_explainability.sh`
- supported metrics/objectives: `sep`, `etd`, `sep_etd_f1`
- both execution modes enabled: `RUN_DEFAULT="true"` and `RUN_OPTIMIZED="true"`

Outputs are separated into two roots:

```text
out/test_explainability/without_optimization/
out/test_explainability/with_optimization/
```

Each explainability run writes:

- `responses.csv`
- `responses_metadata.json`

The script scans `out/prompt_optimization/**/best_prompt.json` automatically. When optimized prompts are found, it reconstructs the output path for the corresponding explainability run, filtering the discovered prompts by the configured `MODELS`, `METRICS`, and `ALGORITHMS`, and invokes:

```bash
python3.10 run_llm_explainability.py --prompt_source best_prompt --best_prompt_path ...
```

## Python entry points

### `run_prompt_optimizer.py`

This script:

- parses optimization and explainability arguments
- prepares train/validation users from the auto-generated split
- loads the knowledge-graph properties file
- builds the selected metric function
- loads the LLM used for path selection
- runs the optimization loop through `PromptOptimizer`
- saves the optimization metadata and the final `best_prompt.json`

Most relevant CLI arguments:

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

The graph metrics now evaluate the full set of generated explanations for each
user, without an additional `@k` cutoff.

### `run_llm_explainability.py`

This script:

- prepares explainability inputs restricted to the held-out test users
- loads the knowledge graph and builds the selected metric function
- loads the LLM wrapper
- optionally replaces the default system prompt with a saved `best_prompt.json`
- generates explanations
- saves both the explanations CSV and the metadata JSON

Most relevant CLI arguments:

- `--datain`
- `--algorithm`
- `--llm_method`
- `--prompt_source`
- `--best_prompt_path`
- `--metric`
- `--sep_beta`
- `--num_recommendations`
- `--num_paths_per_recommendation`
- `--include_user_history`
- `--kg_path`
- `--out`

The graph metrics now evaluate the full set of generated explanations for each
user, without an additional `@k` cutoff.

## Typical usage

### Setup once

```bash
cd explainability-with-LLMs/settings
bash setup-llm.sh
source .venv/bin/activate
cd ..
```

### Run only prompt optimization

```bash
bash bash/run_llm_for_optimization.sh
```

### Run only explainability

```bash
bash bash/run_llm_for_explainability.sh
```

### Run the full pipeline in the recommended order

```bash
bash bash/run.sh
```

`bash/run.sh` just calls the three runner scripts in sequence.

## Output summary

### Explainability outputs

Per run, the current explainability flow writes:

- `responses.csv`
- `responses_metadata.json`

The CSV is saved with the stable schema:

- `userId`
- `recommended_item_id`
- `explanation`
- `tries`
- `valid`
- `raw_model_output`

The metadata JSON records, among other things:

- full CLI arguments
- prompt source
- selected metric and parameters
- metric value
- runtime
- number of processed users
- system prompt used in the run

### Prompt-optimization outputs

Per run, the optimizer writes:

- `best_prompt.json`
- `optimization_process_metadata.json`
- one `epoch_<nnn>/` directory per completed epoch

`best_prompt.json` is currently populated from the best training result. Validation metrics are still recorded in the metadata and may stop the run early when early stopping is enabled.

Each epoch directory contains:

- `epoch.json`
- `train_explanations.csv`
- `val_explanations.csv` when validation runs on that epoch

## Project structure

```text
explainability-with-LLMs/
├── bash/
│   ├── run.sh
│   ├── run_prepare_selected_paths.sh
│   ├── run_llm_for_explainability.sh
│   ├── run_llm_for_optimization.sh
│   └── ...
├── settings/
│   ├── .python-version
│   ├── llm_requirements.txt
│   └── setup-llm.sh
├── src/
│   ├── llm/
│   │   ├── llm_for_explainability.py
│   │   ├── llm_for_prompt_optimization.py
│   │   └── token_id.py
│   ├── metrics/
│   │   ├── etd.py
│   │   ├── graph_utils.py
│   │   └── sep.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── base_representation.py
│   │   ├── bkp_l2v.py
│   │   ├── embedding_utils/
│   │   ├── l2v.py
│   │   └── sbert.py
│   └── utils/
│       ├── args.py
│       └── geral.py
├── out/
├── README.md
├── run_llm_explainability.py
└── run_prompt_optimizer.py
```

## Practical notes

- The CLI exposes `--selection_strategy`, but the current LLM wrapper only implements random path sampling.
- Both Python entry points stop if output files with the configured prefix already exist.
- `bash/run_llm_for_explainability.sh` also skips a run when files matching `responses*` already exist in the destination directory.
- If you change the datasets location, update `DATA_DIR` and related path variables in the shell scripts.
