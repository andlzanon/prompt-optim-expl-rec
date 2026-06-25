# Prompt Optimization for Explanations in Recommender Systems

## Overview

This repository contains the code and generated artifacts for an anonymous
research project on explainability in recommender systems. The central question
is whether prompt optimization can help Large Language Models select better
explanation paths for recommendations when explanations must balance two
competing properties:

- attribute diversity
- attribute popularity

The experiments are grounded in the movie domain. The repository combines:

- recommendation generation
- knowledge-graph-based explanation path extraction
- a non-LLM explanation baseline based on LOD properties
- an LLM-based explanation pipeline with prompt optimization
- analysis scripts, notebooks, and statistical summaries used to inspect the
  results

## Research motivation

Modern recommender systems often behave as black boxes. Users receive item
suggestions but do not know why those recommendations were produced. This
project studies explanations that connect a recommended item to items from the
user's history through shared semantic attributes such as actors, directors,
genres, and related knowledge-graph properties.

The main experimental setting explored here is:

1. generate recommendations for a user
2. build candidate explanation paths linking historical items to recommended
   items
3. ask a language model to select one path among the candidates
4. optimize the system prompt used by the model so that the selected paths
   better balance diversity and popularity

The repository also includes a non-LLM baseline so the LLM-based approach can
be compared against a comparable graph/LOD explanation method.

## End-to-end pipeline

At a high level, the repository is organized around four stages.

### 1. Recommendation generation

The `recommender/` module generates recommendation lists and recommendation
metrics for:

- `user_knn`
- `item_knn`
- `ncf`
- `bprmf`

The current batch runner evaluates each algorithm for:

- `K = 1, 5, 10, 20, 50, 100, 200`

Both default and optimized recommender configurations are produced and saved
under `datasets/recommendation_files/`.

### 2. Knowledge-graph explanation path extraction

The `knowledge-graphs/` module creates candidate explanation paths that connect
historical items to recommended items through shared attributes from the movie
knowledge graph.

The current script uses the optimized recommendation lists with `K=20` and
writes per-user path files under `datasets/explanation_paths/`.

### 3. Explanation generation

The repository contains two explanation pipelines:

- `explainability-LOD/`: a non-LLM baseline that generates ExpLOD-style
  explanations and computes explanation metrics
- `explainability-with-LLMs/`: an LLM-based pipeline that can run with the
  default prompt or with optimized prompts discovered during prompt search

Both pipelines evaluate explanations with graph-based metrics related to
attribute popularity and diversity.

### 4. Result analysis

The `explainability-with-LLMs/analyse_results/` area contains notebooks,
statistical summaries, and derived tables that support result inspection and
article-oriented reporting.

## Repository organization

```text
prompt-optim-expl-rec/
├── datasets/
├── recommender/
├── knowledge-graphs/
├── explainability-LOD/
├── explainability-with-LLMs/
├── requirements.txt
└── README.md
```

### `datasets/`

Central storage for both input data and generated outputs. The most relevant
subdirectories are:

- `ml-latest-small/`: MovieLens source files
- `knowledge-graphs/`: movie-property knowledge graph extracted from Wikidata
- `recommender_train_test_oficial/`: train/test split used by the recommender
  experiments
- `recommender_train_validation/`: train/validation split used by optimization
  procedures
- `recommendation_files/`: recommendation lists, metrics, and saved parameter
  files for each recommender
- `explanation_paths/`: candidate KG explanation paths
- `lod_results/`: outputs from the non-LLM LOD explanation baseline
- `preselected_explanation_paths/`: cached candidate paths prepared for the LLM
  pipeline
- `user_split_train_val_test/`: train/validation/test user partitions used by
  the explainability-with-LLMs workflow

There is a dataset-focused note in [README_datasets.md](datasets/README_datasets.md).

### `recommender/`

Implements the recommendation stage.

Key files:

- `main_recommendation.py`: top-level batch entry point
- `engine.py`: dispatches the experiments for all algorithms and all `K` values
- `algorithms.py`: wrappers around the underlying recommender methods
- `metrics.py`: recommendation quality metrics

This module writes:

- recommendation lists to `datasets/recommendation_files/recommendation_lists/`
- recommendation metrics and saved parameters to
  `datasets/recommendation_files/recommendation_metrics/`

### `knowledge-graphs/`

Generates candidate explanation paths from the knowledge graph.

Key file:

- `find_explanation_paths.py`

This script reads:

- optimized recommendation lists with `K=20`
- the train interactions
- the movie metadata
- the knowledge-graph property file

and writes per-user explanation path files under `datasets/explanation_paths/`.

### `explainability-LOD/`

Implements a non-LLM explanation baseline based on LOD properties.

Key files:

- `run_LOD.py`: main entry point
- `lod_reordering.py`: semantic-profile construction and explanation logic
- `path_reordering.py`: path handling utilities
- `evaluation_utils.py`: metric computation

This module consumes recommendation outputs and writes:

- explanation text files to `datasets/lod_results/output/`
- per-user metrics to `datasets/lod_results/individual_metrics/`
- aggregate metrics to `datasets/lod_results/average_metrics/`

There is a module-specific README in [explainability-LOD/README.md](explainability-LOD/README.md).

### `explainability-with-LLMs/`

Implements the LLM-based explanation workflow and the prompt-optimization
experiments.

Main areas:

- `run_prepare_selected_paths.py`: prepares reusable candidate-path files
- `run_prompt_optimizer.py`: runs prompt optimization
- `run_llm_explainability.py`: generates explanations with the default or the
  optimized prompt
- `bash/`: batch runners for the end-to-end LLM workflow
- `src/`: LLM wrappers, metrics, representations, and utilities
- `out/`: runtime outputs from prompt optimization and explainability runs
- `analyse_results/`: notebooks, statistical summaries, and generated tables

There is a detailed module README in
[explainability-with-LLMs/README.md](explainability-with-LLMs/README.md).

## Main outputs

The repository stores both intermediate artifacts and final results. The most
important output families are:

### Recommendation outputs

- `datasets/recommendation_files/recommendation_lists/<algorithm>/...`
- `datasets/recommendation_files/recommendation_metrics/<algorithm>/...`

These include:

- default recommendation runs
- optimized recommendation runs
- multiple values of `K`

### Knowledge-graph explanation paths

- `datasets/explanation_paths/<algorithm>-opt/...`

These files contain the candidate item-attribute-item paths later used by the
explanation pipelines.

### LOD explanation baseline results

- `datasets/lod_results/output/`
- `datasets/lod_results/individual_metrics/`
- `datasets/lod_results/average_metrics/`

These correspond to the non-LLM baseline used for comparison.

### LLM prompt-optimization outputs

- `explainability-with-LLMs/out/prompt_optimization/`

These runs include:

- `best_prompt.json`
- `optimization_process_metadata.json`
- per-epoch explanation and metadata artifacts

### LLM explainability outputs

- `explainability-with-LLMs/out/test_explainability/without_optimization/`
- `explainability-with-LLMs/out/test_explainability/with_optimization/`

These runs contain:

- `responses.csv`
- `responses_metadata.json`

### Analysis outputs

- `explainability-with-LLMs/analyse_results/analysis_global/`
- `explainability-with-LLMs/analyse_results/statistical_analysis/`
- `explainability-with-LLMs/analyse_results/generated_tables/`

These are the processed artifacts used to inspect optimization behavior,
compare methods, and prepare compact result tables.

## How to run the project

Run commands from the repository root unless stated otherwise.

### 1. Install the main Python dependencies

Create an environment with Python 3.10 and install:

```bash
python -m pip install -r requirements.txt
```

The root `requirements.txt` is intended for the main repository workflow,
including recommendation generation, path extraction, baseline explainability,
and analysis utilities.

### 2. Generate recommendations

Run:

```bash
python recommender/main_recommendation.py
```

This generates default and optimized recommendation outputs for all four
algorithms and all configured `K` values.

### 3. Generate candidate KG explanation paths

Run:

```bash
python knowledge-graphs/find_explanation_paths.py
```

This currently uses the optimized recommendation files with `K=20`.

### 4. Run the non-LLM LOD explanation baseline

Run:

```bash
python explainability-LOD/run_LOD.py
```

This generates LOD explanations and explanation metrics from the recommendation
outputs.

### 5. Run the LLM-based pipeline

The LLM workflow has its own environment helper and more specific requirements.
See [explainability-with-LLMs/README.md](explainability-with-LLMs/README.md)
for the full setup and execution details.

In short, the main batch entry points are:

```bash
bash explainability-with-LLMs/bash/run_prepare_selected_paths.sh
bash explainability-with-LLMs/bash/run_llm_for_optimization.sh
bash explainability-with-LLMs/bash/run_llm_for_explainability.sh
```

or, to run the full LLM workflow:

```bash
bash explainability-with-LLMs/bash/run.sh
```

## Metrics used in the explanation experiments

The project focuses on explanation quality in terms of attribute popularity and
attribute diversity.

The repository uses:

- `SEP`: a measure related to shared-entity popularity
- `ETD`: explanation type diversity
- `SEP_ETD_F1`: a combined objective used to balance the two

These metrics are used:

- in the non-LLM LOD baseline
- in the LLM explainability evaluation
- as optimization objectives during prompt search

## Prompt optimization in the repository

The LLM pipeline treats explanation generation as a path-selection problem.
For each recommendation, the model receives candidate explanation paths and
must select one of them.

The repository supports two prompt modes:

- a built-in default system prompt
- an optimized prompt loaded from `best_prompt.json`

Prompt optimization is performed inside `explainability-with-LLMs/` and writes
its outputs under `explainability-with-LLMs/out/prompt_optimization/`.

The analysis area under `explainability-with-LLMs/analyse_results/` contains:

- notebooks for inspecting optimization trajectories
- statistical comparisons between optimized and non-optimized runs
- compact tables for article reporting

## Recommended reading order inside the repository

If you want to understand the project quickly, the most useful order is:

1. this root README
2. [explainability-with-LLMs/README.md](explainability-with-LLMs/README.md)
3. [explainability-LOD/README.md](explainability-LOD/README.md)
4. [datasets/README_datasets.md](datasets/README_datasets.md)
