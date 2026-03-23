# explainability-with-LLMs

## Project overview

This project runs an LLM-based explanation-path selection workflow for recommender-system explanations and also provides a prompt-optimization pipeline for improving the system instruction used by the LLM.

At a high level, the project supports two main workflows:

- `run_llm_explainability.py`: generates explanation-path selections for users and saves the resulting explanations and metadata.
- `run_prompt_optimizer.py`: evaluates prompt variants across training and validation users, optimizes the system instruction, and saves the best prompt together with per-epoch artifacts.

The codebase is organized around reusable modules for:

- LLM interaction and prompt optimization
- text representations (`llm2vec` and `sbert`)
- graph-based metrics (`SEP` and `ETD`)
- utility functions for data preparation, argument parsing, and persistence

The shell scripts in `bash/` are the main execution entry points and already contain the project’s configured parameters.

The project is configured to use Python `3.10.12` (see `settings/.python-version`).

## Setup and execution

### 1. Run the environment setup script

Run the setup script from the `settings` directory:

```bash
cd explainability-with-LLMs/settings
bash setup-llm.sh
```

What this script does:

- installs `python3.10-venv`
- creates the virtual environment in `settings/.venv`
- appends the `LLMWORKDIR` environment variable to the activation script
- activates the environment
- upgrades `pip`, `wheel`, and `setuptools`
- installs the dependencies from `llm_requirements.txt`

### 2. Activate the virtual environment

After setup, activate the virtual environment:

```bash
source .venv/bin/activate
```

### 3. Leave the `settings` directory

Return to the project root:

```bash
cd ..
```

At this point, you should be in the `explainability-with-LLMs` directory.

### 4. Run the main shell scripts

The execution order is:

1. `cd explainability-with-LLMs/settings`
2. `bash setup-llm.sh`
3. `source .venv/bin/activate`
4. `cd ..`
5. run one of the main shell scripts from the project root

Run the prompt-optimization workflow:

```bash
bash bash/run_llm_for_optimization.sh
```

Run the explainability-generation workflow:

```bash
bash bash/run_llm_for_explainability.sh
```

### Notes on paths and expected data

The shell scripts currently expect:

- datasets at `../datasets`
- the knowledge-graph CSV at `../knowledge-graphs/props_wikidata_movielens_small.csv` for prompt optimization

These paths are defined directly in the shell scripts, so if your local structure is different, you will need to update the script variables.

## Script descriptions

### `bash/run_llm_for_optimization.sh`

This is the main shell entry point for prompt optimization.

Role in the pipeline:

- defines dataset and knowledge-graph paths
- defines models, algorithms, representation models, metrics, and optimization hyperparameters
- loops over the configured combinations
- runs `run_prompt_optimizer.py` with the assembled command-line arguments

This workflow produces optimization artifacts under `out/prompt_optimization/...`, including per-epoch files and the best prompt found during the run.

### `run_prompt_optimizer.py`

This Python script orchestrates the prompt-optimization process.

Role in the pipeline:

- parses optimization arguments
- prepares train/validation inputs and the knowledge graph
- builds the metric function
- loads the LLM wrapper
- runs the optimization loop through `PromptOptimizer`
- saves optimization metadata and the best prompt

### `bash/run_llm_for_explainability.sh`

This is the main shell entry point for explanation generation.

Role in the pipeline:

- defines the model, algorithm, and explainability parameters
- loops over the configured combinations
- runs `run_llm_explainability.py` with the assembled command-line arguments

This workflow produces explanation outputs under `out/explainability/...`.

### `run_llm_explainability.py`

This Python script orchestrates explanation generation for the configured users.

Role in the pipeline:

- parses explainability arguments
- prepares user interaction inputs
- loads the LLM wrapper
- generates explanation-path selections
- saves explanations and metadata

## Project structure

Below is the relevant structure for using and maintaining the project:

```text
explainability-with-LLMs/
├── bash/
│   ├── run_llm_for_explainability.sh
│   └── run_llm_for_optimization.sh
├── out/
├── settings/
│   ├── .python-version
│   ├── llm_requirements.txt
│   └── setup-llm.sh
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_for_explainability.py
│   │   ├── llm_for_prompt_optimization.py
│   │   └── token_id.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── etd.py
│   │   ├── graph_utils.py
│   │   └── sep.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── base_representation.py
│   │   ├── bkp_l2v.py
│   │   ├── l2v.py
│   │   ├── sbert.py
│   │   └── embedding_utils/
│   │       ├── __init__.py
│   │       ├── mmr.py
│   │       └── similarity.py
│   └── utils/
│       ├── __init__.py
│       ├── args.py
│       └── geral.py
├── README.md
├── run_llm_explainability.py
└── run_prompt_optimizer.py
```

### `bash/`

Contains the main shell scripts used to run the project workflows. These files define the execution parameters and serve as the main operational entry points.

### `settings/`

Contains environment-setup files:

- `.python-version`: Python version indicator for the project, currently `3.10.12`
- `setup-llm.sh`: creates and configures the Python environment
- `llm_requirements.txt`: lists the Python dependencies used by the project

### `src/llm/`

Contains the LLM-related implementation:

- `llm_for_explainability.py`: LLM wrapper used to select explanation paths
- `llm_for_prompt_optimization.py`: prompt-optimization loop and support logic
- `token_id.py`: access-token helper used during model loading

### `src/metrics/`

Contains the evaluation logic:

- `etd.py`: ETD metric
- `sep.py`: SEP metric
- `graph_utils.py`: helpers that convert textual explanations into structures consumed by the metrics

### `src/representation/`

Contains the text-representation layer:

- `base_representation.py`: common representation interface
- `sbert.py`: Sentence-Transformers backend
- `l2v.py`: LLM2Vec backend
- `embedding_utils/`: similarity and MMR helpers used during prompt optimization

### `src/utils/`

Contains project utilities:

- `args.py`: command-line argument parsing for both workflows
- `geral.py`: data preparation, split handling, persistence, and metric integration helpers

### `out/`

Stores generated artifacts such as:

- explainability outputs
- prompt-optimization results
- metadata files
- per-epoch CSV and JSON files

## Typical usage summary

If you want the shortest practical sequence, use:

```bash
cd explainability-with-LLMs/settings
bash setup-llm.sh
source .venv/bin/activate
cd ..
bash bash/run_llm_for_optimization.sh
# or
bash bash/run_llm_for_explainability.sh
```