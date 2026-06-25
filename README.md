# Prompt Optimization for Better Explanations in Recommender Systems

This project studies how prompt optimization techniques can improve explanations for recommender systems.

## Reproduction

Run all commands from the repository root.

### 1. Environment

Create and activate a Python environment. For example, with `conda`:

```bash
conda create -n prompt-optim-expl-rec python=3.10
conda activate prompt-optim-expl-rec
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

If `CaseRecommender` is not installed correctly from `requirements.txt`, install it from GitHub:

```bash
python -m pip install -U git+https://github.com/caserec/CaseRecommender.git
```

### 2. Generate Recommendations

This step generates recommendation lists, metrics, and parameter files for:

- `user_knn`
- `item_knn`
- `ncf`
- `bprmf`

It runs each algorithm for:

```text
K = 1, 5, 10, 20, 50, 100, 200
```

Run:

```bash
python recommender/main_recommendation.py
```

The outputs are saved under:

```text
datasets/recommendation_files/recommendation_lists/
datasets/recommendation_files/recommendation_metrics/
```

### 3. Generate Knowledge-Graph Explanation Paths

This step builds item-property explanation paths using the optimized recommendation files with `K=20`.

Run:

```bash
python knowledge-graphs/find_explanation_paths.py
```

The outputs are saved under:

```text
datasets/explanation_paths/
```

### 4. Generate LOD Explanations and Metrics

This step generates ExpLOD-style explanations and computes explanation metrics using the optimized recommendation files with `K=20`.

Run:

```bash
python explainability-LOD/run_LOD.py
```

The outputs are saved under:

```text
datasets/lod_results/output/
datasets/lod_results/individual_metrics/
datasets/lod_results/average_metrics/
```

## Project Organization

```text
datasets/                  Input data and generated experiment outputs
recommender/               Recommendation algorithms and evaluation
knowledge-graphs/          KG-based explanation path extraction
explainability-LOD/        LOD explanation generation and metrics
explainability-with-LLMs/  LLM-based explanation experiments
```

## Notes

- The recommendation step can be expensive because it runs default and optimized versions of all four algorithms.
- The KG and LOD explanation steps currently consume the optimized recommendation files with `K=20`.
- Scripts are configured to be executed from the repository root.

## Project's Development Process

We will develop our code on the dev branch, that represents a paper we are developing. Once we finish the code to this paper, we will merge dev with main.

Therefore, to develop a feature use the following steps:

1. Create a branch from develop

Switch to the develop branch and pull the latest changes:

```bash
git checkout develop
git pull origin develop
```

Create a new branch from dev for your changes:

```bash
git checkout -b <id>_my-new-feature
```

where `<id>` is the id of the GitHub Issue.

Use a descriptive branch name.

2. Make your changes

Edit, add, or remove files as needed. Stage and commit your changes:

```bash
git add .
git commit -m "Add: short description of your change"
```

3. Push your branch to your fork

```bash
git push origin my-new-feature
```

4. Open a Pull Request

- Go to the repo on GitHub.
- Click Compare & pull request next to your branch.
- Set the base repository to the original repo's dev branch.
- Add a clear title and description for your changes.
- Click Create pull request.

5. Review and merge

The maintainers will review your PR. Once approved, it will be merged into dev.

Tips:

- Keep your branch up-to-date with dev:

```bash
git fetch origin
git merge origin/dev
```

- Make small, focused commits with clear messages.
- Follow the project's coding style and conventions.
