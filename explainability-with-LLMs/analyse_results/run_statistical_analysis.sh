#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/.venv/bin/python}"
GENERATOR_SCRIPT="$SCRIPT_DIR/scripts/generate_statistical_analysis.py"
LLM_METHOD="${LLM_METHOD:-Llama3.1-I}"
ARTICLE_DIR="${ARTICLE_DIR:-$SCRIPT_DIR}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN or create the virtual environment at $WORKSPACE_ROOT/.venv." >&2
  exit 1
fi

if [ ! -f "$GENERATOR_SCRIPT" ]; then
  echo "Generator script not found: $GENERATOR_SCRIPT" >&2
  exit 1
fi

MISSING_MODULES="$(
  "$PYTHON_BIN" - <<'PY'
import importlib.util

required = ["pandas", "numpy", "scipy", "sklearn"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"

if [ -n "$MISSING_MODULES" ]; then
  echo "Missing Python modules in $PYTHON_BIN: $MISSING_MODULES" >&2
  echo "Install the project dependencies first, for example:" >&2
  echo "  $PYTHON_BIN -m pip install -r $WORKSPACE_ROOT/requirements.txt" >&2
  exit 1
fi

cd "$WORKSPACE_ROOT"

CMD=(
  "$PYTHON_BIN"
  "$GENERATOR_SCRIPT"
  --project-root "$PROJECT_ROOT"
  --llm-method "$LLM_METHOD"
  --article-dir "$ARTICLE_DIR"
)

if [ "$#" -gt 0 ]; then
  CMD+=( "$@" )
fi

echo "========================================"
echo "Generating statistical analysis"
echo "Workspace root: $WORKSPACE_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "LLM method: $LLM_METHOD"
echo "Article dir: $ARTICLE_DIR"
echo "Output dir: $SCRIPT_DIR/statistical_analysis"
echo "========================================"

"${CMD[@]}"
