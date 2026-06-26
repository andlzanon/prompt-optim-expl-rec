#!/bin/bash

# Minimal shared configuration used by all bash runners.
# Keep only values that are truly common to the batch scripts.

DATA_DIR="../datasets"
SELECTED_PATHS_ROOT="${DATA_DIR}/preselected_explanation_paths"
ALGORITHMS=("user_knn" "item_knn" "ncf" "bprmf")
SELECTION_STRATEGY="random"
NUM_RECOMMENDATIONS=10
NUM_PATHS_PER_RECOMMENDATION=10
SEED=2026

selected_paths_dir() {
  local algorithm="$1"
  local user_scope="$2"

  printf "%s/%s/%s/%s/recs_%s_paths_%s/seed_%s" \
    "$SELECTED_PATHS_ROOT" \
    "$algorithm" \
    "$user_scope" \
    "$SELECTION_STRATEGY" \
    "$NUM_RECOMMENDATIONS" \
    "$NUM_PATHS_PER_RECOMMENDATION" \
    "$SEED"
}

selected_paths_csv_path() {
  local algorithm="$1"
  local user_scope="$2"

  printf "%s/selected_paths.csv" "$(selected_paths_dir "$algorithm" "$user_scope")"
}

selected_paths_metadata_path() {
  local algorithm="$1"
  local user_scope="$2"

  printf "%s/selected_paths_metadata.json" "$(selected_paths_dir "$algorithm" "$user_scope")"
}

files_all_exist() {
  local path

  for path in "$@"; do
    if [ ! -f "$path" ]; then
      return 1
    fi
  done

  return 0
}

any_path_exists() {
  local path

  for path in "$@"; do
    if [ -e "$path" ]; then
      return 0
    fi
  done

  return 1
}

prepare_output_slot() {
  local label="$1"
  local out_dir="$2"
  local cleanup_dir="$3"
  shift 3
  local required_files=("$@")

  if files_all_exist "${required_files[@]}"; then
    echo "Skipping because ${label} output already exists under ${out_dir}"
    return 1
  fi

  if [ -d "$cleanup_dir" ] || any_path_exists "${required_files[@]}"; then
    echo "Partial ${label} output found under ${out_dir}; cleaning partial files and rerunning this case."
    rm -rf "$cleanup_dir"
  fi

  return 0
}
