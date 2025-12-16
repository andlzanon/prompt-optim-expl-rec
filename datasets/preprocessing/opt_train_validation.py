import pandas as pd
from recommenders.datasets.python_splitters import python_stratified_split

# Paths
full_dataset_path = "../train_test_oficial/train.csv"
opt_train_path = "../train_validation/opt_train.csv"
opt_validation_path = "../train_validation/opt_validation.csv"

# Load full training dataset
full_df = pd.read_csv(
    full_dataset_path,
    names=["userId", "movieId", "rating", "timestamp"]
)

# Convert ratings to implicit feedback
full_df["rating"] = (full_df["rating"] > 0).astype(int)

# Keep only required columns for splitting
df_to_split = full_df[["userId", "movieId", "rating"]]

# Stratified split by user (90% train, 10% validation)
opt_train_split, opt_validation_split = python_stratified_split(
    df_to_split,
    filter_by="user",
    min_rating=10,
    ratio=0.90,
    col_user="userId",
    col_item="movieId"
)

# Reattach timestamps to training split
opt_train = opt_train_split.merge(
    full_df[["userId", "movieId", "rating", "timestamp"]],
    on=["userId", "movieId", "rating"],
    how="left"
)

# Reattach timestamps to validation split
opt_validation = opt_validation_split.merge(
    full_df[["userId", "movieId", "rating", "timestamp"]],
    on=["userId", "movieId", "rating"],
    how="left"
)

# Sort for deterministic output
opt_train.sort_values(["userId", "movieId"], inplace=True)
opt_validation.sort_values(["userId", "movieId"], inplace=True)

# Save datasets (no header)
opt_train.to_csv(opt_train_path, index=False, header=False)
opt_validation.to_csv(opt_validation_path, index=False, header=False)
