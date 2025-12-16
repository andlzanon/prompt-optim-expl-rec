import pandas as pd
from recommenders.datasets.python_splitters import python_stratified_split

# Paths
ml_ratings_path = "../ml-latest-small/ratings.csv"
train_path = "../train_test_oficial/train.csv"
test_path = "../train_test_oficial/test.csv"

# Load ratings data
ratings_df = pd.read_csv(ml_ratings_path)

# Convert explicit ratings to implicit feedback
ratings_df["rating"] = (ratings_df["rating"] > 0).astype(int)

# Keep only required columns for splitting
ratings_to_split = ratings_df[["userId", "movieId", "rating"]]

# Stratified split by user (90% train, 10% test)
train_split, test_split = python_stratified_split(
    ratings_to_split,
    filter_by="user",
    min_rating=10,
    ratio=0.90,
    col_user="userId",
    col_item="movieId"
)

# Reattach timestamps
train_df = train_split.merge(
    ratings_df[["userId", "movieId", "rating", "timestamp"]],
    on=["userId", "movieId", "rating"],
    how="left"
)

test_df = test_split.merge(
    ratings_df[["userId", "movieId", "rating", "timestamp"]],
    on=["userId", "movieId", "rating"],
    how="left"
)

# Sort for consistency
train_df.sort_values(["userId", "movieId"], inplace=True)
test_df.sort_values(["userId", "movieId"], inplace=True)

# Save final datasets (no header)
train_df.to_csv(train_path, index=False, header=False)
test_df.to_csv(test_path, index=False, header=False)
