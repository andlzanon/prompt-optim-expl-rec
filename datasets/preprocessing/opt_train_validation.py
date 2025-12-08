import pandas as pd
from recommenders.datasets.python_splitters import (
    python_stratified_split
)

full_dataset_path = "../train_test_oficial/train.csv"

opt_train_path = "../train_validation/opt_train.csv"
opt_validation_path = "../train_validation/opt_validation.csv"


full_df = pd.read_csv(full_dataset_path, names = ["userId", "movieId", "rating", "timestamp"])
full_df["rating"] = (full_df["rating"] > 0).astype(int)
df_to_split = full_df[["userId", "movieId", "rating"]]

# dividir rating em treino e validacao 90/10

opt_train_splitted, opt_validation_splitted = python_stratified_split(
    df_to_split, filter_by="user", min_rating=10, ratio=0.90,
    col_user="userId", col_item="movieId"
)

# Merge on all common columns to reattach the timestamp
opt_train = opt_train_splitted.merge(
    full_df[['userId', 'movieId', 'rating', 'timestamp']],
    on=['userId', 'movieId', 'rating'],
    how='left'
)

opt_validation = opt_validation_splitted.merge(
    full_df[['userId', 'movieId', 'rating', 'timestamp']],
    on=['userId', 'movieId', 'rating'],
    how='left'
)

opt_train.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
opt_validation.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)

opt_train.to_csv(opt_train_path, index=False, header=False)
opt_validation.to_csv(opt_validation_path, index=False, header=False)

