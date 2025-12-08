import pandas as pd
from recommenders.datasets.python_splitters import (
    python_stratified_split
)

ml_ratings_path = "../ml-latest-small/ratings.csv"

train_path = "../train_test_oficial/train.csv"
test_path = "../train_test_oficial/test.csv"


ratings_df = pd.read_csv(ml_ratings_path)
ratings_df["rating"] = (ratings_df["rating"] > 0).astype(int)
ratings_to_split = ratings_df[["userId", "movieId", "rating"]]

# dividir rating em temp e teste 90/10

train_splitted, test_splitted = python_stratified_split(
    ratings_to_split, filter_by="user", min_rating=10, ratio=0.90,
    col_user="userId", col_item="movieId"
)

# Merge on all common columns to reattach the timestamp
train_df = train_splitted.merge(
    ratings_df[['userId', 'movieId', 'rating', 'timestamp']],
    on=['userId', 'movieId', 'rating'],
    how='left'
)

test_df = test_splitted.merge(
    ratings_df[['userId', 'movieId', 'rating', 'timestamp']],
    on=['userId', 'movieId', 'rating'],
    how='left'
)

train_df.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
test_df.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)

train_df.to_csv(train_path, index=False, header=False)
test_df.to_csv(test_path, index=False, header=False)

