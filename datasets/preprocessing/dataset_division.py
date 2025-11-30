import pandas as pd
from recommenders.datasets.python_splitters import (
    python_stratified_split
)

ml_ratings_path = "../ml-latest-small/ratings.csv"

new_train_default_path = "../train_test_oficial/train.csv"
new_test_default_path = "../train_test_oficial/test.csv"

new_train_opt_path = "../train_validation_test_oficial/train.csv"
new_validation_opt_path = "../train_validation_test_oficial/validation.csv"
new_test_opt_path = "../train_validation_test_oficial/test.csv"


ratings_df = pd.read_csv(ml_ratings_path)
ratings_df["rating"] = (ratings_df["rating"] > 0).astype(int)
ratings_df = ratings_df[["userId", "movieId", "rating"]]



default_train, default_test = python_stratified_split(
    ratings_df, filter_by="user", min_rating=10, ratio=0.7,
    col_user="userId", col_item="movieId"
)

opt_validation, opt_test = python_stratified_split(
    default_test, filter_by="user", min_rating=1, ratio=0.5,
    col_user="userId", col_item="movieId"
)

default_train.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
default_test.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
opt_validation.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
opt_test.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)

default_train = default_train[["userId", "movieId", "rating"]]
default_test = default_test[["userId", "movieId", "rating"]]
opt_validation = opt_validation[["userId", "movieId", "rating"]]
opt_test = opt_test[["userId", "movieId", "rating"]]

default_test.to_csv(new_test_default_path, index=False, header=False)
default_train.to_csv(new_train_default_path, index=False, header=False)

default_train.to_csv(new_train_opt_path, index=False, header=False)
opt_validation.to_csv(new_validation_opt_path, index=False, header=False)
opt_test.to_csv(new_test_opt_path, index=False, header=False)

