import pandas as pd
from pathlib import Path

from algorithms import user_knn_recs, item_knn_recs, bprmf_opt_recs, final_ncf_recs

current_path = Path.cwd()
parent_path = current_path.parent
TRAIN_PATH = parent_path / "datasets" / "train_test_oficial" / "train.dat"
TEST_PATH = parent_path / "datasets" / "train_test_oficial" / "test.dat"

TRAIN_PATH_VALID = parent_path / "datasets" / "train_validation_test_oficial" / "train.dat"
VALIDATION_PATH_VALID = parent_path / "datasets" / "train_validation_test_oficial" / "validation.dat"
TEST_PATH_VALID = parent_path / "datasets" / "train_validation_test_oficial" / "test.dat"


def generate_recommendations(algorithm_name, k_vector):

    # -----------------------------
    # Load Dataset
    # -----------------------------
    train_df = pd.read_csv(TRAIN_PATH, names=["userID", "itemID", "rating"])
    test_df = pd.read_csv(TEST_PATH, names=["userID", "itemID", "rating"])


    for k_value in k_vector:

        output_recs_path = f"{parent_path}/datasets/recommendation_files/recommendation_lists/{algorithm_name}/{algorithm_name}_K={k_value}_recs.csv"
        output_metrics_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/{algorithm_name}_K={k_value}_metrics.csv"

        match algorithm_name:
            case "userknn":
                user_knn_recs(k_value, TRAIN_PATH_VALID, VALIDATION_PATH_VALID, TEST_PATH_VALID, output_recs_path, output_metrics_path)

            case "itemknn":
                item_knn_recs(k_value, TRAIN_PATH_VALID, VALIDATION_PATH_VALID, TEST_PATH_VALID, output_recs_path, output_metrics_path)

            case "final_ncf":
                final_ncf_recs(k_value, TRAIN_PATH_VALID, VALIDATION_PATH_VALID, TEST_PATH_VALID, output_recs_path, output_metrics_path)

            case "bprmf":
                bprmf_opt_recs(k_value, TRAIN_PATH_VALID, VALIDATION_PATH_VALID, TEST_PATH_VALID, output_recs_path, output_metrics_path)

            # case "all":



            case _:
                print("Algorithm not found!")
                print("Valid algorithms: userknn, itemknn, ncf, bprmf, all")
                return

    return