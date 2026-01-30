import pandas as pd
from pathlib import Path

from algorithms import default_user_knn_recs, optimized_user_knn_recs, default_bprmf_recs, optimized_bprmf_recs, default_item_knn_recs, optimized_item_knn_recs, default_ncf_recs, optimized_ncf_recs

current_path = Path.cwd()
parent_path = current_path.parent

FINAL_TRAIN_PATH = parent_path / "datasets" / "train_test_oficial" / "train.csv"
FINAL_TEST_PATH = parent_path / "datasets" / "train_test_oficial" / "test.csv"

# OPT_TRAIN_PATH = parent_path / "datasets" / "train_validation_test_oficial" / "train.csv"
# OPT_VALIDATION_PATH = parent_path / "datasets" / "train_validation_test_oficial" / "validation.csv"


def generate_recommendations(algorithm_name, k_vector):

    # -----------------------------
    # Load Dataset
    # -----------------------------
    # train_df = pd.read_csv(TRAIN_PATH, names=["userID", "itemID", "rating"])
    # test_df = pd.read_csv(TEST_PATH, names=["userID", "itemID", "rating"])


    for k_value in k_vector:

        default_recs_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_lists/{algorithm_name}/params_default/K={k_value}/default_{algorithm_name}_K={k_value}_recs.csv"
        default_metrics_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_default/K={k_value}/default_{algorithm_name}_K={k_value}_metrics.csv"
        default_parameters_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_default/K={k_value}/default_{algorithm_name}_K={k_value}_params.csv"

        optimized_recs_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_lists/{algorithm_name}/params_optimized/K={k_value}/optimized_{algorithm_name}_K={k_value}_recs.csv"
        optimized_metrics_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_optimized/K={k_value}/optimized_{algorithm_name}_K={k_value}_metrics.csv"
        optimized_parameters_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_optimized/K={k_value}/optimized_{algorithm_name}_K={k_value}_params.csv"

        match algorithm_name:
            case "user_knn":
                default_user_knn_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, default_recs_output_path, default_metrics_output_path, default_parameters_output_path)
                optimized_user_knn_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, optimized_recs_output_path, optimized_metrics_output_path, optimized_parameters_output_path)

            case "item_knn":
                default_item_knn_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, default_recs_output_path, default_metrics_output_path, default_parameters_output_path)
                optimized_item_knn_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, optimized_recs_output_path, optimized_metrics_output_path, optimized_parameters_output_path)

            case "ncf":
                default_ncf_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, default_recs_output_path, default_metrics_output_path, default_parameters_output_path)
                optimized_ncf_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, optimized_recs_output_path, optimized_metrics_output_path, optimized_parameters_output_path)

            case "bprmf":
                default_bprmf_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, default_recs_output_path, default_metrics_output_path, default_parameters_output_path)
                optimized_bprmf_recs(k_value, FINAL_TRAIN_PATH, FINAL_TEST_PATH, optimized_recs_output_path, optimized_metrics_output_path, optimized_parameters_output_path)
            case "teste":
                # teste(FINAL_TEST_PATH)
                break

            # case "all":



            case _:
                print("Algorithm not found!")
                print("Valid algorithms: userknn, itemknn, ncf, bprmf, all")
                return

    return