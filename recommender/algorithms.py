import pandas as pd
import numpy as np
from pathlib import Path
from recommenders.utils.timer import Timer
from utils.dir_manipulation import delete_file, reset_dir
from utils.print_aux import print_params
import optuna
from optuna.samplers import TPESampler

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset

import warnings
warnings.filterwarnings("ignore", message=".*swapaxes.*")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages


from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from caserec.recommenders.item_recommendation.bprmf import BprMF


current_path = Path.cwd()
parent_path = current_path.parent


# MODELS

#   - User_knn
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/userknn.py

#   - Item_knn
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/itemknn.py

#    - BprMF
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/bprmf.py

#   - NCF
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb



# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------


def default_user_knn_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
):
    """
    Trains and evaluates a UserKNN model with default hyperparameters.
    """

    # Load training data
    train_df = pd.read_csv(
        FINAL_train_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )[["userID", "itemID", "rating"]]

    # Define default hyperparameters
    sim_metric = "cosine"
    num_users = train_df["userID"].nunique()
    k_neighbors = int(num_users ** 0.5)

    params_dict = {
        "sim_metric": sim_metric,
        "k_neighbors": k_neighbors
    }

    def build_userknn_model(output_file):
        """
        Builds a UserKNN model with default settings.
        """
        return UserKNN(
            train_file=FINAL_train_path,
            test_file=FINAL_test_path,
            output_file=output_file,
            sep=",",
            output_sep=",",
            rank_length=TOP_K
        )

    print(f"\nGenerating recommendations: USER_KNN | TOP_K={TOP_K}\n")
    print_params(params_dict)

    # Train model and generate recommendations
    final_model = build_userknn_model(FINAL_recs_output_path)
    final_model.compute(verbose=True)

    # Evaluate on test set
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    # Save metrics
    metrics_df = pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    )
    metrics_df.to_csv(FINAL_metrics_output_path, index=False)

    # Save parameters
    params_df = pd.DataFrame(
        list(params_dict.items()),
        columns=["parameter", "value"]
    )
    params_df.to_csv(FINAL_parameters_output_path, index=False)

    return 0







def optimized_user_knn_recs(
    TOP_K, 
    FINAL_train_path, 
    FINAL_test_path, 
    FINAL_recs_output_path, 
    FINAL_metrics_output_path, 
    FINAL_parameters_output_path
):
    """
    Optimizes and trains a UserKNN model using Optuna, then evaluates and saves the results.
    """

    OPT_DIR = "../datasets/recommender_train_validation"
    OPT_recs_output_path = f"utils/user_item_knn/user_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    def evaluate_user_knn(k, sim_metric):
        """
        Evaluates the UserKNN model with the given hyperparameters (k_neighbors, similarity_metric).
        """
        OPT_train_path = f"{OPT_DIR}/opt_train.csv"
        OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

        delete_file(OPT_recs_output_path)

        # Initialize UserKNN model with current hyperparameters
        OPT_user_knn_model = UserKNN(
            train_file=OPT_train_path,
            test_file=OPT_validation_path,
            output_file=OPT_recs_output_path,
            sep=',',
            output_sep=',',
            k_neighbors=k,
            similarity_metric=sim_metric,
            rank_length=TOP_K
        )

        OPT_user_knn_model.compute(verbose=False)

        # Evaluate results on validation file
        metrics_dict = ItemRecommendationEvaluation(
            sep=",", 
            n_ranks=[TOP_K]
        ).evaluate_with_files(OPT_recs_output_path, OPT_validation_path)

        return metrics_dict[metric_key]

    def objective(trial):
        """
        Objective function for Optuna optimization, defining the hyperparameter search space.
        """
        suggested_k_neighbors = trial.suggest_int("k_neighbors", 1, 100)
        suggested_similarity_metric = trial.suggest_categorical(
            "similarity_metric", ["jaccard", "cosine"]
        )

        score = evaluate_user_knn(suggested_k_neighbors, suggested_similarity_metric)

        return score

    print(f"\nGenerating recommendations: USER_KNN | TOP_K={TOP_K}\n")

    # Create and optimize the study
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    num_trials = 40
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # Train the final model with the best hyperparameters
    FINAL_model = UserKNN(
        train_file=FINAL_train_path,
        test_file=FINAL_test_path,
        output_file=FINAL_recs_output_path,
        sep=',',
        output_sep=',',
        k_neighbors=best_params_dict["k_neighbors"],
        similarity_metric=best_params_dict["similarity_metric"],
        rank_length=TOP_K
    )

    FINAL_model.compute(verbose=True)

    # -----------------------------------------------------------
    # Evaluate on the test set and save the metrics
    # -----------------------------------------------------------
    metrics_dict = ItemRecommendationEvaluation(
        sep=",", 
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save the best parameters
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0



# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------



def default_item_knn_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
):
    """
    Trains a default ItemKNN model and evaluates it on the test set.
    """

    # -----------------------------------------------------------
    # Load training data
    # -----------------------------------------------------------
    train_df = pd.read_csv(
        FINAL_train_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )

    train_df = train_df[["userID", "itemID", "rating"]]

    # -----------------------------------------------------------
    # Default hyperparameters
    # -----------------------------------------------------------
    sim_metric = "cosine"
    num_unique_items = train_df["itemID"].nunique()
    k_neigh = int(num_unique_items ** 0.5)

    params_dict = {
        "sim_metric": sim_metric,
        "k_neighbors": k_neigh
    }

    # -----------------------------------------------------------
    # Model builder
    # -----------------------------------------------------------
    def build_itemknn_model(output_file):
        """
        Creates an ItemKNN model with default parameters.
        """
        return ItemKNN(
            train_file=FINAL_train_path,
            test_file=FINAL_test_path,
            output_file=output_file,
            sep=",",
            output_sep=",",
            rank_length=TOP_K
        )

    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")
    print_params(params_dict)

    # -----------------------------------------------------------
    # Train final model
    # -----------------------------------------------------------
    final_model = build_itemknn_model(FINAL_recs_output_path)
    final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(
        FINAL_recs_output_path,
        FINAL_test_path
    )

    pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    ).to_csv(FINAL_metrics_output_path, index=False)

    # -----------------------------------------------------------
    # Save parameters
    # -----------------------------------------------------------
    pd.DataFrame(
        list(params_dict.items()),
        columns=["parameter", "value"]
    ).to_csv(FINAL_parameters_output_path, index=False)

    return 0










def optimized_item_knn_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
):
    """
    Optimizes an ItemKNN model using Optuna and evaluates the final model
    on the test set using the best hyperparameters.
    """

    OPT_DIR = "../datasets/recommender_train_validation"

    # Temporary file for Optuna trials
    OPT_recs_output_path = "utils/user_item_knn/item_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    # -----------------------------------------------------------
    # Validation evaluation for Optuna
    # -----------------------------------------------------------
    def evaluate_item_knn(k, sim_metric):

        OPT_train_path = f"{OPT_DIR}/opt_train.csv"
        OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

        delete_file(OPT_recs_output_path)

        model = ItemKNN(
            train_file=OPT_train_path,
            test_file=OPT_validation_path,
            output_file=OPT_recs_output_path,
            sep=",",
            output_sep=",",
            k_neighbors=k,
            similarity_metric=sim_metric,
            rank_length=TOP_K
        )

        model.compute(verbose=False)

        metrics_dict = ItemRecommendationEvaluation(
            sep=",",
            n_ranks=[TOP_K]
        ).evaluate_with_files(
            OPT_recs_output_path,
            OPT_validation_path
        )

        return metrics_dict[metric_key]

    # -----------------------------------------------------------
    # Optuna objective
    # -----------------------------------------------------------
    def objective(trial):

        k_neighbors = trial.suggest_int("k_neighbors", 1, 120)
        similarity_metric = trial.suggest_categorical(
            "similarity_metric",
            ["jaccard", "cosine"]
        )

        return evaluate_item_knn(k_neighbors, similarity_metric)

    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")

    # -----------------------------------------------------------
    # Hyperparameter optimization
    # -----------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    num_trials = 50
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # -----------------------------------------------------------
    # Final training with best hyperparameters
    # -----------------------------------------------------------
    final_model = ItemKNN(
        train_file=FINAL_train_path,
        test_file=FINAL_test_path,
        output_file=FINAL_recs_output_path,
        sep=",",
        output_sep=",",
        k_neighbors=best_params_dict["k_neighbors"],
        similarity_metric=best_params_dict["similarity_metric"],
        rank_length=TOP_K
    )

    final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # Final evaluation and output
    # -----------------------------------------------------------
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(
        FINAL_recs_output_path,
        FINAL_test_path
    )

    pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    ).to_csv(FINAL_metrics_output_path, index=False)

    pd.DataFrame(
        list(best_params_dict.items()),
        columns=["parameter", "value"]
    ).to_csv(FINAL_parameters_output_path, index=False)

    return 0





# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------





def default_ncf_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
):
    """
    Trains and evaluates a Neural Collaborative Filtering (NCF) model
    using fixed (default) hyperparameters.

    The model is trained on the full training set and evaluated on the
    test set. Recommendations, metrics, and parameters are saved to disk.
    """

    # -----------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------
    ncf_train = pd.read_csv(
        FINAL_train_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )
    ncf_test = pd.read_csv(
        FINAL_test_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )

    # Sort and ensure consistent data types
    for df in [ncf_train, ncf_test]:
        df.sort_values(by=["userID", "itemID"], inplace=True)
        df["userID"] = df["userID"].astype(int)
        df["itemID"] = df["itemID"].astype(int)

    # Ensure test users and items exist in training data
    ncf_test = ncf_test[
        ncf_test["userID"].isin(ncf_train["userID"]) &
        ncf_test["itemID"].isin(ncf_train["itemID"])
    ]

    # Remove timestamp column
    ncf_train = ncf_train[["userID", "itemID", "rating"]]
    ncf_test = ncf_test[["userID", "itemID", "rating"]]

    # -----------------------------------------------------------
    # Temporary datasets for NCF
    # -----------------------------------------------------------
    temp_path = "utils/ncf/ncf_parcial_datasets"

    train_temp_path = f"{temp_path}/train_ncf.csv"
    test_temp_path = f"{temp_path}/test_ncf.csv"

    ncf_train.to_csv(train_temp_path, index=False)
    ncf_test.to_csv(test_temp_path, index=False)

    # -----------------------------------------------------------
    # Dataset and default hyperparameters
    # -----------------------------------------------------------
    data_final = NCFDataset(
        train_file=train_temp_path,
        test_file=test_temp_path,
        seed=SEED
    )

    n_factors = 4
    layer_sizes = [16, 8, 4]
    n_epochs = 50
    batch_size = 256
    learning_rate = 1e-3

    params_dict = {
        "n_factors": n_factors,
        "layer_sizes": layer_sizes,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    print(f"\nGenerating recommendations: NCF (K={TOP_K})\n")
    print_params(params_dict)

    # -----------------------------------------------------------
    # Model training
    # -----------------------------------------------------------
    final_model = NCF(
        n_users=data_final.n_users,
        n_items=data_final.n_items,
        model_type="NeuMF",
        n_factors=n_factors,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=SEED
    )

    final_model.fit(data_final)

    # -----------------------------------------------------------
    # Generate Top-K recommendations
    # -----------------------------------------------------------
    with Timer() as test_time:

        users, items, preds = [], [], []
        item_list = list(ncf_train.itemID.unique())

        for user in ncf_train.userID.unique():
            users.extend([user] * len(item_list))
            items.extend(item_list)
            preds.extend(final_model.predict(
                [user] * len(item_list),
                item_list,
                is_list=True
            ))

        predictions = pd.DataFrame({
            "userID": users,
            "itemID": items,
            "prediction": preds
        })

        merged = pd.merge(
            ncf_train,
            predictions,
            on=["userID", "itemID"],
            how="outer"
        )

        predictions = merged[merged.rating.isnull()].drop(columns=["rating"])

        topk_predictions = (
            predictions
            .sort_values(by=["userID", "prediction"], ascending=[True, False])
            .groupby("userID")
            .head(TOP_K)
            .reset_index(drop=True)
        )

    print(f"Took {test_time.interval} seconds for prediction.")

    topk_predictions.to_csv(FINAL_recs_output_path, index=False, header=False)

    # -----------------------------------------------------------
    # Final evaluation and output
    # -----------------------------------------------------------
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    ).to_csv(FINAL_metrics_output_path, index=False)

    pd.DataFrame(
        list(params_dict.items()),
        columns=["parameter", "value"]
    ).to_csv(FINAL_parameters_output_path, index=False)

    return
















def optimized_ncf_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
):
    """
    Performs hyperparameter optimization for Neural Collaborative Filtering (NCF)
    using Optuna, then trains a final model and evaluates it on the test set.

    The optimization is carried out on a train/validation split, while the final
    model is trained on the full training set.
    """

    # -----------------------------------------------------------
    # Paths and configuration
    # -----------------------------------------------------------
    OPT_DIR = "../datasets/recommender_train_validation"

    OPT_train_path = f"{OPT_DIR}/opt_train.csv"
    OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

    OPT_recs_output_path = "utils/ncf/ncf_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    # -----------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------
    final_ncf_train = pd.read_csv(
        FINAL_train_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )
    final_ncf_test = pd.read_csv(
        FINAL_test_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )

    opt_ncf_train = pd.read_csv(
        OPT_train_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )
    opt_ncf_validation = pd.read_csv(
        OPT_validation_path,
        names=["userID", "itemID", "rating", "timestamp"]
    )

    # Sort for consistency
    for df in [final_ncf_train, final_ncf_test, opt_ncf_train, opt_ncf_validation]:
        df.sort_values(by=["userID", "itemID"], inplace=True)
        df["userID"] = df["userID"].astype(int)
        df["itemID"] = df["itemID"].astype(int)

    # Ensure test/validation users and items exist in train
    final_ncf_test = final_ncf_test[
        final_ncf_test["userID"].isin(final_ncf_train["userID"]) &
        final_ncf_test["itemID"].isin(final_ncf_train["itemID"])
    ]

    opt_ncf_validation = opt_ncf_validation[
        opt_ncf_validation["userID"].isin(opt_ncf_train["userID"]) &
        opt_ncf_validation["itemID"].isin(opt_ncf_train["itemID"])
    ]

    # Remove timestamp column
    final_ncf_train = final_ncf_train[["userID", "itemID", "rating"]]
    final_ncf_test = final_ncf_test[["userID", "itemID", "rating"]]
    opt_ncf_train = opt_ncf_train[["userID", "itemID", "rating"]]
    opt_ncf_validation = opt_ncf_validation[["userID", "itemID", "rating"]]

    # -----------------------------------------------------------
    # Temporary datasets for NCF
    # -----------------------------------------------------------
    temp_path = "utils/ncf/ncf_parcial_datasets"

    final_train_temp = f"{temp_path}/final_train_ncf.csv"
    final_test_temp = f"{temp_path}/final_test_ncf.csv"
    opt_train_temp = f"{temp_path}/opt_train_ncf.csv"
    opt_validation_temp = f"{temp_path}/opt_validation_ncf.csv"
    no_header_validation_temp = f"{temp_path}/no_header_opt_validation_ncf.csv"

    # Clear previous files
    for path in [
        final_train_temp,
        final_test_temp,
        opt_train_temp,
        opt_validation_temp,
        no_header_validation_temp
    ]:
        delete_file(path)

    # Save temporary CSVs
    final_ncf_train.to_csv(final_train_temp, index=False)
    final_ncf_test.to_csv(final_test_temp, index=False)
    opt_ncf_train.to_csv(opt_train_temp, index=False)
    opt_ncf_validation.to_csv(opt_validation_temp, index=False)
    opt_ncf_validation.to_csv(no_header_validation_temp, index=False, header=False)

    # -----------------------------------------------------------
    # Dataset for hyperparameter optimization
    # -----------------------------------------------------------
    data_opt = NCFDataset(
        train_file=opt_train_temp,
        test_file=opt_validation_temp,
        seed=SEED
    )

    def evaluate_ncf(lr, epochs, batch_size, n_factors, layer_sizes):
        """
        Trains an NCF model and evaluates it on the validation set.
        """

        model = NCF(
            n_users=data_opt.n_users,
            n_items=data_opt.n_items,
            model_type="NeuMF",
            n_factors=n_factors,
            layer_sizes=layer_sizes,
            learning_rate=lr,
            n_epochs=epochs,
            batch_size=batch_size,
            seed=SEED
        )

        model.fit(data_opt)

        # Generate Top-K predictions
        users, items, preds = [], [], []
        item_list = list(opt_ncf_train.itemID.unique())

        for user in opt_ncf_train.userID.unique():
            users.extend([user] * len(item_list))
            items.extend(item_list)
            preds.extend(model.predict([user] * len(item_list), item_list, is_list=True))

        predictions = pd.DataFrame({
            "userID": users,
            "itemID": items,
            "prediction": preds
        })

        merged = pd.merge(opt_ncf_train, predictions, on=["userID", "itemID"], how="outer")
        predictions = merged[merged.rating.isnull()].drop(columns=["rating"])

        topk = (
            predictions
            .sort_values(by=["userID", "prediction"], ascending=[True, False])
            .groupby("userID")
            .head(TOP_K)
            .reset_index(drop=True)
        )

        topk.to_csv(OPT_recs_output_path, index=False, header=False)

        metrics = ItemRecommendationEvaluation(
            sep=",",
            n_ranks=[TOP_K]
        ).evaluate_with_files(OPT_recs_output_path, no_header_validation_temp)

        return metrics[metric_key]

    def objective(trial):
        """
        Optuna objective function.
        """

        return evaluate_ncf(
            lr=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            epochs=trial.suggest_int("epochs", 20, 100),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
            n_factors=trial.suggest_categorical("n_factors", [4, 8, 16, 32, 64]),
            layer_sizes=trial.suggest_categorical(
                "layer_size",
                [[64, 32, 16, 8], [32, 16, 8], [16, 8, 4]]
            )
        )

    # -----------------------------------------------------------
    # Hyperparameter optimization
    # -----------------------------------------------------------
    print(f"\nGenerating recommendations: NCF (K={TOP_K})\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    num_trials = 30
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials
    print_params(best_params_dict)

    # -----------------------------------------------------------
    # Final training
    # -----------------------------------------------------------
    data_final = NCFDataset(
        train_file=final_train_temp,
        test_file=final_test_temp,
        seed=SEED
    )

    final_model = NCF(
        n_users=data_final.n_users,
        n_items=data_final.n_items,
        model_type="NeuMF",
        n_factors=best_params_dict["n_factors"],
        layer_sizes=best_params_dict["layer_size"],
        learning_rate=best_params_dict["learning_rate"],
        n_epochs=best_params_dict["epochs"],
        batch_size=best_params_dict["batch_size"],
        seed=SEED
    )

    final_model.fit(data_final)

    # -----------------------------------------------------------
    # Final prediction and evaluation
    # -----------------------------------------------------------
    users, items, preds = [], [], []
    item_list = list(final_ncf_train.itemID.unique())

    for user in final_ncf_train.userID.unique():
        users.extend([user] * len(item_list))
        items.extend(item_list)
        preds.extend(final_model.predict([user] * len(item_list), item_list, is_list=True))

    predictions = pd.DataFrame({
        "userID": users,
        "itemID": items,
        "prediction": preds
    })

    merged = pd.merge(final_ncf_train, predictions, on=["userID", "itemID"], how="outer")
    predictions = merged[merged.rating.isnull()].drop(columns=["rating"])

    topk = (
        predictions
        .sort_values(by=["userID", "prediction"], ascending=[True, False])
        .groupby("userID")
        .head(TOP_K)
        .reset_index(drop=True)
    )

    topk.to_csv(FINAL_recs_output_path, index=False, header=False)

    metrics = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    pd.DataFrame(
        list(metrics.items()),
        columns=["metric", "value"]
    ).to_csv(FINAL_metrics_output_path, index=False)

    pd.DataFrame(
        list(best_params_dict.items()),
        columns=["parameter", "value"]
    ).to_csv(FINAL_parameters_output_path, index=False)

    return











def pretrain_gmf_mlp(data, gmf_dir, mlp_dir, GLOBAL_N_USERS, GLOBAL_N_ITEMS):

    reset_dir(gmf_dir)
    reset_dir(mlp_dir)

    print("Reset dirs")

    print("\nPretraining GMF")

    gmf_model = NCF(
        n_users=GLOBAL_N_USERS, 
        n_items=GLOBAL_N_ITEMS,
        model_type="GMF",
        n_factors=4,
        layer_sizes=[16,8,4],
        n_epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    with Timer() as train_time:
        gmf_model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    gmf_model.save(dir_name=gmf_dir)





    print("\nPretraining MLP")

    mlp_model = NCF(
        n_users=GLOBAL_N_USERS, 
        n_items=GLOBAL_N_ITEMS,
        model_type="MLP",
        n_factors=8,
        layer_sizes=[16,8,4],
        n_epochs=40,
        batch_size=256,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    

    with Timer() as train_time:
        mlp_model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    mlp_model.save(dir_name=mlp_dir)




# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------



def default_bprmf_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
) -> int:
    """
    Trains and evaluates a BPR-MF model using fixed (default) hyperparameters.

    The model is trained on the full training set and evaluated on the
    test set. Recommendations, metrics, and parameters are saved to disk.
    """

    # Default hyperparameters
    factors = 50
    learn_rate = 0.01
    epochs = 80

    params_dict = {
        "factors": factors,
        "learn_rate": learn_rate,
        "epochs": epochs
    }

    print(f"\nGenerating recommendations: BPR-MF (K={TOP_K})\n")
    print_params(params_dict)

    # -----------------------------------------------------------
    # Final training
    # -----------------------------------------------------------
    bprmf_final_model = BprMF(
        train_file=FINAL_train_path,
        test_file=FINAL_test_path,
        output_file=FINAL_recs_output_path,
        sep=",",
        output_sep=",",
        rank_length=TOP_K,
        factors=factors,
        learn_rate=learn_rate,
        epochs=epochs
    )

    bprmf_final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # Final evaluation on test set
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    )

    metrics_dict = evaluator.evaluate_with_files(
        FINAL_recs_output_path,
        FINAL_test_path
    )

    metrics_df = pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    )
    metrics_df.to_csv(FINAL_metrics_output_path, index=False)

    # Save hyperparameters
    params_df = pd.DataFrame(
        list(params_dict.items()),
        columns=["parameter", "value"]
    )
    params_df.to_csv(FINAL_parameters_output_path, index=False)

    return 0



    






def optimized_bprmf_recs(
    TOP_K,
    FINAL_train_path,
    FINAL_test_path,
    FINAL_recs_output_path,
    FINAL_metrics_output_path,
    FINAL_parameters_output_path
) -> int:
    """
    Performs hyperparameter optimization for BPR-MF using Optuna and
    generates final recommendations and evaluation metrics.

    The optimization is conducted on a train/validation split, and the
    best configuration is retrained on the full training set and evaluated
    on the test set.
    """

    # Directory containing train/validation split for optimization
    OPT_DIR = "../datasets/recommender_train_validation"

    # Temporary recommendation file used during optimization
    output_recs_opt_path = "utils/bprmf/bprmf_parcial_recs.csv"

    # Metric optimized by Optuna
    metric_key = f"NDCG@{TOP_K}"

    def evaluate_bprmf(n_factors, lr, epochs):
        """
        Trains and evaluates a BPR-MF model on the validation set.
        """

        opt_train_path = f"{OPT_DIR}/opt_train.csv"
        opt_validation_path = f"{OPT_DIR}/opt_validation.csv"

        # Ensure clean output before evaluation
        delete_file(output_recs_opt_path)

        bprmf_model = BprMF(
            train_file=opt_train_path,
            test_file=opt_validation_path,
            output_file=output_recs_opt_path,
            factors=n_factors,
            learn_rate=lr,
            epochs=epochs,
            sep=",",
            output_sep=",",
            rank_length=TOP_K
        )

        bprmf_model.compute(verbose=False)

        evaluator = ItemRecommendationEvaluation(
            sep=",",
            n_ranks=[TOP_K]
        )

        metrics_dict = evaluator.evaluate_with_files(
            output_recs_opt_path,
            opt_validation_path
        )

        return metrics_dict[metric_key]

    def objective(trial):
        """
        Optuna objective function defining the search space and score.
        """

        n_factors = trial.suggest_int("num_factors", 10, 200)
        lr = trial.suggest_float("learn_rate", 0.001, 0.05, log=True)
        epochs = trial.suggest_int("num_epochs", 20, 150)

        return evaluate_bprmf(n_factors, lr, epochs)

    print(f"\nGenerating recommendations: BPR-MF (K={TOP_K})\n")

    # -----------------------------------------------------------
    # Hyperparameter optimization (Optuna)
    # -----------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    num_trials = 30
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # -----------------------------------------------------------
    # Final training on full training set
    # -----------------------------------------------------------
    bprmf_final_model = BprMF(
        train_file=FINAL_train_path,
        test_file=FINAL_test_path,
        output_file=FINAL_recs_output_path,
        factors=best_params_dict["num_factors"],
        learn_rate=best_params_dict["learn_rate"],
        epochs=best_params_dict["num_epochs"],
        sep=",",
        output_sep=",",
        rank_length=TOP_K
    )

    bprmf_final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # Final evaluation on test set
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    )

    metrics_dict = evaluator.evaluate_with_files(
        FINAL_recs_output_path,
        FINAL_test_path
    )

    metrics_df = pd.DataFrame(
        list(metrics_dict.items()),
        columns=["metric", "value"]
    )
    metrics_df.to_csv(FINAL_metrics_output_path, index=False)

    # Save best hyperparameters
    params_df = pd.DataFrame(
        list(best_params_dict.items()),
        columns=["parameter", "value"]
    )
    params_df.to_csv(FINAL_parameters_output_path, index=False)

    return 0
