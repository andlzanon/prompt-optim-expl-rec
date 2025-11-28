import pandas as pd
from pathlib import Path
from recommenders.utils.timer import Timer
from metrics import calc_map_at_k, calc_ndcg_at_k, calc_precision_at_k, calc_recall_at_k
from utils.print_aux import print_params
import optuna
from optuna.samplers import TPESampler
import shutil

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset

import warnings
warnings.filterwarnings("ignore", message=".*swapaxes.*")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR


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

#   - NCF
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb

#    - BprMF
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/bprmf.py


import pandas as pd
# Assuming UserKNN, ItemRecommendationEvaluation, and print_params are defined elsewhere

def default_user_knn_recs(TOP_K, train_path, test_path, recs_output_path, metrics_output_path, parameters_output_path):
    """
    Runs the default training and evaluation pipeline for a User-KNN recommendation model.

    This function trains a User-KNN model using fixed, common default hyperparameters 
    (Cosine similarity and k = sqrt(num_unique_users)), generates TOP_K recommendations 
    on the test set, and saves the evaluation metrics and used parameters.
    
    It intentionally skips hyperparameter optimization for simplicity and speed.

    Args:
        TOP_K (int): The number of recommendations to generate and evaluate (e.g., 10).
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the hold-out test dataset CSV file for evaluation.
        recs_output_path (str): File path where the final generated recommendations 
                                will be saved (CSV format).
        metrics_output_path (str): File path where the evaluation metrics 
                                   (e.g., NDCG@K, Recall@K) will be saved (CSV format).
        parameters_output_path (str): File path where the used default parameters 
                                      (k_neighbors and sim_metric) will be saved (CSV format).

    Returns:
        int: Always returns 0 (success status code).
    """

    # 1. Load Training Data
    train_df = pd.read_csv(train_path, names=["userID", "itemID", "rating", "timestamp"])

    # 2. Define Default Hyperparameters
    sim_metric = "cosine"
    
    # Calculate k_neighbors using the common heuristic: k ≈ sqrt(number of unique users)
    num_unique_users = train_df["userID"].nunique() 
    k_neigh = int(num_unique_users**0.5)

    # Store parameters in a dictionary for logging
    params_dict = {}
    params_dict["sim_metric"] = sim_metric
    params_dict["k_neighbors"] = k_neigh

    # 3. Helper Function to Build Model
    def build_userknn_model(output_file):
        """Creates a UserKNN instance with the specified parameters."""
        # Note: Assumes the UserKNN class consumes k_neighbors and sim_metric 
        # internally or uses these defaults by convention.
        return UserKNN(
            train_file=train_path,
            test_file=test_path,
            output_file=output_file,
            sep=',',
            output_sep=',',
            rank_length=TOP_K
        )

    # 4. Model Training and Recommendation Generation
    print(f"\nGenerating recommendations: USER_KNN | TOP_K={TOP_K}\n")

    # Display the parameters being used (assumes print_params is defined)
    print_params(params_dict)

    # Initialize and train the final model using the default hyperparameters
    final_model = build_userknn_model(
        recs_output_path
    )
    final_model.compute(verbose=True)

    # 5. Evaluation on Test Set
    # Instantiate the evaluation class
    evaluator = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    )
    # Perform evaluation by comparing generated recommendations with the ground truth test set
    metrics_dict = evaluator.evaluate_with_files(recs_output_path, test_path)

    # 6. Save Results and Parameters
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0




def optimized_user_knn_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):
    """
    Runs hyperparameter optimization and evaluation for a UserKNN recommendation model.

    The function performs the following steps:
    1. Defines a helper to build UserKNN models with varying parameters.
    2. Uses Optuna to search for the best hyperparameters (k_neighbors and similarity_metric)
       based on validation NDCG@TOP_K.
    3. Trains a final UserKNN model using the best discovered parameters.
    4. Generates recommendations on the test set.
    5. Computes test metrics and saves them to a CSV file.

    Args:
        TOP_K (int): Number of recommendations to generate and evaluate.
        train_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset used in hyperparameter tuning.
        test_path (str): Path to the hold-out test dataset.
        output_recs_path (str): File path where final recommendations will be saved.
        output_metrics_path (str): File path where evaluation metrics will be saved.

    Returns:
        int: Always returns 0 (success status code).
    """

    def build_userknn_model(k, sim_metric, test_file, output_file):
        """Creates a UserKNN instance with the specified parameters."""
        return UserKNN(
            train_file=train_path,
            test_file=test_file,
            output_file=output_file,
            sep=',',
            output_sep=',',
            k_neighbors=k,
            similarity_metric=sim_metric,
            rank_length=TOP_K
        )

    # Temporary output used during Optuna optimization
    output_recs_opt_path = f"utils/user_item_knn/user_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    print(f"\nGenerating recommendations: USER_KNN | TOP_K={TOP_K}\n")

    def objective(trial):
        """Objective function used by Optuna to maximize validation NDCG."""
        # Clear previous results so each trial outputs clean data
        if os.path.exists(output_recs_opt_path):
            os.remove(output_recs_opt_path)

        # Hyperparameter search space
        suggested_k_neighbors = trial.suggest_int("k_neighbors", 1, 80)
        suggested_similarity_metric = trial.suggest_categorical(
            "similarity_metric", ["jaccard", "cosine"]
        )

        # Build and run model with sampled parameters
        opt_model = build_userknn_model(
            suggested_k_neighbors,
            suggested_similarity_metric,
            validation_path,
            output_recs_opt_path
        )
        opt_model.compute(verbose=False)

        # Evaluate results on validation file
        metrics_dict = ItemRecommendationEvaluation(
            sep=",",
            n_ranks=[TOP_K]
        ).evaluate_with_files(output_recs_opt_path, validation_path)

        return metrics_dict[metric_key]

    # Create optimization study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    num_trials = 70
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # Train final model using the best hyperparameters found
    final_model = build_userknn_model(
        best_params_dict["k_neighbors"],
        best_params_dict["similarity_metric"],
        test_path,
        output_recs_path
    )
    final_model.compute(verbose=True)

    # Evaluate on test set
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(output_recs_path, test_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0




































def default_item_knn_recs(TOP_K, train_path, test_path, recs_output_path, metrics_output_path, parameters_output_path):
    """
    Runs hyperparameter optimization and evaluation for an ItemKNN recommendation model.

    The function performs the following workflow:
    1. Defines a helper function to build ItemKNN models with different hyperparameters.
    2. Uses Optuna to search for the best (k_neighbors, similarity_metric) combination
       based on validation NDCG@TOP_K.
    3. Trains a final ItemKNN model using the optimal parameters.
    4. Generates recommendations for the test set.
    5. Computes evaluation metrics and saves them to a CSV file.

    Args:
        TOP_K (int): Number of recommendations generated and evaluated.
        train_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset used in hyperparameter tuning.
        test_path (str): Path to the hold-out test dataset.
        output_recs_path (str): Path where final recommendations will be written.
        output_metrics_path (str): Path where test metrics will be written.

    Returns:
        int: Always returns 0 (success code).
    """

    train_df = pd.read_csv(train_path, names=[["userID", "itemID", "rating", "timestamp"]])

    sim_metric= "cosine"

    num_unique_items = train_df["itemID"].nunique() 
    k_neigh = int(num_unique_items**0.5)

    params_dict = {}
    params_dict["sim_metric"] = sim_metric
    params_dict["k_neighbors"] = k_neigh

    def build_item_model(output_file):
        """Creates an ItemKNN model instance with the specified parameters."""
        return ItemKNN(
            train_file=train_path,
            test_file=test_path,
            output_file=output_file,
            sep=',',
            output_sep=',',
            rank_length=TOP_K
        )

    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")

    print_params(params_dict)
    

    # Train final model with best hyperparameters
    final_model = build_item_model(
        recs_output_path
    )
    final_model.compute(verbose=True)

    # Compute and save final evaluation metrics
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(recs_output_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0





def optimized_item_knn_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):
    """
    Runs hyperparameter optimization and evaluation for an ItemKNN recommendation model.

    The function performs the following workflow:
    1. Defines a helper function to build ItemKNN models with different hyperparameters.
    2. Uses Optuna to search for the best (k_neighbors, similarity_metric) combination
       based on validation NDCG@TOP_K.
    3. Trains a final ItemKNN model using the optimal parameters.
    4. Generates recommendations for the test set.
    5. Computes evaluation metrics and saves them to a CSV file.

    Args:
        TOP_K (int): Number of recommendations generated and evaluated.
        train_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset used in hyperparameter tuning.
        test_path (str): Path to the hold-out test dataset.
        output_recs_path (str): Path where final recommendations will be written.
        output_metrics_path (str): Path where test metrics will be written.

    Returns:
        int: Always returns 0 (success code).
    """

    def build_item_model(k, sim_metric, test_file, output_file):
        """Creates an ItemKNN model instance with the specified parameters."""
        return ItemKNN(
            train_file=train_path,
            test_file=test_file,
            output_file=output_file,
            sep=',',
            output_sep=',',
            k_neighbors=k,
            similarity_metric=sim_metric,
            rank_length=TOP_K
        )

    # Temporary file used for Optuna trial outputs
    output_recs_opt_path = f"utils/user_item_knn/item_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")

    def objective(trial):
        """Objective function for Optuna hyperparameter optimization."""
        # Ensure no previous trial output interferes with evaluation
        if os.path.exists(output_recs_opt_path):
            os.remove(output_recs_opt_path)

        # Hyperparameter search space
        suggested_k_neighbors = trial.suggest_int("k_neighbors", 1, 80)
        suggested_similarity_metric = trial.suggest_categorical(
            "similarity_metric", ["cosine", "hamming", "jaccard", "euclidean", "dice"]
        )

        # Build and run the model using trial parameters
        opt_model = build_item_model(
            suggested_k_neighbors,
            suggested_similarity_metric,
            validation_path,
            output_recs_opt_path
        )
        opt_model.compute(verbose=False)

        # Evaluate results on validation data
        metrics_dict = ItemRecommendationEvaluation(
            sep=",",
            n_ranks=[TOP_K]
        ).evaluate_with_files(output_recs_opt_path, validation_path)

        return metrics_dict[metric_key]

    # Hyperparameter optimization loop
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=20)

    print("\nPARAMS USED:", study.best_params, "\n")

    best_params_dict = study.best_params

    # Train final model with best hyperparameters
    final_model = build_item_model(
        best_params_dict["k_neighbors"],
        best_params_dict["similarity_metric"],
        test_path,
        output_recs_path
    )
    final_model.compute(verbose=True)

    # Compute and save final evaluation metrics
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0





# def item_knn_recs(TOP_K, train_path, test_path, output_recs_path, output_metrics_path):

#     print()

#     print(f"Gerando recomendações: ITEM_KNN K={TOP_K}")

#     model = ItemKNN(
#         train_file=train_path,
#         test_file=test_path,
#         output_file=output_recs_path,
#         sep=',',
#         output_sep=',',
#         k_neighbors=TOP_K,
#         rank_length=TOP_K
#     )
#     model.compute(verbose=True)

#     metrics_dict = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K]).evaluate_with_files(output_recs_path, test_path)

#     metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

#     metrics_df.to_csv(output_metrics_path, sep=",", index=False)

#     return 0











































def default_ncf_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):
    """
    
    """

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    algorithm = "final ncf"
    print(f"\nRunning model: {algorithm.upper()}")

    # ncf_train: 80%       ncf_validation: 10%           ncf_test: 10%
    # using this method in order to do the hyperparameter tuning
    ncf_train = pd.read_csv(train_path, names=["userID", "itemID", "rating", "timestamp"])
    ncf_validation = pd.read_csv(validation_path, names=["userID", "itemID", "rating", "timestamp"])
    ncf_test = pd.read_csv(test_path, names=["userID", "itemID", "rating", "timestamp"])


    ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
    ncf_validation = ncf_validation.sort_values(by=["userID", "itemID"])
    ncf_test = ncf_test.sort_values(by=["userID", "itemID"])

    ncf_train['userID'] = ncf_train['userID'].astype(int)
    ncf_train['itemID'] = ncf_train['itemID'].astype(int)
    ncf_validation['userID'] = ncf_validation['userID'].astype(int)
    ncf_validation['itemID'] = ncf_validation['itemID'].astype(int)
    ncf_test['userID'] = ncf_test['userID'].astype(int)
    ncf_test['itemID'] = ncf_test['itemID'].astype(int)



    # Ensure test users/items exist in train
    ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
    ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]
    # Ensure validation users/items exist in train
    ncf_validation = ncf_validation[ncf_validation["userID"].isin(ncf_train["userID"].unique())]
    ncf_validation = ncf_validation[ncf_validation["itemID"].isin(ncf_train["itemID"].unique())]


    print(ncf_train.shape)
    print(ncf_validation.shape)
    print(ncf_test.shape)



    # Sort by timestamp and choose last INTERACTION per user
    leave_one_out_validation = (
        ncf_validation.sort_values(["userID", "timestamp"])
                    .groupby("userID")
                    .tail(1)
                    .reset_index(drop=True)
    )

    leave_one_out_test = (
        ncf_test.sort_values(["userID", "timestamp"])
                .groupby("userID")
                .tail(1)
                .reset_index(drop=True)
    )


    # remove timestamp
    ncf_train = ncf_train[["userID", "itemID", "rating"]]
    ncf_validation = ncf_validation[["userID", "itemID", "rating"]]
    ncf_test = ncf_test[["userID", "itemID", "rating"]]
    leave_one_out_validation = leave_one_out_validation[["userID", "itemID", "rating"]]
    leave_one_out_test = leave_one_out_test[["userID", "itemID", "rating"]]



    # Paths
    ncf_parcial_datasets_path = f"{parent_path}/utils/ncf/ncf_parcial_datasets"

    train_temp_path = f"{ncf_parcial_datasets_path}/train_ncf.csv"
    validation_temp_path = f"{ncf_parcial_datasets_path}/validation_ncf.csv"
    test_temp_path = f"{ncf_parcial_datasets_path}/test_ncf.csv"
    leave_one_out_validation_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_validation.csv"
    leave_one_out_test_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_test.csv"

    # Save temporary CSVs
    ncf_train.to_csv(train_temp_path, index=False)
    ncf_validation.to_csv(validation_temp_path, index=False)
    ncf_test.to_csv(test_temp_path, index=False)
    leave_one_out_validation.to_csv(leave_one_out_validation_temp_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


    data_opt_hyparam = NCFDataset(
        train_file=train_temp_path,
        test_file=leave_one_out_validation_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    gmf_dir = f"{parent_path}/utils/ncf/gmf_mlp_parameters/gmf"
    mlp_dir = f"{parent_path}/utils/ncf/gmf_mlp_parameters/mlp"

    pretrain_gmf_mlp(data_opt_hyparam, gmf_dir, mlp_dir)







### PARAMETERS OPTIMIZATION #############################################################

    def objective(trial):
        # SEARCH SPACE
        n_factors = 4
        layer_sizes = [16,8,4]
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 15, 70)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        alpha = trial.suggest_float("alpha", 0.0, 1.0)

        # BUILD MODEL
        model = NCF(
            n_users=data_opt_hyparam.n_users,
            n_items=data_opt_hyparam.n_items,
            model_type="NeuMF",
            n_factors=n_factors,
            layer_sizes=layer_sizes,
            learning_rate=lr,
            n_epochs=epochs,
            batch_size=batch_size,
            seed=SEED
        )
        
        # LOAD PRETRAINED GMF + MLP
        model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=alpha)

        # TRAIN NeuMF ONLY
        model.fit(data_opt_hyparam)


        with Timer() as test_time:

            users, items, preds = [], [], []
            item = list(ncf_train.itemID.unique())
            for user in ncf_train.userID.unique():
                user = [user] * len(item) 
                users.extend(user)
                items.extend(item)
                preds.extend(list(model.predict(user, item, is_list=True)))

            all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

            merged = pd.merge(ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
            all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

            topk_predictions = (
                all_predictions
                .sort_values(by=["userID", "prediction"], ascending=[True, False])
                .groupby("userID")
                .head(TOP_K)
                .reset_index(drop=True)
            )

        print("Took {} seconds for prediction.".format(test_time.interval))



        ndcg = calc_ndcg_at_k(ncf_validation, topk_predictions, TOP_K)

        return ndcg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    # study.optimize(objective, n_trials=2)

    print()
    print("PARAMS USED:", study.best_params)
    print()


    data_final = NCFDataset(
        train_file=train_temp_path,
        test_file=leave_one_out_test_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    
    best = study.best_params

    final_model = NCF(
        n_users=data_final.n_users,
        n_items=data_final.n_items,
        model_type="NeuMF",
        n_factors=4,
        layer_sizes=[16,8,4],
        learning_rate=best["learning_rate"],
        n_epochs=best["epochs"],
        batch_size=best["batch_size"],
        seed=SEED
    )

    final_model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=best["alpha"])
    final_model.fit(data_final)








    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------

    with Timer() as test_time:

        users, items, preds = [], [], []
        item = list(ncf_train.itemID.unique())
        for user in ncf_train.userID.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(final_model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

        merged = pd.merge(ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        topk_predictions = (
            all_predictions
            .sort_values(by=["userID", "prediction"], ascending=[True, False])
            .groupby("userID")
            .head(TOP_K)
            .reset_index(drop=True)
        )
        

    print("Took {} seconds for prediction.".format(test_time.interval))



    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    eval_map = calc_map_at_k(ncf_test, topk_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(ncf_test, topk_predictions, TOP_K)
    eval_precision = calc_precision_at_k(ncf_test, topk_predictions, TOP_K)
    eval_recall = calc_recall_at_k(ncf_test, topk_predictions, TOP_K)


    # ----------------------------------------------------
    # Save Results
    # ----------------------------------------------------
    metrics = {
        "MAP": eval_map,
        "NDCG": eval_ndcg,
        "Precision@K": eval_precision,
        "Recall@K": eval_recall,
    }

    results_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    topk_predictions.to_csv(
        output_recs_path,
        index=False
    )

    results_df.to_csv(
        output_metrics_path,
        index=False
    )

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}/{algorithm}_K={TOP_K}_recs.csv'\n")
    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}/{algorithm}_K={TOP_K}_metrics.csv'")

    return



















def optimized_ncf_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):
    """
    
    """

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    algorithm = "final ncf"
    print(f"\nRunning model: {algorithm.upper()}")

    # ncf_train: 80%       ncf_validation: 10%           ncf_test: 10%
    # using this method in order to do the hyperparameter tuning
    ncf_train = pd.read_csv(train_path, names=["userID", "itemID", "rating", "timestamp"])
    ncf_validation = pd.read_csv(validation_path, names=["userID", "itemID", "rating", "timestamp"])
    ncf_test = pd.read_csv(test_path, names=["userID", "itemID", "rating", "timestamp"])


    ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
    ncf_validation = ncf_validation.sort_values(by=["userID", "itemID"])
    ncf_test = ncf_test.sort_values(by=["userID", "itemID"])

    ncf_train['userID'] = ncf_train['userID'].astype(int)
    ncf_train['itemID'] = ncf_train['itemID'].astype(int)
    ncf_validation['userID'] = ncf_validation['userID'].astype(int)
    ncf_validation['itemID'] = ncf_validation['itemID'].astype(int)
    ncf_test['userID'] = ncf_test['userID'].astype(int)
    ncf_test['itemID'] = ncf_test['itemID'].astype(int)



    # Ensure test users/items exist in train
    ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
    ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]
    # Ensure validation users/items exist in train
    ncf_validation = ncf_validation[ncf_validation["userID"].isin(ncf_train["userID"].unique())]
    ncf_validation = ncf_validation[ncf_validation["itemID"].isin(ncf_train["itemID"].unique())]


    print(ncf_train.shape)
    print(ncf_validation.shape)
    print(ncf_test.shape)



    # Sort by timestamp and choose last INTERACTION per user
    leave_one_out_validation = (
        ncf_validation.sort_values(["userID", "timestamp"])
                    .groupby("userID")
                    .tail(1)
                    .reset_index(drop=True)
    )

    leave_one_out_test = (
        ncf_test.sort_values(["userID", "timestamp"])
                .groupby("userID")
                .tail(1)
                .reset_index(drop=True)
    )


    # remove timestamp
    ncf_train = ncf_train[["userID", "itemID", "rating"]]
    ncf_validation = ncf_validation[["userID", "itemID", "rating"]]
    ncf_test = ncf_test[["userID", "itemID", "rating"]]
    leave_one_out_validation = leave_one_out_validation[["userID", "itemID", "rating"]]
    leave_one_out_test = leave_one_out_test[["userID", "itemID", "rating"]]



    # Paths
    ncf_parcial_datasets_path = f"{parent_path}/utils/ncf/ncf_parcial_datasets"

    train_temp_path = f"{ncf_parcial_datasets_path}/train_ncf.csv"
    validation_temp_path = f"{ncf_parcial_datasets_path}/validation_ncf.csv"
    test_temp_path = f"{ncf_parcial_datasets_path}/test_ncf.csv"
    leave_one_out_validation_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_validation.csv"
    leave_one_out_test_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_test.csv"

    # Save temporary CSVs
    ncf_train.to_csv(train_temp_path, index=False)
    ncf_validation.to_csv(validation_temp_path, index=False)
    ncf_test.to_csv(test_temp_path, index=False)
    leave_one_out_validation.to_csv(leave_one_out_validation_temp_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


    data_opt_hyparam = NCFDataset(
        train_file=train_temp_path,
        test_file=leave_one_out_validation_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    gmf_dir = f"{parent_path}/utils/ncf/gmf_mlp_parameters/gmf"
    mlp_dir = f"{parent_path}/utils/ncf/gmf_mlp_parameters/mlp"

    pretrain_gmf_mlp(data_opt_hyparam, gmf_dir, mlp_dir)







### PARAMETERS OPTIMIZATION #############################################################

    def objective(trial):
        # SEARCH SPACE
        n_factors = 4
        layer_sizes = [16,8,4]
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 15, 70)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        alpha = trial.suggest_float("alpha", 0.0, 1.0)

        # BUILD MODEL
        model = NCF(
            n_users=data_opt_hyparam.n_users,
            n_items=data_opt_hyparam.n_items,
            model_type="NeuMF",
            n_factors=n_factors,
            layer_sizes=layer_sizes,
            learning_rate=lr,
            n_epochs=epochs,
            batch_size=batch_size,
            seed=SEED
        )
        
        # LOAD PRETRAINED GMF + MLP
        model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=alpha)

        # TRAIN NeuMF ONLY
        model.fit(data_opt_hyparam)


        with Timer() as test_time:

            users, items, preds = [], [], []
            item = list(ncf_train.itemID.unique())
            for user in ncf_train.userID.unique():
                user = [user] * len(item) 
                users.extend(user)
                items.extend(item)
                preds.extend(list(model.predict(user, item, is_list=True)))

            all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

            merged = pd.merge(ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
            all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

            topk_predictions = (
                all_predictions
                .sort_values(by=["userID", "prediction"], ascending=[True, False])
                .groupby("userID")
                .head(TOP_K)
                .reset_index(drop=True)
            )

        print("Took {} seconds for prediction.".format(test_time.interval))



        ndcg = calc_ndcg_at_k(ncf_validation, topk_predictions, TOP_K)

        return ndcg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    # study.optimize(objective, n_trials=2)

    print()
    print("PARAMS USED:", study.best_params)
    print()


    data_final = NCFDataset(
        train_file=train_temp_path,
        test_file=leave_one_out_test_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    
    best = study.best_params

    final_model = NCF(
        n_users=data_final.n_users,
        n_items=data_final.n_items,
        model_type="NeuMF",
        n_factors=4,
        layer_sizes=[16,8,4],
        learning_rate=best["learning_rate"],
        n_epochs=best["epochs"],
        batch_size=best["batch_size"],
        seed=SEED
    )

    final_model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=best["alpha"])
    final_model.fit(data_final)








    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------

    with Timer() as test_time:

        users, items, preds = [], [], []
        item = list(ncf_train.itemID.unique())
        for user in ncf_train.userID.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(final_model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

        merged = pd.merge(ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        topk_predictions = (
            all_predictions
            .sort_values(by=["userID", "prediction"], ascending=[True, False])
            .groupby("userID")
            .head(TOP_K)
            .reset_index(drop=True)
        )
        

    print("Took {} seconds for prediction.".format(test_time.interval))



    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    eval_map = calc_map_at_k(ncf_test, topk_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(ncf_test, topk_predictions, TOP_K)
    eval_precision = calc_precision_at_k(ncf_test, topk_predictions, TOP_K)
    eval_recall = calc_recall_at_k(ncf_test, topk_predictions, TOP_K)


    # ----------------------------------------------------
    # Save Results
    # ----------------------------------------------------
    metrics = {
        "MAP": eval_map,
        "NDCG": eval_ndcg,
        "Precision@K": eval_precision,
        "Recall@K": eval_recall,
    }

    results_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    topk_predictions.to_csv(
        output_recs_path,
        index=False
    )

    results_df.to_csv(
        output_metrics_path,
        index=False
    )

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}/{algorithm}_K={TOP_K}_recs.csv'\n")
    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}/{algorithm}_K={TOP_K}_metrics.csv'")

    return







def pretrain_gmf_mlp(data, gmf_dir, mlp_dir):

    def reset_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)  # delete folder
        os.makedirs(path)        # recreate empty folder

    reset_dir(gmf_dir)
    reset_dir(mlp_dir)

    print("Reset dirs")

    print("\nPretraining GMF")

    gmf_model = NCF(
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="GMF",
        n_factors=4,
        layer_sizes=[16,8,4],
        n_epochs=40,
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
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="MLP",
        n_factors=4,
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























# def ncf_recs(TOP_K, NUM_EPOCHS, BATCH_SIZE, train_path, test_path, output_recs_path, output_metrics_path):
#     """
    
#     """

#     # ----------------------------------------------------
#     # Setup
#     # ----------------------------------------------------
#     algorithm = "ncf"
#     print(f"\nRunning model: {algorithm.upper()}")

#     ncf_train = pd.read_csv(train_path, names=["userID", "itemID", "rating", "timestamp"])[["userID", "itemID", "rating"]]
#     ncf_test = pd.read_csv(test_path, names=["userID", "itemID", "rating", "timestamp"])[["userID", "itemID", "rating"]]

#     ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
#     ncf_test = ncf_test.sort_values(by=["userID", "itemID"])
#     ncf_train['userID'] = ncf_train['userID'].astype(int)
#     ncf_train['itemID'] = ncf_train['itemID'].astype(int)
#     ncf_test['userID'] = ncf_test['userID'].astype(int)
#     ncf_test['itemID'] = ncf_test['itemID'].astype(int)

#     # Ensure test users/items exist in train
#     ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
#     ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]

#     print(ncf_test.head())
#     print(ncf_train.head())


#     # Leave-one-out per user
#     leave_one_out_test = ncf_test.groupby("userID").last().reset_index()

#     # Paths
#     ncf_parcial_path = f"{current_path}/utils/ncf_parcial_datasets"

#     train_temp_path = f"{ncf_parcial_path}/train_ncf.csv"
#     test_temp_path = f"{ncf_parcial_path}/test_ncf.csv"
#     leave_one_out_test_temp_path = f"{ncf_parcial_path}/leave_one_out_test.csv"



#     # Save temporary CSVs
#     ncf_train.to_csv(train_temp_path, index=False)
#     ncf_test.to_csv(test_temp_path, index=False)
#     leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


#     data = NCFDataset(
#         train_file=train_temp_path,
#         test_file=leave_one_out_test_temp_path,
#         seed=SEED,
#         overwrite_test_file_full=True
#     )




#     # ----------------------------------------------------
#     # Initialize Model
#     # ----------------------------------------------------
    

#     ncf_model = NCF(
#         n_users=data.n_users, 
#         n_items=data.n_items,
#         model_type="NeuMF",
#         n_factors=4,
#         layer_sizes=[16,8,4],
#         n_epochs=NUM_EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=1e-3,
#         verbose=10,
#         seed=SEED
#     )



#     # ----------------------------------------------------
#     # Training
#     # ----------------------------------------------------
    

#     with Timer() as train_time:
#         ncf_model.fit(data)
#     print(f"✅ Training completed in {train_time.interval:.2f} seconds.\n")

#     # ----------------------------------------------------
#     # Generate Predictions
#     # ----------------------------------------------------

#     ncf_train.rename(columns={"userID": "userId", "itemID": "movieId"}, inplace=True)
#     ncf_test.rename(columns={"userID": "userId", "itemID": "movieId"}, inplace=True)

    

#     with Timer() as test_time:

#         users, items, preds = [], [], []
#         item = list(ncf_train.movieId.unique())
#         for user in ncf_train.userId.unique():
#             user = [user] * len(item) 
#             users.extend(user)
#             items.extend(item)
#             preds.extend(list(ncf_model.predict(user, item, is_list=True)))

#         all_predictions = pd.DataFrame(data={"userId": users, "movieId":items, "prediction":preds})

#         merged = pd.merge(ncf_train, all_predictions, on=["userId", "movieId"], how="outer")
#         all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

#     print("Took {} seconds for prediction.".format(test_time.interval))



#     # ----------------------------------------------------
#     # Evaluation
#     # ----------------------------------------------------
#     eval_map = calc_map_at_k(ncf_test, all_predictions, TOP_K)
#     eval_ndcg = calc_ndcg_at_k(ncf_test, all_predictions, TOP_K)
#     eval_precision = calc_precision_at_k(ncf_test, all_predictions, TOP_K)
#     eval_recall = calc_recall_at_k(ncf_test, all_predictions, TOP_K)


#     # ----------------------------------------------------
#     # Save Results
#     # ----------------------------------------------------
#     metrics = {
#         "MAP": eval_map,
#         "NDCG": eval_ndcg,
#         "Precision@K": eval_precision,
#         "Recall@K": eval_recall,
#     }

#     results_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

#     results_df.to_csv(
#         output_metrics_path,
#         index=False
#     )

#     all_predictions.to_csv(
#         output_recs_path,
#         index=False
#     )

#     print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}/{algorithm}_K={TOP_K}_metrics.csv'")
#     print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}/{algorithm}_K={TOP_K}_recs.csv'\n")

#     return






































def default_bprmf_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):

    print(f"\nGerando recomendações: BprMF K={TOP_K}\n")

    # -----------------------------------------------------------
    # FUNÇÃO OBJETIVO DO OPTUNA (train + validation)
    # -----------------------------------------------------------
    def objective(trial):

        # Busca de hiperparâmetros
        n_factors = trial.suggest_int("num_factors", 8, 200)
        epochs = trial.suggest_int("num_epochs", 5, 100)
        lr = trial.suggest_float("learn_rate", 1e-4, 1e-1, log=True)

        # Caminho para arquivo temporário por trial
        output_trial = f"temp_bpr_recs_trial_{trial.number}.csv"

        # Treina modelo BPR-MF
        model = BprMF(
            train_file=train_path,
            test_file=None,             # evita leakage
            output_file=output_trial,
            factors=n_factors,
            learn_rate=lr,
            epochs=epochs,
            sep=",",
            output_sep=",",
            rank_length=TOP_K
        )
        model.compute(verbose=False)

        # Avaliação no conjunto de validação
        evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
        metrics = evaluator.evaluate_with_files(output_trial, validation_path)

        # Remove arquivo temporário
        if os.path.exists(output_trial):
            os.remove(output_trial)

        # Retorna a métrica a ser maximizada
        return metrics["MAP"]

    # -----------------------------------------------------------
    # EXECUTA OTIMIZAÇÃO COM OPTUNA
    # -----------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    print("\nPARAMS USED:", study.best_params, "\n")
    best = study.best_params

    # -----------------------------------------------------------
    # TREINAMENTO FINAL NO TREINO + TEST
    # -----------------------------------------------------------
    final_model = BprMF(
        train_file=train_path,
        test_file=test_path,
        output_file=output_recs_path,
        factors=best["num_factors"],
        learn_rate=best["learn_rate"],
        epochs=best["num_epochs"],
        sep=",",
        output_sep=",",
        rank_length=TOP_K
    )
    final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # AVALIAÇÃO FINAL NO TEST SET
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
    metrics_dict = evaluator.evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0









def optimized_bprmf_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path, parameters_output_path):

    print(f"\nGerando recomendações: BprMF K={TOP_K}\n")

    # -----------------------------------------------------------
    # FUNÇÃO OBJETIVO DO OPTUNA (train + validation)
    # -----------------------------------------------------------
    def objective(trial):

        # Busca de hiperparâmetros
        n_factors = trial.suggest_int("num_factors", 8, 200)
        epochs = trial.suggest_int("num_epochs", 5, 100)
        lr = trial.suggest_float("learn_rate", 1e-4, 1e-1, log=True)

        # Caminho para arquivo temporário por trial
        output_trial = f"temp_bpr_recs_trial_{trial.number}.csv"

        # Treina modelo BPR-MF
        model = BprMF(
            train_file=train_path,
            test_file=None,             # evita leakage
            output_file=output_trial,
            factors=n_factors,
            learn_rate=lr,
            epochs=epochs,
            sep=",",
            output_sep=",",
            rank_length=TOP_K
        )
        model.compute(verbose=False)

        # Avaliação no conjunto de validação
        evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
        metrics = evaluator.evaluate_with_files(output_trial, validation_path)

        # Remove arquivo temporário
        if os.path.exists(output_trial):
            os.remove(output_trial)

        # Retorna a métrica a ser maximizada
        return metrics["MAP"]

    # -----------------------------------------------------------
    # EXECUTA OTIMIZAÇÃO COM OPTUNA
    # -----------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    print("\nPARAMS USED:", study.best_params, "\n")
    best = study.best_params

    # -----------------------------------------------------------
    # TREINAMENTO FINAL NO TREINO + TEST
    # -----------------------------------------------------------
    final_model = BprMF(
        train_file=train_path,
        test_file=test_path,
        output_file=output_recs_path,
        factors=best["num_factors"],
        learn_rate=best["learn_rate"],
        epochs=best["num_epochs"],
        sep=",",
        output_sep=",",
        rank_length=TOP_K
    )
    final_model.compute(verbose=True)

    # -----------------------------------------------------------
    # AVALIAÇÃO FINAL NO TEST SET
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
    metrics_dict = evaluator.evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

    return 0