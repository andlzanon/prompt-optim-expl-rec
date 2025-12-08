import pandas as pd
import numpy as np
from pathlib import Path
from recommenders.utils.timer import Timer
from metrics import calc_map_at_k, calc_ndcg_at_k, calc_precision_at_k, calc_recall_at_k
from utils.dir_manipulation import delete_file, reset_dir
from utils.print_aux import print_params
import optuna
# import optuna.storages
from optuna.samplers import TPESampler
# import multiprocessing as mp

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

#    - BprMF
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/bprmf.py

#   - NCF
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb


def default_user_knn_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    # 1. Load Training Data
    train_df = pd.read_csv(FINAL_train_path, names=["userID", "itemID", "rating", "timestamp"])
    train_df = train_df[["userID", "itemID", "rating"]]

    # 2. Define Default Hyperparameters
    sim_metric = "cosine"
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
            train_file=FINAL_train_path,
            test_file=FINAL_test_path,
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
    FINAL_model = build_userknn_model(
        FINAL_recs_output_path
    )

    FINAL_model.compute(verbose=True)

    # 5. Evaluation on Test Set
    # Instantiate the evaluation class
    evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
    metrics_dict = evaluator.evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    # 6. Save Results and Parameters
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0













def optimized_user_knn_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    OPT_DIR = "../datasets/train_validation"

    # Temporary output used during Optuna optimization
    OPT_recs_output_path = f"utils/user_item_knn/user_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    def evaluate_user_knn(k, sim_metric):

        OPT_train_path = f"{OPT_DIR}/opt_train.csv"
        OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

        delete_file(OPT_recs_output_path)

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
        # Hyperparameter search space
        suggested_k_neighbors = trial.suggest_int("k_neighbors", 1, 100)
        suggested_similarity_metric = trial.suggest_categorical(
            "similarity_metric", ["jaccard", "cosine"]
        )

        score = evaluate_user_knn(suggested_k_neighbors, suggested_similarity_metric)

        return score





    print(f"\nGenerating recommendations: USER_KNN | TOP_K={TOP_K}\n")

    # Create optimization study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    num_trials = 40
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

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

    # Evaluate on test set
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0

















def default_item_knn_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    train_df = pd.read_csv(FINAL_train_path, names=["userID", "itemID", "rating", "timestamp"])
    train_df = train_df[["userID", "itemID", "rating"]]

    sim_metric= "cosine"
    num_unique_items = train_df["itemID"].nunique() 
    k_neigh = int(num_unique_items**0.5)

    params_dict = {}
    params_dict["sim_metric"] = sim_metric
    params_dict["k_neighbors"] = k_neigh

    def build_itemknn_model(output_file):
        """Creates an ItemKNN model instance with the specified parameters."""
        return ItemKNN(
            train_file=FINAL_train_path,
            test_file=FINAL_test_path,
            output_file=output_file,
            sep=',',
            output_sep=',',
            rank_length=TOP_K
        )

    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")

    print_params(params_dict)
    

    # Train final model with best hyperparameters
    FINAL_model = build_itemknn_model(
        FINAL_recs_output_path
    )

    FINAL_model.compute(verbose=True)

    # Compute and save final evaluation metrics
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0















def optimized_item_knn_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    OPT_DIR = "../datasets/train_validation"

    # Temporary file used for Optuna trial outputs
    OPT_recs_output_path = f"utils/user_item_knn/item_knn_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"

    def evaluate_item_knn(k, sim_metric):

        OPT_train_path = f"{OPT_DIR}/opt_train.csv"
        OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

        delete_file(OPT_recs_output_path)

        OPT_user_knn_model = ItemKNN(
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
        # Hyperparameter search space
        suggested_k_neighbors = trial.suggest_int("k_neighbors", 1, 120)
        suggested_similarity_metric = trial.suggest_categorical(
            "similarity_metric", ["jaccard", "cosine"]
        )

        score = evaluate_item_knn(suggested_k_neighbors, suggested_similarity_metric)

        return score




    print(f"\nGenerating recommendations: ITEM_KNN | TOP_K={TOP_K}\n")

    # Hyperparameter optimization loop
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    num_trials = 50
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # Train final model with best hyperparameters
    FINAL_item_knn_model = ItemKNN(
        train_file=FINAL_train_path,
        test_file=FINAL_test_path,
        output_file=FINAL_recs_output_path,
        sep=',',
        output_sep=',',
        k_neighbors=best_params_dict["k_neighbors"],
        similarity_metric=best_params_dict["similarity_metric"],
        rank_length=TOP_K
    )

    FINAL_item_knn_model.compute(verbose=True)

    # Compute and save final evaluation metrics
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0









































# def default_ncf_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

#     ncf_train = pd.read_csv(FINAL_train_path, names=["userID", "itemID", "rating", "timestamp"])
#     ncf_test = pd.read_csv(FINAL_test_path, names=["userID", "itemID", "rating", "timestamp"])


#     ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
#     ncf_test = ncf_test.sort_values(by=["userID", "itemID"])

#     ncf_train['userID'] = ncf_train['userID'].astype(int)
#     ncf_train['itemID'] = ncf_train['itemID'].astype(int)
#     ncf_test['userID'] = ncf_test['userID'].astype(int)
#     ncf_test['itemID'] = ncf_test['itemID'].astype(int)



#     # Ensure test users/items exist in train
#     ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
#     ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]


#     print(ncf_train.shape)
#     print(ncf_test.shape)

#     # -------------------------------------------------------------------------
#     # 3. UNIFIED ID MAPPING (Crucial Step)
#     # -------------------------------------------------------------------------
    
#     all_user_ids = pd.concat([ncf_train["userID"], ncf_test["userID"]]).unique()
#     all_item_ids = pd.concat([ncf_train["itemID"], ncf_test["itemID"]]).unique()

#     user_map = {id: i for i, id in enumerate(all_user_ids)}
#     item_map = {id: i for i, id in enumerate(all_item_ids)}

#     # Apply mapping to Train
#     ncf_train['userID'] = ncf_train['userID'].map(user_map).astype(int)
#     ncf_train['itemID'] = ncf_train['itemID'].map(item_map).astype(int)

#     # Apply mapping to Test
#     ncf_test['userID'] = ncf_test['userID'].map(user_map).astype(int)
#     ncf_test['itemID'] = ncf_test['itemID'].map(item_map).astype(int)

#     leave_one_out_test = (
#         ncf_test.sort_values(["userID", "timestamp"])
#                 .groupby("userID")
#                 .tail(1)
#                 .reset_index(drop=True)
#     )


#     # remove timestamp
#     ncf_train = ncf_train[["userID", "itemID", "rating"]]
#     ncf_test = ncf_test[["userID", "itemID", "rating"]]
#     leave_one_out_test = leave_one_out_test[["userID", "itemID", "rating"]]



#     # Paths
#     ncf_parcial_datasets_path = f"utils/ncf/ncf_parcial_datasets"

#     train_temp_path = f"{ncf_parcial_datasets_path}/train_ncf.csv"
#     test_temp_path = f"{ncf_parcial_datasets_path}/test_ncf.csv"
#     leave_one_out_test_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_test.csv"

#     # Save temporary CSVs
#     ncf_train.to_csv(train_temp_path, index=False)
#     ncf_test.to_csv(test_temp_path, index=False)
#     leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


#     gmf_dir = f"utils/ncf/gmf_mlp_parameters/gmf"
#     mlp_dir = f"utils/ncf/gmf_mlp_parameters/mlp"

#     data_final = NCFDataset(
#         train_file=train_temp_path,
#         test_file=leave_one_out_test_temp_path,
#         seed=SEED,
#         overwrite_test_file_full=True
#     )

#     GLOBAL_N_USERS = len(user_map)
#     GLOBAL_N_ITEMS = len(item_map)

#     pretrain_gmf_mlp(data_final, gmf_dir, mlp_dir, GLOBAL_N_USERS, GLOBAL_N_ITEMS)

#     n_factors=8
#     layer_sizes=[16, 8, 4]
#     n_epochs=50
#     batch_size=256
#     learning_rate=5e-3
#     alpha = 0.5

#     params_dict = {}
#     params_dict["n_factors"] = n_factors
#     params_dict["layer_sizes"] = layer_sizes
#     params_dict["n_epochs"] = n_epochs
#     params_dict["batch_size"] = batch_size
#     params_dict["learning_rate"] = learning_rate
#     params_dict["alpha"] = alpha

#     final_model = NCF(
#         n_users=data_final.n_users,
#         n_items=data_final.n_items,
#         model_type="NeuMF",
#         n_factors=n_factors,
#         layer_sizes=layer_sizes,
#         learning_rate=learning_rate,
#         n_epochs=n_epochs,
#         batch_size=batch_size,
#         seed=SEED
#     )

#     final_model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=alpha)
#     final_model.fit(data_final)

#     print(f"\nGenerating recommendations: NCF | TOP_K={TOP_K}\n")

#     print_params(params_dict)

#     # ----------------------------------------------------
#     # Generate Predictions
#     # ----------------------------------------------------

#     with Timer() as test_time:

#         users, items, preds = [], [], []
#         item = list(ncf_train.itemID.unique())
#         for user in ncf_train.userID.unique():
#             user = [user] * len(item) 
#             users.extend(user)
#             items.extend(item)
#             preds.extend(list(final_model.predict(user, item, is_list=True)))

#         all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

#         merged = pd.merge(ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
#         all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

#         topk_predictions = (
#             all_predictions
#             .sort_values(by=["userID", "prediction"], ascending=[True, False])
#             .groupby("userID")
#             .head(TOP_K)
#             .reset_index(drop=True)
#         )
        

#     print("Took {} seconds for prediction.".format(test_time.interval))



#     # # ----------------------------------------------------
#     # # Evaluation
#     # # ----------------------------------------------------
#     # eval_map = calc_map_at_k(ncf_test, topk_predictions, TOP_K)
#     # eval_ndcg = calc_ndcg_at_k(ncf_test, topk_predictions, TOP_K)
#     # eval_precision = calc_precision_at_k(ncf_test, topk_predictions, TOP_K)
#     # eval_recall = calc_recall_at_k(ncf_test, topk_predictions, TOP_K)


#     # # ----------------------------------------------------
#     # # Save Results
#     # # ----------------------------------------------------
#     # metrics_dict = {
#     #     "Precision@K": eval_precision,
#     #     "Recall@K": eval_recall,
#     #     "NDCG@K": eval_ndcg,
#     #     "MAP@K": eval_map,
#     # }

#     # metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
#     # metrics_df.to_csv(metrics_output_path, index=False)

#     topk_predictions.to_csv(FINAL_recs_output_path, index=False, header=False)

#     # Compute and save final evaluation metrics
#     metrics_dict = ItemRecommendationEvaluation(
#         sep=",",
#         n_ranks=[TOP_K]
#     ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

#     metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
#     metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

#     # Save parameters as CSV
#     params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
#     params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

#     return


























def default_ncf_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    ncf_train = pd.read_csv(FINAL_train_path, names=["userID", "itemID", "rating", "timestamp"])
    ncf_test = pd.read_csv(FINAL_test_path, names=["userID", "itemID", "rating", "timestamp"])


    ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
    ncf_test = ncf_test.sort_values(by=["userID", "itemID"])

    # ncf_train['userID'] = ncf_train['userID'].astype(int)
    # ncf_train['itemID'] = ncf_train['itemID'].astype(int)
    # ncf_test['userID'] = ncf_test['userID'].astype(int)
    # ncf_test['itemID'] = ncf_test['itemID'].astype(int)



    # Ensure test users/items exist in train
    ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
    ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]


    print(f"shape do treino: {ncf_train.shape}")
    print(f"shape do teste: {ncf_test.shape}")

    # -------------------------------------------------------------------------
    # 3. UNIFIED ID MAPPING (Crucial Step)
    # -------------------------------------------------------------------------
    
    # all_user_ids = pd.concat([ncf_train["userID"], ncf_test["userID"]]).unique()
    # all_item_ids = pd.concat([ncf_train["itemID"], ncf_test["itemID"]]).unique()

    # user_map = {id: i for i, id in enumerate(all_user_ids)}
    # item_map = {id: i for i, id in enumerate(all_item_ids)}

    # # Apply mapping to Train
    # ncf_train['userID'] = ncf_train['userID'].map(user_map).astype(int)
    # ncf_train['itemID'] = ncf_train['itemID'].map(item_map).astype(int)

    # # Apply mapping to Test
    # ncf_test['userID'] = ncf_test['userID'].map(user_map).astype(int)
    # ncf_test['itemID'] = ncf_test['itemID'].map(item_map).astype(int)

    # leave_one_out_test = (
    #     ncf_test.sort_values(["userID", "timestamp"])
    #             .groupby("userID")
    #             .tail(1)
    #             .reset_index(drop=True)
    # )


    # # remove timestamp
    # ncf_train = ncf_train[["userID", "itemID", "rating"]]
    # ncf_test = ncf_test[["userID", "itemID", "rating"]]
    # leave_one_out_test = leave_one_out_test[["userID", "itemID", "rating"]]



    # Paths
    ncf_parcial_datasets_path = f"utils/ncf/ncf_parcial_datasets"

    train_temp_path = f"{ncf_parcial_datasets_path}/train_ncf.csv"
    test_temp_path = f"{ncf_parcial_datasets_path}/test_ncf.csv"
    # leave_one_out_test_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_test.csv"

    # Save temporary CSVs
    ncf_train.to_csv(train_temp_path, index=False)
    ncf_test.to_csv(test_temp_path, index=False)
    # leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


    gmf_dir = f"utils/ncf/gmf_mlp_parameters/gmf"
    mlp_dir = f"utils/ncf/gmf_mlp_parameters/mlp"

    data_final = NCFDataset(
        train_file=train_temp_path,
        test_file=test_temp_path,
        seed=SEED,
        # overwrite_test_file_full=True
    )

    # GLOBAL_N_USERS = len(user_map)
    # GLOBAL_N_ITEMS = len(item_map)

    # pretrain_gmf_mlp(data_final, gmf_dir, mlp_dir, GLOBAL_N_USERS, GLOBAL_N_ITEMS)

    n_factors=4
    layer_sizes=[16, 8, 4]
    n_epochs=50
    batch_size=256
    learning_rate=1e-3
    alpha = 0.5

    params_dict = {}
    params_dict["n_factors"] = n_factors
    params_dict["layer_sizes"] = layer_sizes
    params_dict["n_epochs"] = n_epochs
    params_dict["batch_size"] = batch_size
    params_dict["learning_rate"] = learning_rate
    params_dict["alpha"] = alpha

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

    # final_model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=alpha)
    print(f"\nGenerating recommendations: NCF | TOP_K={TOP_K}\n")
    final_model.fit(data_final)

    print_params(params_dict)

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

    topk_predictions.to_csv(FINAL_recs_output_path, index=False, header=False)

    # Compute and save final evaluation metrics
    metrics_dict = ItemRecommendationEvaluation(
        sep=",",
        n_ranks=[TOP_K]
    ).evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return



















def optimized_ncf_recs(TOP_K, FINAL_train_path, FINAL_test_path, output_recs_path, output_metrics_path, parameters_output_path):
    
    OPT_DIR = "../datasets/train_validation"
    OPT_train_path = f"{OPT_DIR}/opt_train.csv"
    OPT_validation_path = f"{OPT_DIR}/opt_validation.csv"

    # ncf_train: 80%       ncf_validation: 10%           ncf_test: 10%
    # using this method in order to do the hyperparameter tuning
    final_ncf_train = pd.read_csv(FINAL_train_path, names=["userID", "itemID", "rating", "timestamp"])
    final_ncf_test = pd.read_csv(FINAL_test_path, names=["userID", "itemID", "rating", "timestamp"])

    opt_ncf_train = pd.read_csv(OPT_train_path, names=["userID", "itemID", "rating", "timestamp"])
    opt_ncf_validation = pd.read_csv(OPT_validation_path, names=["userID", "itemID", "rating", "timestamp"])


    final_ncf_train = final_ncf_train.sort_values(by=["userID", "itemID"])
    final_ncf_test = final_ncf_test.sort_values(by=["userID", "itemID"])
    opt_ncf_train = opt_ncf_train.sort_values(by=["userID", "itemID"])
    opt_ncf_validation = opt_ncf_validation.sort_values(by=["userID", "itemID"])


    final_ncf_train['userID'] = final_ncf_train['userID'].astype(int)
    final_ncf_train['itemID'] = final_ncf_train['itemID'].astype(int)
    final_ncf_test['userID'] = final_ncf_test['userID'].astype(int)
    final_ncf_test['itemID'] = final_ncf_test['itemID'].astype(int)

    opt_ncf_train['userID'] = opt_ncf_train['userID'].astype(int)
    opt_ncf_train['itemID'] = opt_ncf_train['itemID'].astype(int)
    opt_ncf_validation['userID'] = opt_ncf_validation['userID'].astype(int)
    opt_ncf_validation['itemID'] = opt_ncf_validation['itemID'].astype(int)



    # Ensure test users/items exist in train
    final_ncf_test = final_ncf_test[final_ncf_test["userID"].isin(final_ncf_train["userID"].unique())]
    final_ncf_test = final_ncf_test[final_ncf_test["itemID"].isin(final_ncf_train["itemID"].unique())]
    # Ensure validation users/items exist in train
    opt_ncf_validation = opt_ncf_validation[opt_ncf_validation["userID"].isin(opt_ncf_train["userID"].unique())]
    opt_ncf_validation = opt_ncf_validation[opt_ncf_validation["itemID"].isin(opt_ncf_train["itemID"].unique())]


    print(final_ncf_train.shape)
    print(final_ncf_test.shape)

    print(opt_ncf_train.shape)
    print(opt_ncf_validation.shape)



    # Sort by timestamp and choose last INTERACTION per user
    leave_one_out_validation = (
        opt_ncf_validation.sort_values(["userID", "timestamp"])
                    .groupby("userID")
                    .tail(1)
                    .reset_index(drop=True)
    )

    leave_one_out_test = (
        final_ncf_test.sort_values(["userID", "timestamp"])
                .groupby("userID")
                .tail(1)
                .reset_index(drop=True)
    )


    all_user_ids = pd.concat([final_ncf_train["userID"], final_ncf_test["userID"], opt_ncf_train["userID"], opt_ncf_validation["userID"]]).unique()
    all_item_ids = pd.concat([final_ncf_train["itemID"], final_ncf_test["itemID"], opt_ncf_train["itemID"], opt_ncf_validation["itemID"]]).unique()

    user_map = {id: i for i, id in enumerate(all_user_ids)}
    item_map = {id: i for i, id in enumerate(all_item_ids)}

    dfs_to_map = [
        final_ncf_train, final_ncf_test, 
        opt_ncf_train, opt_ncf_validation, 
        leave_one_out_validation, leave_one_out_test
    ]

    for df in dfs_to_map:
        # Use .loc to avoid SettingWithCopyWarning
        df.loc[:, 'userID'] = df['userID'].map(user_map).astype(int)
        df.loc[:, 'itemID'] = df['itemID'].map(item_map).astype(int)
        
    # NOTE: Drop any rows where mapping failed (e.g., if a user/item appeared in test but not in the combined train sets)
    # The previous filtering should have handled this, but this is a safety check.
    final_ncf_train.dropna(subset=['userID', 'itemID'], inplace=True)
    final_ncf_test.dropna(subset=['userID', 'itemID'], inplace=True)
    opt_ncf_train.dropna(subset=['userID', 'itemID'], inplace=True)
    opt_ncf_validation.dropna(subset=['userID', 'itemID'], inplace=True)
    leave_one_out_validation.dropna(subset=['userID', 'itemID'], inplace=True)
    leave_one_out_test.dropna(subset=['userID', 'itemID'], inplace=True)
    # ... repeat for all other DFs if necessary

    # --- END OF NEW MAPPING CODE ---

    # ... (rest of the code - removing timestamp and saving CSVs)


    # remove timestamp
    final_ncf_train = final_ncf_train[["userID", "itemID", "rating"]]
    final_ncf_test = final_ncf_test[["userID", "itemID", "rating"]]
    opt_ncf_train = opt_ncf_train[["userID", "itemID", "rating"]]
    opt_ncf_validation = opt_ncf_validation[["userID", "itemID", "rating"]]
    leave_one_out_validation = leave_one_out_validation[["userID", "itemID", "rating"]]
    leave_one_out_test = leave_one_out_test[["userID", "itemID", "rating"]]



    # Paths
    ncf_parcial_datasets_path = f"utils/ncf/ncf_parcial_datasets"

    final_train_temp_path = f"{ncf_parcial_datasets_path}/final_train_ncf.csv"
    final_test_temp_path = f"{ncf_parcial_datasets_path}/final_test_ncf.csv"
    opt_train_temp_path = f"{ncf_parcial_datasets_path}/opt_train_ncf.csv"
    opt_validation_temp_path = f"{ncf_parcial_datasets_path}/opt_validation_ncf.csv"
    leave_one_out_validation_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_validation.csv"
    leave_one_out_test_temp_path = f"{ncf_parcial_datasets_path}/leave_one_out_test.csv"

    # Clear previous results so each trial outputs clean data
    delete_file(final_train_temp_path)
    delete_file(final_test_temp_path)
    delete_file(opt_train_temp_path)
    delete_file(opt_validation_temp_path)
    delete_file(leave_one_out_validation_temp_path)
    delete_file(leave_one_out_test_temp_path)

    # Save temporary CSVs
    final_ncf_train.to_csv(final_train_temp_path, index=False)
    final_ncf_test.to_csv(final_test_temp_path, index=False)
    opt_ncf_train.to_csv(opt_train_temp_path, index=False)
    opt_ncf_validation.to_csv(opt_validation_temp_path, index=False)
    leave_one_out_validation.to_csv(leave_one_out_validation_temp_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)



    data_opt_hyparam = NCFDataset(
        train_file=opt_train_temp_path,
        test_file=leave_one_out_validation_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    GLOBAL_N_USERS = len(user_map)
    GLOBAL_N_ITEMS = len(item_map)

    gmf_dir = f"utils/ncf/gmf_mlp_parameters/gmf"
    mlp_dir = f"utils/ncf/gmf_mlp_parameters/mlp"

    pretrain_gmf_mlp(data_opt_hyparam, gmf_dir, mlp_dir, GLOBAL_N_USERS, GLOBAL_N_ITEMS)



    def evaluate_ncf(lr, epochs, batch_size, alpha):

       # BUILD MODEL
        model = NCF(
            n_users=data_opt_hyparam.n_users,
            n_items=data_opt_hyparam.n_items,
            model_type="NeuMF",
            n_factors=8,
            layer_sizes=[16, 8, 4],
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
            item = list(opt_ncf_train.itemID.unique())
            for user in opt_ncf_train.userID.unique():
                user = [user] * len(item) 
                users.extend(user)
                items.extend(item)
                preds.extend(list(model.predict(user, item, is_list=True)))

            all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

            merged = pd.merge(opt_ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
            all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

            topk_predictions = (
                all_predictions
                .sort_values(by=["userID", "prediction"], ascending=[True, False])
                .groupby("userID")
                .head(TOP_K)
                .reset_index(drop=True)
            )

        print("Took {} seconds for prediction.".format(test_time.interval))



        ndcg = calc_ndcg_at_k(opt_ncf_validation, topk_predictions, TOP_K)

        return ndcg


    def objective(trial):
        # Hyperparameter search space
        suggested_lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        suggested_epochs = trial.suggest_int("epochs", 15, 70)
        suggested_batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        suggested_alpha = trial.suggest_float("alpha", 0.0, 1.0)

        score = evaluate_ncf(suggested_lr, suggested_epochs, suggested_batch_size, suggested_alpha)

        return score


    print(f"\nGenerating recommendations: NCF | TOP_K={TOP_K}\n")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    num_trials = 20
    study.optimize(objective, n_trials=num_trials)


    data_final = NCFDataset(
        train_file=final_train_temp_path,
        test_file=leave_one_out_test_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )

    
    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    final_model = NCF(
        n_users=data_final.n_users,
        n_items=data_final.n_items,
        model_type="NeuMF",
        n_factors=8,
        layer_sizes=[16,8,4],
        learning_rate=best_params_dict["learning_rate"],
        n_epochs=best_params_dict["epochs"],
        batch_size=best_params_dict["batch_size"],
        seed=SEED
    )

    final_model.load(gmf_dir=gmf_dir, mlp_dir=mlp_dir, alpha=best_params_dict["alpha"])
    final_model.fit(data_final)








    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------

    with Timer() as test_time:

        users, items, preds = [], [], []
        item = list(final_ncf_train.itemID.unique())
        for user in final_ncf_train.userID.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(final_model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

        merged = pd.merge(final_ncf_train, all_predictions, on=["userID", "itemID"], how="outer")
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
    eval_map = calc_map_at_k(final_ncf_test, topk_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(final_ncf_test, topk_predictions, TOP_K)
    eval_precision = calc_precision_at_k(final_ncf_test, topk_predictions, TOP_K)
    eval_recall = calc_recall_at_k(final_ncf_test, topk_predictions, TOP_K)


    # ----------------------------------------------------
    # Save Results
    # ----------------------------------------------------
    metrics_dict = {
        "Precision@K": eval_precision,
        "Recall@K": eval_recall,
        "NDCG@K": eval_ndcg,
        "MAP@K": eval_map,
    }

    topk_predictions.to_csv(output_recs_path, index=False, header=False)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    metrics_df.to_csv(output_metrics_path, index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(parameters_output_path, sep=",", index=False)

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
        n_factors=8,
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





















def default_bprmf_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):

    factors = 50
    learn_rate = 0.01
    epochs = 80

    params_dict = {
        "factors": factors,
        "learn_rate": learn_rate,
        "epochs": epochs
    }

    print(f"\nGenerating recommendations: bprmf | TOP_K={TOP_K}\n")
    print_params(params_dict)

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
    # AVALIAÇÃO FINAL NO TEST SET
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
    metrics_dict = evaluator.evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0









def optimized_bprmf_recs(TOP_K, FINAL_train_path, FINAL_test_path, FINAL_recs_output_path, FINAL_metrics_output_path, FINAL_parameters_output_path):


    OPT_DIR = "../datasets/train_validation"

    # Temporary output used during Optuna optimization
    output_recs_opt_path = f"utils/bprmf/bprmf_parcial_recs.csv"
    metric_key = f"NDCG@{TOP_K}"


    def evaluate_bprmf(n_factors, lr, epochs):

        opt_train_path = f"{OPT_DIR}/opt_train.csv"
        opt_validation_path = f"{OPT_DIR}/opt_validation.csv"

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

        evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
        metrics_dict = evaluator.evaluate_with_files(output_recs_opt_path, opt_validation_path)

        return metrics_dict[metric_key]


    def objective(trial):
        # Hyperparameter search space
        suggested_n_factors = trial.suggest_int("num_factors", 10, 200)
        suggested_lr = trial.suggest_float("learn_rate", 0.001, 0.05, log=True)
        suggested_epochs = trial.suggest_int("num_epochs", 20, 150)

        score = evaluate_bprmf(suggested_n_factors, suggested_lr, suggested_epochs)

        return score
    





    print(f"\nGerando recomendações: BprMF K={TOP_K}\n")

    # -----------------------------------------------------------
    # EXECUTA OTIMIZAÇÃO COM OPTUNA
    # -----------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    num_trials = 30
    study.optimize(objective, n_trials=num_trials)

    best_params_dict = study.best_params
    best_params_dict["trials"] = num_trials

    print_params(best_params_dict)

    # -----------------------------------------------------------
    # TREINAMENTO FINAL NO TREINO + TEST
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
    # AVALIAÇÃO FINAL NO TEST SET
    # -----------------------------------------------------------
    evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K])
    metrics_dict = evaluator.evaluate_with_files(FINAL_recs_output_path, FINAL_test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
    metrics_df.to_csv(FINAL_metrics_output_path, sep=",", index=False)

    # Save parameters as CSV
    params_df = pd.DataFrame(list(best_params_dict.items()), columns=["parameter", "value"])
    params_df.to_csv(FINAL_parameters_output_path, sep=",", index=False)

    return 0























def teste(FINAL_test_path):
    print("UMA VEZ")
    k_values = [1, 5, 10, 20, 50, 100, 200]
    for k in k_values:
        algorithm_name = "ncf"

        default_recs_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_lists/{algorithm_name}/params_default/K={k}/default_{algorithm_name}_K={k}_recs.csv"
        default_metrics_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_default/K={k}/default_{algorithm_name}_K={k}_metrics.csv"

        optimized_recs_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_lists/{algorithm_name}/params_optimized/K={k}/optimized_{algorithm_name}_K={k}_recs.csv"
        optimized_metrics_output_path = f"{parent_path}/datasets/recommendation_files/recommendation_metrics/{algorithm_name}/params_optimized/K={k}/optimized_{algorithm_name}_K={k}_metrics.csv"

        delete_file(default_metrics_output_path)
        delete_file(optimized_metrics_output_path)

        evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[k])
        metrics_dict = evaluator.evaluate_with_files(default_recs_output_path, FINAL_test_path)
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
        metrics_df.to_csv(default_metrics_output_path, sep=",", index=False)

        evaluator = ItemRecommendationEvaluation(sep=",", n_ranks=[k])
        metrics_dict = evaluator.evaluate_with_files(optimized_recs_output_path, FINAL_test_path)
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
        metrics_df.to_csv(optimized_metrics_output_path, sep=",", index=False)

        


    return 0