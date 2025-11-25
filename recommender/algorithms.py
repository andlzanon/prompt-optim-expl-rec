import pandas as pd
from pathlib import Path
from recommenders.utils.timer import Timer
from metrics import calc_map_at_k, calc_ndcg_at_k, calc_precision_at_k, calc_recall_at_k
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

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

#   - User_knn e Item_knn
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/userknn.py

#   - NCF
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb

#    - BprMF
#       https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/bprmf.py


def user_knn_recs(TOP_K, train_path, test_path, output_recs_path, output_metrics_path):

    print()

    print(f"Gerando recomendações: USER_KNN K={TOP_K}")

    model = UserKNN(
        train_file=train_path,
        test_file=test_path,
        output_file=output_recs_path,
        sep=',',
        output_sep=',',
        k_neighbors=TOP_K,
        rank_length=TOP_K
    )
    model.compute(verbose=True)

    metrics_dict = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K]).evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    return 0

def item_knn_recs(TOP_K, train_path, test_path, output_recs_path, output_metrics_path):

    print()

    print(f"Gerando recomendações: ITEM_KNN K={TOP_K}")

    model = ItemKNN(
        train_file=train_path,
        test_file=test_path,
        output_file=output_recs_path,
        sep=',',
        output_sep=',',
        k_neighbors=TOP_K,
        rank_length=TOP_K
    )
    model.compute(verbose=True)

    metrics_dict = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K]).evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    return 0



















































def final_ncf_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path):
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
    # pretrain_gmf_mlp(data_opt_hyparam, gmf_dir, mlp_dir)







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
    # study.optimize(objective, n_trials=20)
    study.optimize(objective, n_trials=2)

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

    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}/{algorithm}_K={TOP_K}_recs.csv'\n")
    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}/{algorithm}_K={TOP_K}_metrics.csv'")

    return







def pretrain_gmf_mlp(data, gmf_dir, mlp_dir):
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








































def bprmf_opt_recs(TOP_K, train_path, validation_path, test_path, output_recs_path, output_metrics_path):

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

    return 0






































# def bprmf_recs(TOP_K, NUM_FACTORS, NUM_EPOCHS, LEARN_RATE, train_path, test_path, output_recs_path, output_metrics_path):
#     print()

#     print(f"Gerando recomendações: BprMF K={TOP_K}")


#     def objective(trial):

#         # ⭐ Hyperparameters to search
#         n_factors = trial.suggest_int("num_factors", 8, 200)
#         epochs = trial.suggest_int("num_epochs", 5, 100)
#         lr = trial.suggest_float("learn_rate", 1e-4, 1e-1, log=True)

#         # Temporary output path (Optuna trial-specific)
#         output_recs_path = f"temp_bpr_recs_trial_{trial.number}.csv"

#         # Train BPR-MF
#         model = BprMF(
#             train_file=train_path,
#             test_file=test_path,
#             output_file=output_recs_path,
#             factors=n_factors,
#             learn_rate=lr,
#             epochs=epochs,
#             sep=",",
#             output_sep=",",
#             rank_length=TOP_K
#         )
#         model.compute(verbose=False)

#         # Evaluate using the same function you already have
#         metrics = ItemRecommendationEvaluation(
#             sep=",",
#             n_ranks=[TOP_K]
#         ).evaluate_with_files(output_recs_path, test_path)

#         # Choose metric to optimize (use your preferred one)
#         target_metric = metrics["NDCG"]

#         return target_metric
    
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=20)

#     print("Best params:", study.best_params)

#     best = study.best_params

#     final_bpr_model = BprMF(
#         train_file=train_path,
#         test_file=test_path,
#         output_file=output_recs_path,
#         factors=best['num_factors'],
#         learn_rate=best['learn_rate'],
#         epochs=best['num_epochs'],
#         sep=',',
#         output_sep=',',
#         rank_length=TOP_K
#     )
#     final_bpr_model.compute(verbose=True)

#     metrics_dict = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K]).evaluate_with_files(output_recs_path, test_path)

#     metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

#     metrics_df.to_csv(output_metrics_path, sep=",", index=False)

#     return 0