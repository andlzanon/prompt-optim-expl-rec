import pandas as pd
from pathlib import Path
from recommenders.utils.timer import Timer
from metrics import calc_map_at_k, calc_ndcg_at_k, calc_precision_at_k, calc_recall_at_k

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



def ncf_recs(TOP_K, NUM_EPOCHS, BATCH_SIZE, train_path, test_path, output_recs_path, output_metrics_path):
    """
    
    """

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    algorithm = "ncf"
    print(f"\nRunning model: {algorithm.upper()}")

    ncf_train = pd.read_csv(train_path, names=["userID", "itemID", "rating", "timestamp"])[["userID", "itemID", "rating"]]
    ncf_test = pd.read_csv(test_path, names=["userID", "itemID", "rating", "timestamp"])[["userID", "itemID", "rating"]]

    ncf_train = ncf_train.sort_values(by=["userID", "itemID"])
    ncf_test = ncf_test.sort_values(by=["userID", "itemID"])
    ncf_train['userID'] = ncf_train['userID'].astype(int)
    ncf_train['itemID'] = ncf_train['itemID'].astype(int)
    ncf_test['userID'] = ncf_test['userID'].astype(int)
    ncf_test['itemID'] = ncf_test['itemID'].astype(int)

    # Ensure test users/items exist in train
    ncf_test = ncf_test[ncf_test["userID"].isin(ncf_train["userID"].unique())]
    ncf_test = ncf_test[ncf_test["itemID"].isin(ncf_train["itemID"].unique())]

    print(ncf_test.head())
    print(ncf_train.head())


    # Leave-one-out per user
    leave_one_out_test = ncf_test.groupby("userID").last().reset_index()

    # Paths
    ncf_parcial_path = f"{current_path}/utils/ncf_parcial_datasets"

    train_temp_path = f"{ncf_parcial_path}/train_ncf.csv"
    test_temp_path = f"{ncf_parcial_path}/test_ncf.csv"
    leave_one_out_test_temp_path = f"{ncf_parcial_path}/leave_one_out_test.csv"



    # Save temporary CSVs
    ncf_train.to_csv(train_temp_path, index=False)
    ncf_test.to_csv(test_temp_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_temp_path, index=False)


    data = NCFDataset(
        train_file=train_temp_path,
        test_file=leave_one_out_test_temp_path,
        seed=SEED,
        overwrite_test_file_full=True
    )




    # ----------------------------------------------------
    # Initialize Model
    # ----------------------------------------------------
    

    ncf_model = NCF(
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=4,
        layer_sizes=[16,8,4],
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )



    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------
    

    with Timer() as train_time:
        ncf_model.fit(data)
    print(f"✅ Training completed in {train_time.interval:.2f} seconds.\n")

    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------

    ncf_train.rename(columns={"userID": "userId", "itemID": "movieId"}, inplace=True)
    ncf_test.rename(columns={"userID": "userId", "itemID": "movieId"}, inplace=True)

    

    with Timer() as test_time:

        users, items, preds = [], [], []
        item = list(ncf_train.movieId.unique())
        for user in ncf_train.userId.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(ncf_model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userId": users, "movieId":items, "prediction":preds})

        merged = pd.merge(ncf_train, all_predictions, on=["userId", "movieId"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    print("Took {} seconds for prediction.".format(test_time.interval))



    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    eval_map = calc_map_at_k(ncf_test, all_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(ncf_test, all_predictions, TOP_K)
    eval_precision = calc_precision_at_k(ncf_test, all_predictions, TOP_K)
    eval_recall = calc_recall_at_k(ncf_test, all_predictions, TOP_K)


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

    results_df.to_csv(
        output_metrics_path,
        index=False
    )

    all_predictions.to_csv(
        output_recs_path,
        index=False
    )

    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}/{algorithm}_K={TOP_K}_metrics.csv'")
    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}/{algorithm}_K={TOP_K}_recs.csv'\n")

    return



def bprmf_recs(TOP_K, NUM_FACTORS, NUM_EPOCHS, LEARN_RATE, train_path, test_path, output_recs_path, output_metrics_path):
    print()

    print(f"Gerando recomendações: BprMF K={TOP_K}")

    model = BprMF(
        train_file=train_path,
        test_file=test_path,
        output_file=output_recs_path,
        factors=NUM_FACTORS,
        learn_rate=LEARN_RATE,
        epochs=NUM_EPOCHS,
        sep=',',
        output_sep=',',
        rank_length=TOP_K
    )
    model.compute(verbose=True)

    metrics_dict = ItemRecommendationEvaluation(sep=",", n_ranks=[TOP_K]).evaluate_with_files(output_recs_path, test_path)

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])

    metrics_df.to_csv(output_metrics_path, sep=",", index=False)

    return 0