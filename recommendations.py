import sys
import pandas as pd
import cornac

from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    get_top_k_items,
)

from surprise import SVD, Dataset, Reader
from recommenders.models.cornac.cornac_utils import predict_ranking
from cornac.models import BPR
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.surprise.surprise_utils import compute_ranking_predictions



def calc_map_at_k(test_df, all_predictions, TOP_K):
    
    eval_map = map_at_k(
        test_df,
        all_predictions,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_map

def calc_ndcg_at_k(test_df, all_predictions, TOP_K):
    
    eval_ndcg = ndcg_at_k(
        test_df,
        all_predictions,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_ndcg


def calc_precision_at_k(test_df, all_predictions, TOP_K):
    
    eval_precision = precision_at_k(
        test_df,
        all_predictions,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_precision

def calc_recall_at_k(test_df, all_predictions, TOP_K):
    
    eval_recall = recall_at_k(
        test_df,
        all_predictions,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_recall






# MODELS

#   - BPR
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb

#   - SVD
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb

#   - NCF
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb

#   - LightGCN
#       https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/lightgcn_deep_dive.ipynb


def do_recommendations(algorithm_name):

    # available algorithms:

        # bprmf
        # item knn


    # -----------------------------
    # System Info
    # -----------------------------
    print(f"System version: {sys.version}")
    print(f"Cornac version: {cornac.__version__}")



    # -----------------------------
    # Constants
    # -----------------------------
    TOP_K = 10
    NUM_FACTORS = 200
    NUM_EPOCHS = 100
    TRAIN_PATH = "datasets/train_test_oficial/train_llm_oficial.csv"
    TEST_PATH = "datasets/train_test_oficial/test_llm_oficial.csv"
    # output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_{algorithm_name}.csv"



    # -----------------------------
    # Load Dataset
    # -----------------------------
    train_df = pd.read_csv(TRAIN_PATH, sep="\t", names=["userId", "movieId", "rating"])
    test_df = pd.read_csv(TEST_PATH, sep="\t", names=["userId", "movieId", "rating"])


    match algorithm_name:
        case "bprmf":
            print(f"Running: {algorithm_name}")
            bprmf_recs(TOP_K, NUM_FACTORS, NUM_EPOCHS, train_df, test_df)

        case "svd":
            print(f"Running: {algorithm_name}")
            svd_recs(TOP_K, train_df, test_df)

        # case "item_knn":
        #     print(f"Started to train: {algorithm_name}")
        #     ItemKNN(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()

        # case "all":

        #     output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_bprmf.csv"

        #     print("RUNNING ALL: ")
        #     print(f"Started to train: bprmf")
        #     BprMF(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()
        #     print()

        #     output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_item_knn.csv"
        #     print(f"Started to train: item_knn")
        #     ItemKNN(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()
        #     print()

        #     print(f"\nRecs successfully saved!")
        #     print(f"Recs file_path: {datasets_path}/recommendation_files/")
        #     return

        case _:
            print("Algorithm not found!")
            print("PLEASE, ENTER A VALID ALGORITHM")
            return

    # print(f"\nRecs successfully saved!")
    # print(f"Recs file_path: {output_file_path}")
    return




def bprmf_recs(TOP_K, NUM_FACTORS, NUM_EPOCHS, train_df, test_df):
    """
    Train and evaluate a BPR-MF model using Cornac.

    Parameters
    ----------
    TOP_K : int
        Number of top recommendations to generate per user.
    NUM_FACTORS : int
        Number of latent factors.
    NUM_EPOCHS : int
        Number of training epochs.
    train_df : pd.DataFrame
        Training data with columns ['userId', 'movieId', 'rating'].
    test_df : pd.DataFrame
        Test data with columns ['userId', 'movieId', 'rating'].
    """

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    algorithm = "bpr"
    print(f"\nRunning model: {algorithm.upper()}")


    # ----------------------------------------------------
    # Initialize Model
    # ----------------------------------------------------
    bpr = BPR(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=SEED,
    )

    train_cornac = cornac.data.Dataset.from_uir(train_df.itertuples(index=False), seed=SEED)

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------
    with Timer() as t:
        bpr.fit(train_cornac)
    print(f"✅ Training completed in {t} seconds.\n")

    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------
    with Timer() as t:
        all_predictions = predict_ranking(
            model=bpr,
            data=train_df,
            usercol="userId",
            itemcol="movieId",
            remove_seen=True,
        )
    print(f"✅ Prediction completed in {t} seconds.\n")

    # Keep top-K predictions per user
    all_predictions = (
        all_predictions
        .sort_values(["userId", "prediction"], ascending=[True, False])
        .groupby("userId")
        .head(TOP_K)
        .reset_index(drop=True)
    )

    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    eval_map = calc_map_at_k(test_df, all_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(test_df, all_predictions, TOP_K)
    eval_precision = calc_precision_at_k(test_df, all_predictions, TOP_K)
    eval_recall = calc_recall_at_k(test_df, all_predictions, TOP_K)

    # ----------------------------------------------------
    # Display Results
    # ----------------------------------------------------
    print(
        f"Model Evaluation Results:",
        f"Top K:\t\t {TOP_K}",
        f"MAP:\t\t {eval_map:.4f}",
        f"NDCG:\t\t {eval_ndcg:.4f}",
        f"Precision@K:\t {eval_precision:.4f}",
        f"Recall@K:\t {eval_recall:.4f}",
        sep="\n"
    )

    # ----------------------------------------------------
    # Save Results
    # ----------------------------------------------------
    metrics = {
        "MAP@K": eval_map,
        "NDCG@K": eval_ndcg,
        "Precision@K": eval_precision,
        "Recall@K": eval_recall,
    }

    # Save metrics in long format (Metric, Value)
    results_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    results_df.to_csv(
        f"datasets/recommendation_files/recommendation_metrics/{algorithm}_metrics.csv",
        index=False
    )

    # Save recommendation list
    all_predictions.to_csv(
        f"datasets/recommendation_files/recommendation_lists/{algorithm}_recs.csv",
        index=False
    )

    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}_metrics.csv'")
    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}_recs.csv'\n")





# Assuming you already have:
# calc_map_at_k(), calc_ndcg_at_k(), calc_precision_at_k(), calc_recall_at_k()
# and compute_ranking_predictions() defined elsewhere

def svd_recs(TOP_K, train_df, test_df):
    """
    Train and evaluate an SVD model using Surprise.

    Parameters
    ----------
    TOP_K : int
        Number of top recommendations to generate per user.
    train_df : pd.DataFrame
        Training data with columns ['userId', 'movieId', 'rating'].
    test_df : pd.DataFrame
        Test data with columns ['userId', 'movieId', 'rating'].
    """

    # ----------------------------------------------------
    # Setup
    # ----------------------------------------------------
    algorithm = "svd"
    print(f"\nRunning model: {algorithm.upper()}")

    # ----------------------------------------------------
    # Prepare Surprise Dataset
    # ----------------------------------------------------
    reader = Reader(rating_scale=(train_df["rating"].min(), train_df["rating"].max()))
    train_surprise = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
    train_surprise = train_surprise.build_full_trainset()

    # ----------------------------------------------------
    # Initialize Model
    # ----------------------------------------------------
    svd = SVD(
        random_state=0,
        n_factors=200,
        n_epochs=30,
        verbose=True
    )

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------
    with Timer() as t_train:
        svd.fit(train_surprise)
    print(f"✅ Training completed in {t_train.interval:.2f} seconds.\n")

    # ----------------------------------------------------
    # Generate Predictions
    # ----------------------------------------------------
    with Timer() as t_pred:
        all_predictions = compute_ranking_predictions(
            svd,
            train_df,
            usercol="userId",
            itemcol="movieId",
            remove_seen=True
        )
    print(f"✅ Prediction completed in {t_pred.interval:.2f} seconds.\n")

    # Keep top-K predictions per user
    all_predictions = (
        all_predictions
        .sort_values(["userId", "prediction"], ascending=[True, False])
        .groupby("userId")
        .head(TOP_K)
        .reset_index(drop=True)
    )

    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    eval_map = calc_map_at_k(test_df, all_predictions, TOP_K)
    eval_ndcg = calc_ndcg_at_k(test_df, all_predictions, TOP_K)
    eval_precision = calc_precision_at_k(test_df, all_predictions, TOP_K)
    eval_recall = calc_recall_at_k(test_df, all_predictions, TOP_K)

    # ----------------------------------------------------
    # Display Results
    # ----------------------------------------------------
    print(
        f"Model Evaluation Results:",
        f"Top K:\t\t {TOP_K}",
        f"MAP:\t\t {eval_map:.4f}",
        f"NDCG:\t\t {eval_ndcg:.4f}",
        f"Precision@K:\t {eval_precision:.4f}",
        f"Recall@K:\t {eval_recall:.4f}",
        sep="\n"
    )

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
        f"datasets/recommendation_files/recommendation_metrics/{algorithm}_metrics.csv",
        index=False
    )

    all_predictions.to_csv(
        f"datasets/recommendation_files/recommendation_lists/{algorithm}_recs.csv",
        index=False
    )

    print(f"✅ Metrics saved to 'datasets/recommendation_files/recommendation_metrics/{algorithm}_metrics.csv'")
    print(f"✅ Predictions saved to 'datasets/recommendation_files/recommendation_lists/{algorithm}_recs.csv'\n")

    return



    
do_recommendations(algorithm_name="bprmf")
# do_recommendations(algorithm_name="svd")
# do_recommendations(algorithm_name="item_knn")
# do_recommendations(algorithm_name="all")