from recommenders.evaluation.python_evaluation import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    map_at_k
)


def calc_map_at_k(test_df, all_predictions, TOP_K):
    
    eval_map = map_at_k(
        test_df,
        all_predictions,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_map

def calc_ndcg_at_k(test_df, all_predictions, TOP_K):
    
    eval_ndcg = ndcg_at_k(
        test_df,
        all_predictions,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_ndcg


def calc_precision_at_k(test_df, all_predictions, TOP_K):
    
    eval_precision = precision_at_k(
        test_df,
        all_predictions,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_precision

def calc_recall_at_k(test_df, all_predictions, TOP_K):
    
    eval_recall = recall_at_k(
        test_df,
        all_predictions,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_prediction="prediction",
        k=TOP_K
    )

    return eval_recall
