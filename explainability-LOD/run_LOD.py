from path_reordering import PathReordering
import pandas as pd


def run_explod(rec_algs=None, expl_alg='explod', n_explain=10):
    """
    Run explanation experiments for the movielens-latest-small
    :param rec_algs: list recommender algorithms that has recommendations to be explained
    :param n_explain: quantity of items to explain
    :param: expl_alg: explanation algorithm to run experiments.
    :return: users are displayed on console with interacted items, recommended items, semantic profile, reordered items
        and explanation paths for each recommended item
    """

    recs_files = []

    if rec_algs is not None and 'BPRMF' in rec_algs:
        recs_files.append("../datasets/recommendation_files/recommendation_lists/bprmf/params_optimized/K=20/optimized_bprmf_K=20_recs.csv")

    if rec_algs is not None and 'UserKNN' in rec_algs:
        recs_files.append("../datasets/recommendation_files/recommendation_lists/user_knn/params_optimized/K=20/optimized_user_knn_K=20_recs.csv")

    if rec_algs is not None and 'ItemKNN' in rec_algs:
        recs_files.append("../datasets/recommendation_files/recommendation_lists/item_knn/params_optimized/K=20/optimized_item_knn_K=20_recs.csv")

    if rec_algs is not None and 'NCF' in rec_algs:
        recs_files.append("../datasets/recommendation_files/recommendation_lists/ncf/params_optimized/K=20/optimized_ncf_K=20_recs.csv")


    train_file = "../datasets/recommender_train_test_oficial/train.csv"

    user_list_path = "../datasets/explanation_raw_files/user_split_train_val_test/test_users.csv"
    users = pd.read_csv(user_list_path)
    user_list = users["userId"].values

    # Path reorder
    for rec_file in recs_files:
        path_reord = PathReordering(train_file, rec_file,
                                    "../datasets/knowledge-graphs/props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], user_list=user_list)
        path_reord.prepare_data(expl_alg, n_explain)











explanation_configs = {
    "rec_algs": "BPRMF NCF UserKNN ItemKNN",
    # "rec_algs": "NCF",
    "pitems": 0.1 ,
    "policy": "last" ,
    "expl_alg": "explod" ,
    "n_explain": 10
}


run_explod(
    explanation_configs["rec_algs"].split(),
    explanation_configs["expl_alg"],
    explanation_configs["n_explain"]
    )