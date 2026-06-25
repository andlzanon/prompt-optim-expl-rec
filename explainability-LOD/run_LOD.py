from path_reordering import PathReordering
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"


RECOMMENDATION_FILES = {
    "BPRMF": DATASETS_DIR / "recommendation_files" / "recommendation_lists" / "bprmf" / "params_optimized" / "K=20" / "optimized_bprmf_K=20_recs.csv",
    "UserKNN": DATASETS_DIR / "recommendation_files" / "recommendation_lists" / "user_knn" / "params_optimized" / "K=20" / "optimized_user_knn_K=20_recs.csv",
    "ItemKNN": DATASETS_DIR / "recommendation_files" / "recommendation_lists" / "item_knn" / "params_optimized" / "K=20" / "optimized_item_knn_K=20_recs.csv",
    "NCF": DATASETS_DIR / "recommendation_files" / "recommendation_lists" / "ncf" / "params_optimized" / "K=20" / "optimized_ncf_K=20_recs.csv",
}


def run_explod(rec_algs=None, expl_alg='explod', n_explain=10):
    """
    Run LOD explanation experiments for MovieLens recommendations.
    :param rec_algs: recommender algorithms whose recommendation files will be explained
    :param n_explain: number of recommended items to explain per user
    :param expl_alg: explanation algorithm to run. Currently, the implemented option is "explod".
    :return: explanation and metric files are written under datasets/lod_results
    """

    recs_files = []

    if rec_algs is not None:
        for rec_alg, rec_file in RECOMMENDATION_FILES.items():
            if rec_alg in rec_algs:
                recs_files.append(rec_file)

    train_file = DATASETS_DIR / "recommender_train_test_oficial" / "train.csv"

    user_list_path = DATASETS_DIR / "explanation_raw_files" / "user_split_train_val_test" / "test_users.csv"
    users = pd.read_csv(user_list_path)
    user_list = users["userId"].values

    for rec_file in recs_files:
        path_reord = PathReordering(train_file, rec_file,
                                    DATASETS_DIR / "knowledge-graphs" / "props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], user_list=user_list)
        path_reord.prepare_data(expl_alg, n_explain)


def main():
    explanation_configs = {
        "rec_algs": "BPRMF NCF UserKNN ItemKNN",
        "expl_alg": "explod",
        "n_explain": 10
    }

    run_explod(
        explanation_configs["rec_algs"].split(),
        explanation_configs["expl_alg"],
        explanation_configs["n_explain"]
    )


if __name__ == "__main__":
    main()
