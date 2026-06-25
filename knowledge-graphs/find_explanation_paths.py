import networkx as nx
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"

def get_train_set(train_path: str, cols_used: list) -> pd.DataFrame:
    train_set = pd.read_csv(train_path, header=None)
    train_set.columns = cols_used
    train_set = train_set.set_index(cols_used[0])

    return train_set

def get_prop_set(props_wikidata_path, prop_cols) -> pd.DataFrame:
    prop_set = pd.read_csv(props_wikidata_path, usecols=prop_cols)
    prop_set = prop_set.dropna()
    prop_set = prop_set.set_index(prop_cols[0])

    return prop_set

def get_output_rec_set(output_rec_path: str, cols_used: list) -> pd.DataFrame:
    output_cols = ['user_id', 'movie_id', 'score']
    output_rec_set = pd.read_csv(output_rec_path, header=None)
    output_rec_set.columns = output_cols
    output_rec_set = output_rec_set.set_index(cols_used[0])

    return output_rec_set

def get_movie_set(movies_path: str) -> pd.DataFrame :
    movie_set = pd.read_csv(movies_path)

    return movie_set


def build_graph(train_set, prop_set) -> nx.Graph:
    """
    Build a graph with the information from the test set and the wikidata or dbpedia set to create
    a graph with users, items and property nodes e.g.: user 1 interacted the item Inception with Di Caprio as an actor
    therefore, on the graph there is a three nodes, one for each entity (user, item and actor/property) and three
    edges, one connecting user to item and other item to property: user 1 --> Inception --> Di Caprio
    :return: networkx graph with users, items and properties from the dbpedia or wikidata
    """

    user_item_set = train_set.copy()
    edgelist = pd.DataFrame(columns=['origin', 'destination'])

    user_item_set['origin'] = ['U' + x for x in user_item_set.index.astype(str)]
    user_item_set['destination'] = ['I' + x for x in user_item_set[user_item_set.columns[0]].astype(str)]

    edgelist = pd.concat([edgelist, user_item_set[['origin', 'destination']]], ignore_index=True)

    item_prop_copy = prop_set.copy()
    item_prop_copy['origin'] = ['I' + x for x in item_prop_copy.index.astype(str)]
    item_prop_copy['destination'] = item_prop_copy[prop_set.columns[-1]]

    edgelist = pd.concat([edgelist, item_prop_copy[['origin', 'destination']]], ignore_index=True)

    G = nx.from_pandas_edgelist(edgelist, 'origin', 'destination')

    return G



def user_interacted_recommended_paths(algorithm: str, graph: nx.Graph, ID_user: int, prop_set: pd.DataFrame, train_set: pd.DataFrame, recs_set: pd.DataFrame, movie_set: pd.DataFrame, cols_used: list):

    """
    Generate explanation paths between a user's interacted items and recommended items
    using a property-based graph, and store the results in a CSV file.

    Parameters
    ----------

    algorithm : str
        A string which can be:
        - bprmf
        - item_knn
        - ncf
        - user_knn

    graph : nx.Graph
        A NetworkX graph where:
        - Item nodes are prefixed with 'I'
        - Property nodes represent shared attributes between items

    ID_user : int
        The user identifier used to retrieve interactions and recommendations.

    prop_set : pd.DataFrame
        DataFrame containing item-to-property relationships.
        The index must correspond to item IDs and include a column named 'obj'
        representing property nodes.

    train_set : pd.DataFrame
        User-item interaction dataset indexed by user ID.
        Must contain item IDs and timestamps.

    output_rec_set : pd.DataFrame
        Recommendation dataset indexed by user ID.
        Must contain recommended item IDs ordered by relevance score.

    movie_set : pd.DataFrame
        DataFrame mapping item IDs to movie titles.
        Must contain columns ['movieId', 'title'].

    cols_used : list
        List of column names used in the datasets.
        Expected structure:
        - cols_used[1]: item ID column
        - cols_used[-1]: timestamp column

    Output
    ------
    CSV file
        A CSV file is saved to:
        "datasets/explanation_paths/{algorithm}-opt/{algorithm}_{ID_user}_user_id.csv"

        The file contains the following columns:
        - interacted_item_id
        - recommended_item_id
        - common_props
        - interacted_item_name
        - recommended_item_name

    Returns
    -------
    None
        The function writes results to disk and does not return a value.

    Notes
    -----
    - Only simple paths with a maximum length of 2 are considered:
      interacted_item -> property -> recommended_item
    - If no path exists between an interacted and recommended item,
      it is silently ignored.
    - The function assumes that the graph and DataFrames are already
      preprocessed and consistent.
    """

    output_path = DATASETS_DIR / "explanation_paths" / f"{algorithm}-opt" / f"{algorithm}_{ID_user}_user_id.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ordenacao decrescente por timestamp
    items_interacted = train_set.loc[ID_user].sort_values(by=cols_used[-1], ascending=False)

    try:
        items_interacted = items_interacted[cols_used[1]].to_list()
    except AttributeError:
        items_interacted = list(train_set.loc[ID_user][cols_used[1]])[:-1]

    # Ordenacao pelo score da lista de recomendados
    items_recommended = list(recs_set.loc[ID_user][cols_used[1]])


    interacted_codes = ['I' + str(i) for i in items_interacted]
    recommended_codes = ['I' + str(i) for i in items_recommended]
    interacted_props = list(set(prop_set.loc[prop_set.index.isin(items_interacted)]['obj']))
    subgraph = graph.subgraph(interacted_codes + recommended_codes + interacted_props)

    rows = []


    for rm in items_recommended:
        rm_node = 'I' + str(rm)
        for im in items_interacted:
            im_node = 'I' + str(im)

            try:
                paths = nx.all_simple_paths(subgraph, source=im_node, target=rm_node, cutoff=2)
                paths_s = [p for p in paths]
                for p in paths_s:
                    rows.append({
                        "interacted_item_id": im,
                        "recommended_item_id": rm,
                        "common_props": p[1]
                    })

            except (nx.exception.NetworkXNoPath, ValueError):
                pass




    df = pd.DataFrame(rows)

    df = df.merge(
        movie_set[["movieId", "title"]],
        left_on="interacted_item_id",
        right_on="movieId",
        how="left"
    ).rename(columns={"title": "interacted_item_name"}).drop(columns="movieId")

    df = df.merge(
        movie_set[["movieId", "title"]],
        left_on="recommended_item_id",
        right_on="movieId",
        how="left"
    ).rename(columns={"title": "recommended_item_name"}).drop(columns="movieId")


    df.to_csv(output_path, index=False)

    return


def create_explanation_paths_file(algorithm, graph, prop_set, train_set, recs_set, movie_set, cols_used):

    for user_id, _ in train_set.groupby("user_id"):

        user_interacted_recommended_paths(algorithm, graph, user_id, prop_set, train_set, recs_set, movie_set, cols_used)

    return







def main():

    props_wikidata_path = DATASETS_DIR / "knowledge-graphs" / "props_wikidata_movielens_small.csv"
    train_path = DATASETS_DIR / "recommender_train_test_oficial" / "train.csv"
    movies_path = DATASETS_DIR / "ml-latest-small" / "movies.csv"
    cols_used = ['user_id', 'movie_id', 'interaction', 'timestamp']
    prop_cols = ['movieId', 'title', 'prop', 'obj']
    
    train_set = get_train_set(train_path, cols_used)
    prop_set = get_prop_set(props_wikidata_path, prop_cols)
    movie_set = get_movie_set(movies_path)
    graph = build_graph(train_set, prop_set)

    algs_list = ["bprmf", "item_knn", "ncf", "user_knn"]
    
    for algorithm in algs_list:

        recs_path = DATASETS_DIR / "recommendation_files" / "recommendation_lists" / algorithm / "params_optimized" / "K=20" / f"optimized_{algorithm}_K=20_recs.csv"
        recs_set = get_output_rec_set(recs_path, cols_used)

        create_explanation_paths_file(algorithm, graph, prop_set, train_set, recs_set, movie_set, cols_used)


if __name__ == "__main__":
    main()
