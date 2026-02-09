import pandas as pd

def random_mode(rec_props_set, num_props):

    if(num_props > rec_props_set.shape[0]):
        print("Num props maior que a quantidade de props existentes.\n Retornando todas as props existentes...")
        return rec_props_set

    random_rows = rec_props_set.sample(n=num_props, random_state=42)
    return random_rows

def gerar_texto(df_list: list, user_train_set: pd.DataFrame, interactions_flag: bool =True, debug: bool = False) -> str:

    """
    Build a natural language prompt describing user interactions, recommendations,
    and their corresponding explanation paths.

    This function generates a structured textual prompt used to explain
    recommendations in a recommender system context. The prompt includes:
    1. The user's most recent interacted items (optional).
    2. A list of recommended items.
    3. Enumerated explanation paths connecting interacted items to recommended items
       via shared attributes.
    4. Instructions specifying how to select exactly one explanation path per
       recommendation.

    Explanation paths follow the format:
        Interacted Item -> Attribute -> Recommended Item

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        A list of DataFrames, one per recommended item.
        Each DataFrame must contain the following columns:
        - 'interacted_item_name'
        - 'recommended_item_name'
        - 'common_props'

    user_train_set : pd.DataFrame
        DataFrame containing the user's interaction history.
        Must include a 'title' column and a 'timestamp' column.
        Interactions are sorted by recency (descending timestamp).

    interactions_flag : bool, optional (default=True)
        If True, includes the user's recently interacted items
        at the beginning of the generated prompt.
        If False, the prompt starts directly from the recommendation section.

    Returns
    -------
    str
        A formatted text prompt describing user interactions,
        recommendations, and explanation paths.

    Notes
    -----
    - The function assumes that `df_list` is already filtered according to the
      desired number of recommendations and explanation paths.
    - The output text is intended to be consumed by a language model or
      explanation selection system.
    - Debug `print` statements are present and may be removed in production.

    """

    text = ""

    if(interactions_flag == True):
        # Maior timestamp -> mais recente
        user_train_set = user_train_set.sort_values(by='timestamp', ascending=False)

        user_interacted_items = user_train_set[['title']]

        text += "In a recommender system, a user has interacted some items, chronologically, in descending order, the" \
        " last items interacted were:\n"

        if(debug):
            print(user_interacted_items.values)

        for i, name in enumerate(user_interacted_items.values):
            text += f"{i}. {name[0]}\n"

    text += f"\nThe user has top-{len(df_list)} recommendations, and possible explanations paths. Explanation paths " \
            "connect an interacted item to a recommended item by attributes. Below are the user recommendations" \
            " followed by enumerated explanation paths, and the symbol '->' means the connection between an item and an " \
            "attribute:\n\n"

    for df in df_list:
        text += f"For the recommended item '{df['recommended_item_name'].head(1).values[0]}':\n"
        if(debug):
            print("----------------------------------------------------")
            print(df['recommended_item_name'].head(1).values)
        for i, (i_name, r_name, prop) in enumerate(
        zip(df['interacted_item_name'], df['recommended_item_name'], df['common_props']),
        start=1):
            text += f"{i}. {i_name} -> {prop} -> {r_name}\n"
            if(debug):
                print(f"{i}. {i_name} -> {prop} -> {r_name}")

        text += "\n"

    # text += "Please choose one explanation path for each recommendation considering the following criteria:\n" \
    #         "1. Diversity of attributes: Each path is composed of attributes that connect an interacted item node " \
    #         "with a recommended. Diversify these attributes for the chosen explanation paths set of all recommendations;\n" \
    #         "2. Popularity of attributes: Each path is composed of attributes that connect an interacted item node with " \
    #         "a recommended. Use popular attributes for the chosen explanation paths set of all recommendations;\n" \
    #         "3. Recency of interacted items: Use explanation paths that connects recently interacted items with the " \
    #         "recommended.\n\n" \

    text += "Please choose one explanation path for each recommendation considering the following criteria:\n" \
            "1. Diversity of attributes: Each path is composed of attributes that connect an interacted item node " \
            "with a recommended. Diversify these attributes for the chosen explanation paths set of all recommendations;\n\n" \


    text += "Output exactly in the same format as below with only the chosen explanation for each recommendation " \
            "starting with the name and then the explanation, separated with the symbol '|'.The path's attributes " \
            "are separated with '->'. An example of the exact format I want you to output is:\n" \
            "Gangs of New York | Titanic -> Leonardo DiCaprio -> Gangs of New York\n" \
            "Gladiator | Erin Brockovich -> Academy Award for Best Director -> Gladiator\n" \
            "Tarzan | Ratatouille -> Walt Disney Pictures -> Tarzan\n" \
            "A Bug's Life | Ghostbusters -> adventure film -> A Bug's Life\n" \
            "War Horse | Band of Brothers -> Steven Spielberg -> War Horse\n" \

    return text

def gerar_prompt(user_id: int, num_recs: int, num_props_per_rec: int, user_train_set: pd.DataFrame, algorithm: str, mode: str ='random') -> str | None:

    """
    Generate a textual prompt containing explanation paths for a given user.

    For a given user_id, this function:
    1. Loads the user's explanation paths (properties) from a CSV file.
    2. Selects the first `num_recs` recommended items.
    3. For each selected recommendation, chooses
       `num_props_per_rec` explanation paths according to the specified strategy.
    4. Builds and returns a formatted text prompt using the selected paths.

    Example
    -------
    For num_recs = 3 and num_props_per_rec = 4, the structure is:

        - Recommendation 1
            - path_prop_1
            - path_prop_2
            - path_prop_3
            - path_prop_4
        - Recommendation 2
            - path_prop_1
            - path_prop_2
            - path_prop_3
            - path_prop_4
        - Recommendation 3
            - path_prop_1
            - path_prop_2
            - path_prop_3
            - path_prop_4

    Parameters
    ----------
    user_id : int
        Identifier of the user for whom the prompt is generated.

    num_recs : int
        Number of recommendations to include in the prompt.

    num_props_per_rec : int
        Number of explanation paths (property paths) to include per recommendation.

    user_train_set : pd.DataFrame
        DataFrame containing the user's interaction history.
        Used to contextualize the generated prompt.

    algorithm : str
        String to determine from which algorithm the system is gonna get the explanation_paths
        
    mode : str, optional (default='random')
        Strategy used to select explanation paths.
        Currently supported:
        - 'random': randomly samples explanation paths per recommendation.

    Returns
    -------
    str or None
        The generated prompt as a string.
        Returns None if `num_recs` exceeds the number of available recommendations.

    Notes
    -----
    - The function assumes the existence of a CSV file at:
      '../knowledge-graphs/user_props/{user_id}_user_id.csv'
    - The CSV must contain a column named 'recommended_item_id'.
    - Path selection strategies (e.g., popularity, diversity) can be added
      by extending the `mode` parameter logic.
    """

    user_props_PATH = f"../datasets/explanation_paths/{algorithm}-opt/{algorithm}_{user_id}_user_id.csv"
    user_props_set = pd.read_csv(user_props_PATH)

    all_recs = user_props_set['recommended_item_id'].drop_duplicates()

    if(num_recs > len(all_recs)):
        print(f"O número de recomendações máximo é {len(all_recs)}. Por favor, escolha um valor menor ou "
              f"igual a {len(all_recs)}")
        return
    
    selected_recs = all_recs.head(num_recs)

    # Lista de dataframes, cada dataframe com apenas as informações necessarias, conforme as restrições dos parametros(num_recs e num_props_per_rec)
    selected_props_set_list = []

    for rec_id in selected_recs:
        props_rec_set = user_props_set[user_props_set['recommended_item_id'] == rec_id]
        if(mode == "random"):
            selected_props_set = random_mode(props_rec_set, num_props_per_rec)
        selected_props_set_list.append(selected_props_set)

    prompt = gerar_texto(selected_props_set_list, user_train_set)

    return prompt

def main():

    # Getting the train set to get the interactions of all users
    train_PATH = "../datasets/recommender_train_test_oficial/train.csv"
    movie_PATH = "../datasets/ml-latest-small/movies.csv"

    train_set = pd.read_csv(train_PATH, names=['user_id', 'item_id', 'relevance', 'timestamp'])
    train_set = train_set[['user_id', 'item_id', 'timestamp']]
    movies_set = pd.read_csv(movie_PATH)

    # Mutable parameters
    user_id = 1
    num_recs = 20
    num_props_per_rec = 5
    mode="random"
    # algorithm = "user_knn"
    # algorithm = "item_knn"
    # algorithm = "bprmf"
    algorithm = "ncf"

    user_train_set = train_set[train_set['user_id'] == user_id]
    user_train_set = user_train_set.merge(
        movies_set[['movieId', 'title']],
        left_on='item_id',
        right_on='movieId',
        how='left'
    ).drop(columns='movieId')

    prompt = gerar_prompt(user_id, num_recs, num_props_per_rec, user_train_set, algorithm, mode)

    print("\n\n\n\n")
    print("PROMPT:\n")
    print(prompt)
    return

main()