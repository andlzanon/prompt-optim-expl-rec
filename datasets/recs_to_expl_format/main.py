import pandas as pd


def get_train_into_expl_format(
    train_csv_path: str,
    movies_path: str,
    rating_path: str,
    destination_path: str
) -> int:
    """
    Converts the training dataset into an enriched format containing
    explicit ratings and movie titles.

    Output columns:
        userId, movieId, rating, title
    """

    # Load datasets
    train_df = pd.read_csv(
        train_csv_path,
        names=["userId", "movieId", "implicit_feedback", "timestamp"]
    )
    movies_df = pd.read_csv(movies_path)
    rating_df = pd.read_csv(rating_path)

    # Keep only identifiers required for joins
    train_df = train_df[["userId", "movieId"]]

    # Add explicit ratings
    train_df = train_df.merge(
        rating_df,
        on=["userId", "movieId"],
        how="left"
    )

    # Add movie metadata
    train_df = train_df.merge(
        movies_df,
        on="movieId",
        how="left"
    )

    # Ensure integer ratings
    train_df["rating"] = train_df["rating"].astype(int)

    # Save enriched dataset
    train_df.to_csv(destination_path, index=False)

    return 0



def transform_recs_in_list(movies_path: str) -> int:
    """
    Converts recommendation files into lists of movie titles per user.

    Output format:
        userId | response (list of titles)
    """

    # Load movieId -> title mapping
    movie_df = pd.read_csv(movies_path)[["movieId", "title"]]

    BASE_RECS_PATH = "../recommendation_files/recommendation_lists/"
    OUTPUT_PATH = "recommendation_lists/"

    ALGORITHMS = ["bprmf", "item_knn", "ncf", "user_knn"]
    PARAM_TYPES = ["params_default", "params_optimized"]
    K_VALUES = [1, 5, 10, 20, 50, 100, 200]

    for algorithm in ALGORITHMS:
        for param_type in PARAM_TYPES:
            for k in K_VALUES:

                # Build file names
                prefix = "default" if param_type == "params_default" else "optimized"
                file_name = f"{prefix}_{algorithm}_K={k}_recs.csv"

                input_path = (
                    f"{BASE_RECS_PATH}{algorithm}/"
                    f"{param_type}/K={k}/{file_name}"
                )
                output_path = (
                    f"{OUTPUT_PATH}{algorithm}/"
                    f"{param_type}/K={k}/named_{file_name}"
                )

                print(input_path)

                # Load recommendations
                recs_df = pd.read_csv(
                    input_path,
                    names=["userId", "movieId", "score"]
                )[["userId", "movieId"]]

                # Replace movieId with titles
                recs_df = recs_df.merge(
                    movie_df,
                    on="movieId",
                    how="left"
                )

                # Aggregate recommendations per user
                recs_df = recs_df.groupby(
                    "userId",
                    as_index=False
                ).agg(
                    response=("title", list)
                )

                # Save final file
                recs_df[["response", "userId"]].to_csv(
                    output_path,
                    index=False
                )

    return 0


# ======================
# Script execution
# ======================

TRAIN_CSV_PATH = "../recommender_train_test_oficial/train.csv"
MOVIES_PATH = "../ml-latest-small/movies.csv"
RATING_PATH = "../ml-latest-small/ratings.csv"
DESTINATION_PATH = "train_llm.csv"

get_train_into_expl_format(
    TRAIN_CSV_PATH,
    MOVIES_PATH,
    RATING_PATH,
    DESTINATION_PATH
)

transform_recs_in_list(MOVIES_PATH)
