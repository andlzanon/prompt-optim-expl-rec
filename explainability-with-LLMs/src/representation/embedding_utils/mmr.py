from src.representation.embedding_utils.similarity import cosine_similarity, to_2d_array

import numpy as np

def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    lambda_param: float = 0.5,
) -> list[int]:
    """
    Select candidate indices with Maximal Marginal Relevance (MMR).

    The function ranks candidate embeddings by balancing query relevance and
    diversity among the already selected candidates. Relevance and diversity
    are both measured with cosine similarity.

    Parameters
    ----------
    query_embedding : np.ndarray
        Query embedding used to measure candidate relevance.
    candidate_embeddings : np.ndarray
        Matrix of candidate embeddings to rank. One-dimensional inputs are
        converted to a single-row matrix.
    top_k : int
        Maximum number of candidate indices to select.
    lambda_param : float, default=0.5
        Trade-off between relevance and diversity. Values closer to ``1.0``
        prioritize relevance more heavily, while values closer to ``0.0``
        emphasize diversity.

    Returns
    -------
    list[int]
        List of selected candidate indices in the order they are chosen. An
        empty list is returned when ``top_k`` is non-positive or when there
        are no candidate embeddings.

    Raises
    ------
    ValueError
        Raised when ``lambda_param`` is outside the inclusive range
        ``[0.0, 1.0]``.

    Notes
    -----
    This function is used in the prompt-optimization flow to diversify the
    set of reference instructions while still favoring candidates that are
    relevant to the current query embedding.
    """

    candidate_embeddings = to_2d_array(candidate_embeddings)
    if top_k <= 0 or len(candidate_embeddings) == 0:
        return []

    if not 0.0 <= float(lambda_param) <= 1.0:
        raise ValueError("lambda_param must be between 0.0 and 1.0.")

    top_k = min(int(top_k), len(candidate_embeddings))
    # Relevance scores compare the query against every candidate.
    relevance = cosine_similarity(query_embedding, candidate_embeddings)[0]
    # Pairwise similarities are used to penalize candidates that are too close
    # to items already selected.
    pairwise = cosine_similarity(candidate_embeddings, candidate_embeddings)

    selected = [int(np.argmax(relevance))]

    while len(selected) < top_k:
        scores = []

        for index in range(len(candidate_embeddings)):
            if index in selected:
                # Prevent already selected items from being chosen again.
                scores.append(-np.inf)
                continue

            diversity = max(pairwise[index][selected_index] for selected_index in selected)
            score = lambda_param * relevance[index] - (1.0 - lambda_param) * diversity
            scores.append(score)

        selected.append(int(np.argmax(scores)))

    return selected