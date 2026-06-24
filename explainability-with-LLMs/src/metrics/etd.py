def etd_metric(explanation_types: list, k: int, total_types: int):
    """
    Compute the ETD score for one user's set of explanation types.

    ETD measures how many distinct explanation types appear in the selected
    explanations, normalized by the maximum possible diversity under the
    current cutoff ``k`` and the total number of available types.

    Parameters
    ----------
    explanation_types : list
        List of explanation-type labels extracted from the user's selected
        explanations.
    k : int
        Recommendation cutoff used during evaluation.
    total_types : int
        Total number of distinct explanation types available in the dataset.

    Returns
    -------
    float
        Ratio between the number of unique explanation types present in
        ``explanation_types`` and ``min(k, total_types)``.
    """

    return len(set(explanation_types)) / (min(k, total_types))