def etd_metric(explanation_types: list, total_types: int):
    """
    Compute the ETD score for one user's set of explanation types.

    ETD measures how many distinct explanation types appear in the selected
    explanations, normalized by the maximum possible diversity under the
    number of generated explanations and the total number of available types.

    Parameters
    ----------
    explanation_types : list
        List of explanation-type labels extracted from the user's selected
        explanations.
    total_types : int
        Total number of distinct explanation types available in the dataset.

    Returns
    -------
    float
        Ratio between the number of unique explanation types present in
        ``explanation_types`` and ``min(len(explanation_types), total_types)``.
    """

    max_distinct = min(len(explanation_types), total_types)
    if max_distinct <= 0:
        return 0.0
    return len(set(explanation_types)) / max_distinct
