def etd_metric(explanation_types: list, k: int, total_types: int):
    """
    Metric proposed by Ballocu 2022
    :param explanation_types: list of explanation types used in the explanations
    :param k: number of recommendations
    :param total_types: total number of explanation types in the dataset
    :return: the division between the explanation types in the explanations and the minimum between the k and total_types
    """
    return len(set(explanation_types)) / (min(k, total_types))