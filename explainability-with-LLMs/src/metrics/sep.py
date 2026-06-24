import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def _build_links_key(prop_set: pd.DataFrame, prop_value: str) -> tuple[str, ...]:
    """
    Return a deterministic key representing all links associated with a property.

    Parameters
    ----------
    prop_set : pd.DataFrame
        Knowledge-graph property table expected to contain at least the
        columns ``"obj"`` and ``"prop"``.
    prop_value : str
        Property value whose outgoing relation labels should be collected.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple containing the unique relation labels associated with
        ``prop_value``. Returns an empty tuple when the property does not have
        matching links in ``prop_set``.

    Raises
    ------
    KeyError
        May be raised if ``prop_set`` does not expose the required columns.
    """

    links = (
        prop_set.loc[prop_set["obj"] == prop_value, "prop"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return tuple(links)

def _build_sep_table(
    beta: float,
    prop_set: pd.DataFrame,
    links_key: tuple[str, ...],
) -> pd.DataFrame:
    """
    Build the normalized SEP lookup table for one exact set of links.

    Parameters
    ----------
    beta : float
        Exponential decay parameter used when computing the cumulative SEP
        values from sorted popularity counts.
    prop_set : pd.DataFrame
        Knowledge-graph property table expected to contain at least the
        columns ``"obj"`` and ``"prop"``.
    links_key : tuple[str, ...]
        Exact set of relation labels that defines the lookup table to be
        built.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by property value with the columns ``"count"``,
        ``"sep"``, and ``"normalized"``. Returns an empty DataFrame when
        ``links_key`` is empty, when the filtered subset is empty, or when
        normalization cannot be computed.

    Raises
    ------
    KeyError
        May be raised if ``prop_set`` does not expose the required columns.
    """

    if not links_key:
        return pd.DataFrame()

    link_df = prop_set[prop_set["prop"].isin(links_key)]
    if link_df.empty:
        return pd.DataFrame()

    count_link = link_df.groupby("obj").size().to_frame(name="count")
    count_link = count_link.sort_values(by="count", ascending=True, kind="mergesort")
    count_link["sep"] = -1.0

    min_count = float(count_link["count"].min())
    last_value = min_count
    last_sep = min_count

    for obj, row in count_link.iterrows():
        current_count = float(row["count"])

        if current_count == min_count:
            count_link.at[obj, "sep"] = min_count
        elif current_count == last_value:
            count_link.at[obj, "sep"] = last_sep
        else:
            current_sep = (1 - beta) * last_sep + beta * current_count
            count_link.at[obj, "sep"] = current_sep
            last_value = current_count
            last_sep = current_sep

    try:
        scaler = MinMaxScaler()
        count_link["normalized"] = scaler.fit_transform(
            count_link[["sep"]].astype(np.float64)
        ).reshape(-1)
    except ValueError:
        return pd.DataFrame()

    return count_link

def sep_metric(beta: float, props: list, prop_set: pd.DataFrame, memo_sep: dict):
    """
    Compute the Shared Entity Popularity (SEP) score for one user's explanations.

    SEP rewards explanation paths whose intermediate properties are less
    popular within the knowledge graph. For each property in each explanation,
    the function builds or reuses a normalized lookup table over the exact set
    of associated relation labels, retrieves the property's normalized SEP
    value, averages over the properties in the explanation, and then averages
    again over the user's explanations.

    Parameters
    ----------
    beta : float
        Exponential decay parameter used by the SEP recurrence.
    props : list
        Nested list in which each element represents one explanation and
        contains the properties that appear in that explanation path.
    prop_set : pd.DataFrame
        Knowledge-graph property table extracted from Wikidata.
    memo_sep : dict
        Mutable memoization dictionary used to cache normalized SEP lookup
        tables across repeated calls.

    Returns
    -------
    float
        Mean SEP score across the provided explanations. Returns ``0.0`` when
        no explanation contributes a valid property score.

    Raises
    ------
    KeyError
        May be raised if ``prop_set`` does not expose the required columns.
    """

    # user variables for the mean sep of each explanation and scaler
    total_sum = 0.0
    total_n = 0
    # for every list of properties in the user list of explanations
    for expl_props in props:
        # explanation variables for the mean sep of each explanation
        items_sum = 0.0
        items_n = 0
        # for every property list of each explanation
        for p in expl_props:
            links_key = _build_links_key(prop_set=prop_set, prop_value=p)
            if not links_key:
                continue

            memo_df = memo_sep.get(links_key)
            if memo_df is None:
                memo_df = _build_sep_table(
                    beta=beta,
                    prop_set=prop_set,
                    links_key=links_key,
                )
                if memo_df.empty:
                    continue
                memo_sep[links_key] = memo_df

            if p not in memo_df.index:
                continue

            p_sep_value = float(memo_df.loc[p, "normalized"])

            # obtain sep value for the property and calculate mean
            items_sum += p_sep_value
            items_n += 1

        # calculate total mean
        total_n += 1
        if items_n > 0:
            total_sum += items_sum / items_n

    if total_n == 0:
        return 0.0

    return total_sum / total_n