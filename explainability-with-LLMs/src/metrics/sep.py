import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def sep_metric(beta: float, props: list, prop_set: pd.DataFrame, memo_sep: dict):
    """
    Shared Entity Popularity (SEP) metric proposed in https://dl.acm.org/doi/abs/10.1145/3477495.3532041
    :param beta: parameter for the exponential decay
    :param props: list of list of properties used for each recommendation explanation path
    :param prop_set: property set extrated from Wikidata
    :param memo_sep: memoization for sep values across users
    :return: the sep metric for the user, the sep for every recommendation is the mean of the sep of every recommendation
        and the sep for every recommendation is the mean of the sep for every item in the explanation path
    """

    # user variables for the mean sep of each explanation and scaler
    total_sum = 0
    total_n = 0
    scaler = MinMaxScaler()
    # for every list of properties in the user list of explanations
    for expl_props in props:
        # explanation variables for the mean sep of each explanation
        items_sum = 0
        items_n = 0
        # for every property list of each explanation
        for p in expl_props:
            # obtain the most popular link to of the property e.g. link actor from property Brad Pitt
            links = list(set(prop_set[prop_set["obj"] == p]['prop'].values))
            l_memo = list(set(memo_sep.keys()).intersection(set(links)))
            if len(l_memo) > 0:
                memo_df = memo_sep[l_memo[0]]
                p_sep_value = memo_df.loc[p].iloc[-1]
            else:
                link_df = prop_set[prop_set["prop"].isin(links)]
                # generate dataset with property as index and count as column
                count_link = pd.DataFrame(link_df.groupby("obj").count())
                count_link = count_link.sort_values(by=count_link.columns[0], ascending=True)
                count_link = pd.DataFrame(count_link[count_link.columns[0]])
                # initialize sep column with value -1
                count_link["sep"] = -1
                count_link['sep'] = count_link['sep'].astype(float)

                # obtain min value so we do not need to calculate every time
                # and initialize the last value and last sep as min according to the base case
                min = count_link[count_link.columns[0]].min()
                last_value = min
                last_sep = min
                for i, row in count_link.iterrows():
                    # if it is min, then lir is the value
                    if row.iloc[0] == min:
                        count_link.at[i, "sep"] = min
                    # else if the count is the same repeat the sep, otherwise, calculate new sep
                    else:
                        if row.iloc[0] == last_value:
                            count_link.at[i, "sep"] = last_sep
                        else:
                            sep = (1 - beta) * last_sep + beta * row.iloc[0]
                            count_link.at[i, "sep"] = sep
                            last_value = row.iloc[0]
                            last_sep = sep

                # generate normalized sep column
                try:
                    count_link['normalized'] = scaler.fit_transform(
                        np.asarray(count_link[count_link.columns[-1]]).astype(np.float64).reshape(-1, 1)).reshape(-1)
                except ValueError:
                    continue
                p_sep_value = count_link.loc[p].iloc[-1]
                for l in links:
                    memo_sep[l] = count_link

            # obtain sep value for the property and calculate mean
            items_sum = items_sum + p_sep_value
            items_n = items_n + 1

        # calculate total mean
        try:
            total_sum = total_sum + (items_sum / items_n)
            total_n = total_n + 1
        except ZeroDivisionError:
            total_n = total_n + 1

    return total_sum / total_n