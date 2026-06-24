import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate_explanations(file_name: str, m_items: list, m_props: list, total_items: dict, total_props: dict,
                          all_sep: list, all_etd: list, all_f1: list):
    """
    Diversity of explanation metrics. We will use 10 metrics: mean and std item and prop user diversity
    total item and prop aggregate diversity and entropy and gini
    :param file_name: the name of the results file
    :param m_items: distribution of different historic items shown to user in explanations
    :param m_props:distribution of different KG props shown to user in explanations
    :param total_items: dict of items, quantity of times used in explanations
    :param total_props: dict of prop, quantity of times used in explanations
    :param all_etd: list of all users etd metric
    :param all_sep: list of all users sep metric
    :return: file saved with metrics
    """

    mean_useritem_aggr = "Mean user item aggregate diversity: " + str(np.array(m_items).mean())
    std_useritem_aggr = "Std user item aggregate diversity: " + str(np.array(m_items).std())
    mean_userprop_aggr = "Mean user property aggregate diversity: " + str(np.array(m_props).mean())
    std_userprop_aggr = "Std user property aggregate diversity: " + str(np.array(m_props).std())
    total_items_str = "Total items aggregate diversity: " + str(len(total_items))
    total_props_str = "Total property aggregate diversity: " + str(len(total_props))
    mean_etd = "ETD metric: " + str(np.array(all_etd).mean())
    mean_sep = "SEP metric: " + str(np.array(all_sep).mean())
    mean_f1 = "f1 metric: " + str(np.array(all_f1).mean())
    std_etd = "std ETD metric: " + str(np.array(all_etd).std())
    std_sep = "std SEP metric: " + str(np.array(all_sep).std())
    std_f1 = "std F1 metric: " + str(np.array(all_f1).std())

    f = open(file_name, mode="w", encoding='utf-8')
    f.write(file_name + "\n")
    f.write(mean_useritem_aggr + "\n")
    f.write(std_useritem_aggr + "\n")
    f.write(mean_userprop_aggr + "\n")
    f.write(std_userprop_aggr + "\n")
    f.write(total_items_str + "\n")
    f.write(total_props_str + "\n")
    f.write(mean_etd + "\n")
    f.write(mean_sep + "\n")
    f.write(mean_f1 + "\n")
    f.write(std_etd + "\n")
    f.write(std_sep + "\n")
    f.write(std_f1 + "\n")
    f.close()

    print("\n" + file_name)
    print(mean_useritem_aggr)
    print(std_useritem_aggr)
    print(mean_userprop_aggr)
    print(std_userprop_aggr)
    print(total_items_str)
    print(total_props_str)
    print(mean_etd)
    print(mean_sep)
    print(mean_f1)
    print(std_etd)
    print(std_sep)
    print(std_f1)





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
            links = sorted(set(prop_set[prop_set["obj"] == p]['prop'].values))
            l_memo = sorted(set(memo_sep.keys()).intersection(set(links)))
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


def etd_metric(explanation_types: list, k: int, total_types: int):
    """
    Metric proposed by Ballocu 2022
    :param explanation_types: list of explanation types used in the explanations
    :param k: number of recommendations
    :param total_types: total number of explanation types in the dataset
    :return: the division between the explanation types in the explanations and the minimum between the k and total_types
    """
    return len(set(explanation_types)) / (min(k, total_types))

def f1_metric(sep_value: float, etd_value: float):
    if sep_value + etd_value == 0:
        return 0
    return (2 * sep_value * etd_value) / (sep_value + etd_value)