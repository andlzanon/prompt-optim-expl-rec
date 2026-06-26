import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate_explanations(file_name: str, m_items: list, m_props: list, total_items: dict, total_props: dict,
                          all_sep: list, all_etd: list, all_f1: list):
    """
    Save aggregate explanation metrics.

    The output includes user-level item/property diversity, aggregate item/property
    diversity, and mean/std values for ETD, SEP, and their F1 combination.
    :param file_name: results file path
    :param m_items: number of distinct historical items used per user's explanations
    :param m_props: number of distinct KG property values used per user's explanations
    :param total_items: historical items and how many times they were used in explanations
    :param total_props: property values and how many times they were used in explanations
    :param all_sep: SEP metric values for all users
    :param all_etd: ETD metric values for all users
    :param all_f1: F1 values combining SEP and ETD for all users
    :return: file saved with aggregate metrics
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

    file_name_str = str(file_name)

    f = open(file_name, mode="w", encoding='utf-8')
    f.write(file_name_str + "\n")
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

    print("\n" + file_name_str)
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
    :param props: list of property-value lists used in each recommendation explanation
    :param prop_set: property set extracted from Wikidata
    :param memo_sep: memoization for sep values across users
    :return: SEP metric for the user
    """

    # User-level accumulators and scaler.
    total_sum = 0
    total_n = 0
    scaler = MinMaxScaler()
    # For every recommendation explanation.
    for expl_props in props:
        # Recommendation-level accumulators.
        items_sum = 0
        items_n = 0
        # For every property value used in the explanation.
        for p in expl_props:
            # Obtain the property types linked to the property value, e.g. actor for Brad Pitt.
            links = sorted(set(prop_set[prop_set["obj"] == p]['prop'].values))
            l_memo = sorted(set(memo_sep.keys()).intersection(set(links)))
            if len(l_memo) > 0:
                memo_df = memo_sep[l_memo[0]]
                p_sep_value = memo_df.loc[p].iloc[-1]
            else:
                link_df = prop_set[prop_set["prop"].isin(links)]
                # Generate dataset with property value as index and count as column.
                count_link = pd.DataFrame(link_df.groupby("obj").count())
                count_link = count_link.sort_values(by=count_link.columns[0], ascending=True)
                count_link = pd.DataFrame(count_link[count_link.columns[0]])
                # Initialize sep column with placeholder values.
                count_link["sep"] = -1
                count_link['sep'] = count_link['sep'].astype(float)

                # Initialize the recurrence with the minimum popularity count.
                min = count_link[count_link.columns[0]].min()
                last_value = min
                last_sep = min
                for i, row in count_link.iterrows():
                    # If the count is minimal, SEP starts with that value.
                    if row.iloc[0] == min:
                        count_link.at[i, "sep"] = min
                    # Reuse SEP for ties; otherwise calculate a new value.
                    else:
                        if row.iloc[0] == last_value:
                            count_link.at[i, "sep"] = last_sep
                        else:
                            sep = (1 - beta) * last_sep + beta * row.iloc[0]
                            count_link.at[i, "sep"] = sep
                            last_value = row.iloc[0]
                            last_sep = sep

                # Generate normalized SEP column.
                try:
                    count_link['normalized'] = scaler.fit_transform(
                        np.asarray(count_link[count_link.columns[-1]]).astype(np.float64).reshape(-1, 1)).reshape(-1)
                except ValueError:
                    continue
                p_sep_value = count_link.loc[p].iloc[-1]
                for l in links:
                    memo_sep[l] = count_link

            # Add SEP value for this property value.
            items_sum = items_sum + p_sep_value
            items_n = items_n + 1

        # Add mean SEP for this recommendation explanation.
        try:
            total_sum = total_sum + (items_sum / items_n)
            total_n = total_n + 1
        except ZeroDivisionError:
            total_n = total_n + 1

    return total_sum / total_n


def etd_metric(explanation_types: list, k: int, total_types: int):
    """
    Explanation Type Diversity (ETD) metric proposed by Ballocu 2022.
    :param explanation_types: property values used in the explanations
    :param k: number of recommendations
    :param total_types: total number of available property values in the dataset
    :return: number of distinct explanation types divided by min(k, total_types)
    """
    return len(set(explanation_types)) / (min(k, total_types))

def f1_metric(sep_value: float, etd_value: float):
    if sep_value + etd_value == 0:
        return 0
    return (2 * sep_value * etd_value) / (sep_value + etd_value)
