import _io
from collections import Counter

import pandas as pd
from pandas.core.indexing import IndexingError

import evaluation_utils as eval
from lod_reordering import LODPersonalizedReordering


class PathReordering(LODPersonalizedReordering):
    def __init__(self, train_file: str, recs_file: str, prop_path: str, prop_cols: list, cols_used: list, user_list: list):
        """
        Path Reordering class: this algorithm will reorder the output of other recommendation algorithm based on the
        best path from an historic item and a recommended one. The best paths are extracted based on the value for each
        object of the LOD with the semantic profile
        :param train_file: train file in which the recommendations of where computed
        :param recs_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param cols_used: columns used from the test and train set
        :param user_list: list to represent which users to run the explod algorithm
        """


        super().__init__(train_file, recs_file, prop_path, prop_cols, cols_used, user_list)

    
    def prepare_data(self, expl_alg: str, n_explain: int):

        file_to_be_explained = self.recs_file.split("/")[-1]

        output_path = f"../datasets/lod_results/output/output_explanations_{file_to_be_explained}"
        average_metrics_path = f"../datasets/lod_results/average_metrics/avg_metrics_explanations_{file_to_be_explained}"
        individual_metrics_path = f"../datasets/lod_results/individual_metrics/indiv_metrics_explanations_{file_to_be_explained}"
        

        f = open(output_path, mode="w", encoding='utf-8')
        f.write(output_path + "\n")
        m_items = []
        m_props = []
        total_items = {}
        total_props = {}
        total_etd = []
        total_sep = []
        total_f1 = []
        memo_sep = {}
        usep = -1
        uetd = -1
        uf1 = -1

        individual_metrics_data = []
        for u in self.recs_set.index.unique():

            # get items that the user interacted and recommended by an algorithm
            items_historic = self.train_set.loc[u].sort_values(by=self.cols_used[-1], ascending=False)

            

            try:
                items_historic = items_historic[self.cols_used[1]].to_list()
            except AttributeError:
                items_historic = list(self.train_set.loc[u][self.cols_used[1]])[:-1]

            print("User: " + str(u))
            f.write("\nUser: " + str(u) + "\n")
            f.write("Items interacted by the user\n")
            
            for i in items_historic:
                value = self.prop_set.loc[i, "title"]
                if(isinstance(value, pd.Series)):
                    movie_name = value.iloc[0]
                else:
                    movie_name = value
                    
                f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            # get semantic profile and extract the best paths from the suggested item to the recommended
            user_semantic_profile = self.user_semantic_profile(items_historic)

            f.write("\nUsers favorites attributes on the kG\n")
            s_user_sem_pro = dict(sorted(user_semantic_profile.items(), key=lambda item: item[1], reverse=True))
            n = 0

            for k in s_user_sem_pro.keys():
                if n < 5:
                    f.write(k + "\n")
                    n = n + 1
                else:
                    break

            
            item_rank = self.recs_set
            item_rank = list(item_rank.loc[u][:n_explain]["item_id"])

            f.write("\nRecommendations\n")
            for i in item_rank:
                try:
                    movie_name = self.prop_set.loc[i].iloc[0, 0]
                except IndexingError:
                    movie_name = self.prop_set.loc[i].iloc[0]
                f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")


            items, props = [], []
            if expl_alg == 'explod':
                items, props, (usep, uetd, uf1) = self.__explod_ranked_paths(item_rank, items_historic,
                                                                        user_semantic_profile, u, f, memo_sep)

            f.write("\n")

            total_items = dict(Counter(total_items) + Counter(items))
            m_items.append(len(items))
            total_props = dict(Counter(total_props) + Counter(props))
            m_props.append(len(props))
            total_sep.append(usep)
            total_etd.append(uetd)
            total_f1.append(uf1)
            
            individual_metrics_data.append({
                'userId': str(u),
                'sep': usep,
                'etd': uetd,
                'f1': uf1,
            })

        f.close()

        individual_metrics_df = pd.DataFrame(individual_metrics_data)
        individual_metrics_df.to_csv(individual_metrics_path, index=False)

        eval.evaluate_explanations(average_metrics_path, m_items, m_props, total_items, total_props,
                                   total_sep, total_etd, total_f1)




######## IMPORTANTE ############
    def __explod_ranked_paths(self, ranked_items: list, items_historic: list, semantic_profile: dict,
                              user: int, file: _io.TextIOWrapper, memo_sep: dict):
        """
        Build explanation to recommendations based on the ExpLOD, method, explained in https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param ranked_items: list of the recommended items
        :param items_historic: list of historic items
        :param semantic_profile: dictionary with property as key and score as value
        :param user: user id of user to show explanations to
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """

        # get properties from historic and recommended items
        items_historic = list(dict.fromkeys(items_historic))  # remove duplicates, preserve order
        hist_props = self.prop_set.loc[items_historic]
        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for r in ranked_items:
            rec_props = self.prop_set.loc[r]

            # check properties on both sets
            intersection = pd.Series(sorted(set(hist_props['obj']).intersection(set(rec_props['obj']))))
            # get properties with max value
            max = -1
            max_props = []
            for pi in intersection:
                value = semantic_profile[pi]
                if value > max:
                    max = value
                    max_props.clear()
                    max_props.append(pi)
                elif value == max:
                    max_props.append(pi)

            max_props = sorted(max_props)

            # build sentence
            user_df = self.train_set.loc[user]
            user_item = user_df[
                user_df[user_df.columns[0]].isin(list(hist_props[hist_props['obj'].isin(max_props)].index.unique()))]
            hist_ids = list(user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]])
            hist_lists.append(hist_ids)
            hist_names = hist_props.loc[hist_ids][self.prop_cols[1]].unique()
            try:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]].unique()[0]
            except AttributeError:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]]

            file.write("\nPaths for the Recommended Item: " + str(r) + "\n")
            origin = ""
            # check for others with same value
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
                hist_items = self.__add_dict(hist_items, i)
            origin = origin[:-2]

            path_sentence = " nodes: "
            prop_lists.append(max_props)
            for n in max_props:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + rec_name + "\""
            file.write(origin + path_sentence + destination)

        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(ranked_items), len(self.prop_set['obj'].unique()))
        f1 = eval.f1_metric(sep, etd)

        print()
        print()
        print(len(self.prop_set['obj'].unique()))
        print((self.prop_set['obj'].unique()))
        print()
        print()

        return hist_items, nodes, (sep, etd, f1)

   

    def __add_dict(self, d: dict, key) -> dict:
        """
        Function to increment one in the key
        :param d: dictionary
        :param key: key to increment value
        :return: new dictionary
        """
        try:
            d[key] = d[key] + 1
        except KeyError:
            d[key] = 1

        return d
