import numpy as np
import pandas as pd
import networkx as nx


class LODPersonalizedReordering(object):

    def __init__(self, train_file: str, recs_file: str, prop_path: str, prop_cols: list, cols_used: list, user_list: list):
        """
        Base class for LOD explanation generation.

        This class loads user interactions, recommendation lists, and KG properties,
        filters them to the selected users, and builds the graph used by the explanation step.
        It does not train a recommender and does not change the recommendation ranking.
        :param train_file: training interactions used as user history
        :param recs_file: recommendation output file to explain
        :param prop_path: path to the properties from DBpedia or Wikidata
        :param prop_cols: columns of the property set
        :param cols_used: columns used from the train set
        :param user_list: users to keep in the explanation pipeline
        """
        self.cols_used = cols_used

        self.train_file = train_file
        self.train_set = pd.read_csv(self.train_file, header=None)
        self.train_set.columns = self.cols_used
        # Filter dataset to keep only users from user_list.
        self.train_set = self.train_set[self.train_set["user_id"].isin(user_list)]
        self.train_set = self.train_set.set_index(self.cols_used[0])

        self.output_cols = ['user_id', 'item_id', 'score']
        self.recs_file = recs_file
        self.recs_set = pd.read_csv(self.recs_file, header=None)
        self.recs_set.columns = self.output_cols
        # Filter dataset to keep only users from user_list.
        self.recs_set = self.recs_set[self.recs_set["user_id"].isin(user_list)]
        self.recs_set = self.recs_set.set_index(self.cols_used[0])


        self.prop_path = prop_path
        self.prop_cols = prop_cols
        self.prop_set = pd.read_csv(self.prop_path, usecols=self.prop_cols)
        self.prop_set = self.prop_set.dropna()
        self.prop_set = self.prop_set.set_index(self.prop_cols[0])
        

        self.graph = self.__build_graph()
        self.user_list = user_list


    def user_semantic_profile(self, historic: list) -> dict:
        """
        Generate the user semantic profile from the user's historical items.

        Property values (e.g. George Lucas, action films, etc.) are scored as:
            score = (npi/i) * log(N/dft)
        where npi is the number of times the value appears in the user's history,
        i is the number of historical items, N is the total number of items in the
        property set, and dft is the number of items containing the value.
        :param historic: list of the items interacted by a user
        :return: dictionary with property values as keys and semantic scores as values
        """

        # Create npi, i and n columns used in the semantic profile formula.
        interacted_props = self.prop_set.loc[self.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby(self.prop_set.columns[-1])[self.prop_set.columns[-1]].transform('count')
        interacted_props['i'] = len(historic)
        interacted_props['n'] = len(self.prop_set.index.unique())

        # Count in how many distinct items each property value appears.
        items_per_obj = self.prop_set.reset_index().drop_duplicates(subset=[self.prop_set.columns[0], self.prop_set.columns[-1]]).set_index(
            self.prop_set.columns[-1])
        df_dict = items_per_obj.index.value_counts().to_dict()

        # Generate dft and the final semantic score.
        interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x[self.prop_set.columns[-1]]], axis=1)

        interacted_props['score'] = (interacted_props['npi'] / interacted_props['i']) * (
            np.log(interacted_props['n'] / interacted_props['dft']))

        # Return one score per property value.
        interacted_props.reset_index(inplace=True)
        interacted_props = interacted_props.set_index(self.prop_set.columns[-1])
        fav_prop = interacted_props['score'].to_dict()

        return fav_prop


    def __build_graph(self) -> nx.Graph:
        """
        Build a graph from train interactions and KG properties.

        The graph contains user, item, and property nodes. For example, if user 1
        interacted with Inception and Inception has Leonardo DiCaprio as an actor,
        the graph contains edges user 1 -> Inception and Inception -> Leonardo DiCaprio.
        :return: NetworkX graph with users, items, and property values
        """


        user_item_set = self.train_set.copy()
        edgelist = pd.DataFrame(columns=['origin', 'destination'])

        user_item_set['origin'] = ['U' + x for x in user_item_set.index.astype(str)]
        user_item_set['destination'] = ['I' + x for x in user_item_set[user_item_set.columns[0]].astype(str)]

        edgelist = pd.concat([edgelist, user_item_set[['origin', 'destination']]], ignore_index=True)

        item_prop_copy = self.prop_set.copy()
        item_prop_copy['origin'] = ['I' + x for x in item_prop_copy.index.astype(str)]
        item_prop_copy['destination'] = item_prop_copy[self.prop_set.columns[-1]]

        edgelist = pd.concat([edgelist, item_prop_copy[['origin', 'destination']]], ignore_index=True)

        G = nx.from_pandas_edgelist(edgelist, 'origin', 'destination')

        return G
