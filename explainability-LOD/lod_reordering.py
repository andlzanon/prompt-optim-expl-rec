import numpy as np
import pandas as pd
import networkx as nx


class LODPersonalizedReordering(object):

    def __init__(self, train_file: str, recs_file: str, prop_path: str, prop_cols: list, cols_used: list, user_list: list):
        """
        LOD personalized reordering class does not extend the base recommender because it does not recommend items.
        Instead it reorders them based on the best explanations to the item. Therefore, initially a semantic profile
        (favorite properties) of the user is obtained based on a policy that will use 'all' items of the historic,
        'random' items 'last' items or 'first' n_moives percentage of total from the user.
        :param train_file: train file in which the recommendations of where computed
        :param recs_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param cols_used: columns used from the test and train set
        :param user_list: list to represent which users to run the explod algorithm
        """
        self.cols_used = cols_used

        self.train_file = train_file
        self.train_set = pd.read_csv(self.train_file, header=None)
        self.train_set.columns = self.cols_used
        # Filtrar dataset para apenas usuários do user_list
        self.train_set = self.train_set[self.train_set["user_id"].isin(user_list)]
        self.train_set = self.train_set.set_index(self.cols_used[0])

        self.output_cols = ['user_id', 'item_id', 'score']
        self.recs_file = recs_file
        self.recs_set = pd.read_csv(self.recs_file, header=None)
        self.recs_set.columns = self.output_cols
        # Filtrar dataset para apenas usuários do user_list
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
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc)
        are ordered by a score that is calculated as:
            score = (npi/i) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :return: dictionary with properties' values as keys and scores as values
        """

        # create npi, i and n columns
        interacted_props = self.prop_set.loc[self.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby(self.prop_set.columns[-1])[self.prop_set.columns[-1]].transform('count')
        interacted_props['i'] = len(historic)
        interacted_props['n'] = len(self.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        items_per_obj = self.prop_set.reset_index().drop_duplicates(subset=[self.prop_set.columns[0], self.prop_set.columns[-1]]).set_index(
            self.prop_set.columns[-1])
        df_dict = items_per_obj.index.value_counts().to_dict()

        # generate the dft column based on items per property and score column base on all new created columns
        interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x[self.prop_set.columns[-1]]], axis=1)

        interacted_props['score'] = (interacted_props['npi'] / interacted_props['i']) * (
            np.log(interacted_props['n'] / interacted_props['dft']))

        # generate the dict
        interacted_props.reset_index(inplace=True)
        interacted_props = interacted_props.set_index(self.prop_set.columns[-1])
        fav_prop = interacted_props['score'].to_dict()

        return fav_prop


    def __build_graph(self) -> nx.Graph:
        """
        Build a graph with the information from the test set and the wikidata or dbpedia set to create
        a graph with users, items and property nodes e.g.: user 1 interacted the item Inception with Di Caprio as an actor
        therefore, on the graph there is a three nodes, one for each entity (user, item and actor/property) and three
        edges, one connecting user to item and other item to property: user 1 --> Inception --> Di Caprio
        :return: networkx graph with users, items and properties from the dbpedia or wikidata
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