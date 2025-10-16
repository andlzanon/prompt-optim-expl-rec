from caserec.recommenders.item_recommendation.bprmf import BprMF
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
import pandas as pd
import numpy as np

np.asfarray = np.asarray


def do_recommendations(algorithm_name):

    # available algorithms:

        # bprmf


    datasets_path = "datasets"

    train_file_path = datasets_path + "/train_test_oficial/train_llm_oficial.csv"
    test_file_path = datasets_path + "/train_test_oficial/test_llm_oficial.csv"

    output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_{algorithm_name}.csv"

    match algorithm_name:
        case "bprmf":
            print(f"Started to train: {algorithm_name}")
            BprMF(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()

        case "item_knn":
            print(f"Started to train: {algorithm_name}")
            ItemKNN(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()

        case "all":

            output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_bprmf.csv"

            print("RUNNING ALL: ")
            print(f"Started to train: bprmf")
            BprMF(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()
            print()

            output_file_path = datasets_path + f"/recommendation/recommendation_lists/recs_item_knn.csv"
            print(f"Started to train: item_knn")
            ItemKNN(train_file=train_file_path, test_file=test_file_path, output_file=output_file_path).compute()
            print()

            print(f"\nRecs successfully saved!")
            print(f"Recs file_path: {datasets_path}/recommendation_files/")
            return

        case _:
            print("Algorithm not found!")
            print("PLEASE, ENTER A VALID ALGORITHM")
            return

    print(f"\nRecs successfully saved!")
    print(f"Recs file_path: {output_file_path}")
    return
    
# do_recommendations(algorithm_name="bprmf")
# do_recommendations(algorithm_name="item_knn")
do_recommendations(algorithm_name="all")
