import os
import json
import pandas as pd
import ast

def str2bool(x):
    if x.lower() in ['y', 'yes', 's', 'sim', '1', 'abacaxi']:
        return True
    return False

def check_if_out_file_exists(args):
    if os.path.exists(args.outfilename+'_time'+'.json'):
        print("Out dir already exist!")
        exit()

def read_movie_titles(inputdir):
    return pd.read_csv(f"{inputdir}/movies.csv")

def read_train_test(inputdir):
    trainset = pd.read_csv(f'{inputdir}/train_llm.csv')
    testset  = pd.read_csv(f'{inputdir}/test_llm.csv')
    return trainset, testset

def read_recommendations(inputdir):
    df = pd.read_csv(inputdir)
    df['response'] = df['response'].apply(ast.literal_eval)
    return df

def save_file(save_dir, info):
    with open(save_dir+'.json', 'w') as arquivo_json:
        json.dump(info, arquivo_json, indent=4)

def prepare_explainability_inputs(args):
    trainset, testset = read_train_test(args.inputdir)

    all_titles = read_movie_titles(args.inputdir)
    movie_dict = {title.lower(): title for title in all_titles.title}
    all_lower_movie_set = set(movie_dict.keys())

    users = testset.userId.unique().tolist()
    trainset = trainset[trainset.userId.isin(users)]

    user_movies_dict_train = (
        trainset.groupby('userId')['title']
        .apply(list)
        .to_dict()
    )

    user_movies_dict_test = (
        testset.groupby('userId')['title']
        .apply(list)
        .to_dict()
    )

    df_recommendations = read_recommendations(args.inputdir_recommendation)

    return {
        "user_movies_dict_train": user_movies_dict_train,
        "df_recommendations": df_recommendations,
    }