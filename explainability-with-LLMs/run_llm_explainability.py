import time
import pandas as pd
import os

from src.utils.args import args_llm
from src.llm.llm_for_explainability import LLM
from src.utils.geral import prepare_explainability_inputs, save_file

if __name__ == '__main__':

    args, info = args_llm()
    
    # Load data
    data = prepare_explainability_inputs(args)
    user_movies_dict_train = data["user_movies_dict_train"]
    user_movies_dict_train = dict(list(user_movies_dict_train.items())[:10])
    df_recommendations = data["df_recommendations"]
    users = list(user_movies_dict_train.keys())

    # Create model
    llm = LLM(llm_method = args.llm_method, seed=args.seed)
    llm.set_model()
        
    # Generate explainability
    print("Explainability!")
    start_time = time.time()
    user_explanations = llm.generate_explanations(users, df_recommendations, user_movies_dict_train)
    end_time = time.time()

    # Save data
    info["time_to_explain"] = end_time - start_time
    info["time_to_explain_avg"] = (end_time - start_time) / len(user_movies_dict_train)
    # info["system_prompt"] = llm.system_prompt
    # info["user_prompt"] = llm.user

    save_file(args.outfilename + '_time', info)

    output_csv = args.outfilename + '.csv'
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["userId", "explanation"]).to_csv(output_csv, index=False)

    for user_id, explanation in user_explanations.items():
        df_temp = pd.DataFrame([{
            "userId": user_id,
            "explanation": explanation
        }])
        df_temp.to_csv(output_csv, mode="a", header=False, index=False)

    print(f"Explanations saved to {output_csv} and metadata saved to JSON.")