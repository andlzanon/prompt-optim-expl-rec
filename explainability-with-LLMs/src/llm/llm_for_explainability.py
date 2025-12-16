import torch
from tqdm import tqdm
from src.llm.token_id import get_token
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    
MODEL_ID = {
    'Llama3.1-I' : 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Llama3.1'   : 'meta-llama/Meta-Llama-3.1-8B'
}

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

class LLM():
    def __init__(
                    self, 
                    llm_method: str = '',
                    seed: int = 2025
                ):
        
        self.seed = seed
        self.llm_method = llm_method
        self.model_name = MODEL_ID[self.llm_method]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_prompt(self):
        self.system_prompt = 'You are an explainable recommender system. Your task is to explain briefly and clearly why each recommended movie is suitable for the user. Focus on aspects such as genre, themes, cast, director, and overall tone. Keep explanations short and factual.'        
        self.prompt = self.create_base_prompt()
        
    def set_model(self):
        self.set_prompt()

        self.token_access = get_token()

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          torch_dtype="auto", 
                                                          device_map=self.device, 
                                                          offload_buffers=True, 
                                                          token=self.token_access, 
                                                          trust_remote_code=True,
                                                          use_cache=False,
                                                          quantization_config=quantization_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       torch_dtype="auto", 
                                                       device_map=self.device, 
                                                       offload_buffers=True, 
                                                       token=self.token_access, 
                                                       use_safetensors=True, 
                                                       trust_remote_code=True) 
         
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def create_base_prompt(self):
        prompt = [{'role': 'system', 'content': self.system_prompt}]
        return prompt

    def create_prompt_for_explainability(self, watched_movies, recommended_movies, number_of_recommended_movies):
        prompt = self.prompt.copy()
        prompt.append({
            'role': 'user',
            'content': f"User has watched: {watched_movies}\n\nRecommended movies: {recommended_movies}\n\nFor each one of the {number_of_recommended_movies} recommended movies, explain briefly and clearly why it is suitable for the user. "
        })
        return prompt     

    def request_model(self, prompt):
        set_seed(self.seed)

        inputs = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

        outputs = self.model.generate(inputs['input_ids'],
                                      attention_mask = inputs['attention_mask'],
                                      max_new_tokens=1000,
                                      eos_token_id=self.terminators,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      do_sample=True,
                                      # temperature=0.7,
                                      # top_p=0.9,
                                      use_cache=True)
        
        response_model = outputs[0][inputs['input_ids'].shape[-1]:]
        response_model = self.tokenizer.decode(response_model, skip_special_tokens=True)        
        return response_model

    def generate_explanations(self, users, df_recommendations, user_movies_dict_train):    
        responses = {}

        for user_id in tqdm(users, desc="Generating explanations", ascii=True):
            df_user = df_recommendations[df_recommendations.userId == user_id]
            recommended_movies = df_user.iloc[0]['response'] 
            number_of_recommended_movies = len(recommended_movies)
            recommended_movies = ', '.join(recommended_movies)
            
            watched_movies = user_movies_dict_train[user_id]
            watched_movies = ', '.join(watched_movies).lower()

            prompt = self.create_prompt_for_explainability(watched_movies, recommended_movies, number_of_recommended_movies)
            response = self.request_model(prompt)

            responses[user_id] = response

            print(len(df_user.iloc[0]['response'] ), df_user.iloc[0]['response'] )
            print(response)

            torch.cuda.empty_cache()
        
        return responses