# Diretório contendo todos os datasets utilizados no projeto

datasets/
├──explanation_paths     # Pasta contendo, para cada algoritmo na versão otimizada, com K=20, todos os caminhos de explicação
│
│
├──ml-latest-small/       # Pasta contendo os arquivos base do MovieLens latest small
│
├──preprocessing/         # Pré-processamento construção do recommender_train_test_oficial e do recommender_train_validation 
│ 
├── recommendation_files/
│   ├── recommendation_lists/
│   │   ├── bprmf/         # Contém arquivos de recomendações para diferentes valores de K
│   │   ├── itemknn/       # (K=1,5,10,20,50,100,200)
│   │   ├── ncf/           # Parâmetros default e otimizados
│   │   └── userknn/
│   │
│   └── recommendation_metrics/
│       ├── bprmf/         # Métricas calculadas a partir das listas de recomendação
│       ├── itemknn/
│       ├── ncf/
│       └── userknn/
│
├── recommender_train_test_oficial/
│   ├── train.dat          # Dataset completo de treino
│   └── test.dat           # Dataset completo de teste
│
├──recommender_train_validation/      # Dataset de validação para realizar a otimização
│   ├── opt_train.csv               
│   └── opt_validation.csv  
|
├──recs_to_expl_format/                # Pasta para guardar infos para a explicação!!!
│   ├── recommendation_lists/          # Recomendações com nomes dos filmes
│   ├── main.py                        
│   └── train_llm.csv                 
│
└── README_datasets.md

