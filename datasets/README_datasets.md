# Diretório contendo todos os datasets utilizados no projeto

datasets/
├── recommendation_files/
│   ├── recommendation_lists/
│   │   ├── bprmf/         # Contém arquivos de recomendações para diferentes valores de K
│   │   ├── itemknn/       # (K=1,5,10,20,50,100,200)
│   │   ├── ncf/
│   │   └── userknn/
│   │
│   └── recommendation_metrics/
│       ├── bprmf/         # Métricas calculadas a partir das listas de recomendação
│       ├── itemknn/
│       ├── ncf/
│       └── userknn/
│
├── train_test_oficial/
│   ├── train.dat          # Dataset completo de treino
│   └── test.dat           # Dataset completo de teste
│
└── README_datasets.md

