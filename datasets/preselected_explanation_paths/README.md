# Preselected Explanation Paths

Esta pasta guarda os caminhos de explicação pré-selecionados antes de rodar o LLM.

A ideia é simples: em vez de sortear os caminhos candidatos toda vez dentro do processo de otimização ou de explainability, o projeto pode gerar esse subconjunto uma vez só e depois reutilizar exatamente o mesmo CSV em execuções diferentes.

Isso ajuda em três pontos:

- reprodutibilidade
- comparação justa entre runs
- separação entre a etapa de amostragem dos caminhos e a etapa de escolha final feita pelo LLM

## O que esta pasta contém

A estrutura segue este padrão:

```text
preselected_explanation_paths/
  <algorithm>/
    <user_scope>/
      <selection_strategy>/
        recs_<N>_paths_<K>/
          seed_<S>/
            selected_paths.csv
            selected_paths_metadata.json
```

Exemplo real:

```text
preselected_explanation_paths/user_knn/optimization/random/recs_10_paths_10/seed_2026/selected_paths.csv
```

## Significado de cada nível

- `<algorithm>`: recomendador usado para buscar os caminhos, por exemplo `user_knn`, `item_knn`, `ncf` ou `bprmf`
- `<user_scope>`: conjunto de usuários usado na pré-seleção
  - `optimization`: corresponde a `train_val`
  - `explainability`: corresponde a `test`
- `<selection_strategy>`: estratégia usada para selecionar os caminhos candidatos
- `recs_<N>`: quantidade de recomendações consideradas por usuário
- `paths_<K>`: quantidade de caminhos candidatos escolhidos para cada recomendação
- `seed_<S>`: seed usada para a amostragem

## O que o `selected_paths.csv` representa

O `selected_paths.csv` não é a resposta final do LLM.

Ele contém os caminhos candidatos que foram separados para entrar no prompt. Depois, quando o LLM roda, ele escolhe um entre esses caminhos para cada recomendação.

Ou seja:

- este CSV representa a entrada candidata do processo
- o `responses.csv` representa a saída final escolhida pelo LLM

Cada linha do `selected_paths.csv` corresponde a um caminho candidato específico de uma recomendação específica de um usuário específico.

## Colunas do CSV

As colunas atuais são:

- `userId`: identificador do usuário
- `recommendation_order`: ordem da recomendação dentro do usuário, começando em `1`
- `recommended_item_id`: item recomendado ao usuário
- `interacted_item_id`: item do histórico do usuário que participa do caminho
- `selection_strategy`: estratégia usada para escolher os candidatos
- `selection_seed`: seed usada na amostragem
- `selection_order`: ordem do caminho dentro do conjunto de candidatos daquela recomendação
- `available_paths_for_recommendation`: total de caminhos disponíveis no arquivo original para essa recomendação
- `selected_paths_for_recommendation`: total de caminhos efetivamente mantidos no subconjunto
- `common_props`: propriedade ou atributo que conecta o item interagido ao item recomendado
- `interacted_item_name`: nome do item do histórico
- `recommended_item_name`: nome do item recomendado
- `selected_path`: caminho textual já formatado, no padrão usado pelo projeto
- `source_paths_file`: arquivo original de onde os caminhos foram lidos

## Como interpretar uma linha

Exemplo simplificado:

```text
userId=69
recommendation_order=1
recommended_item_id=2858
interacted_item_id=296
common_props=National Board of Review Award for Best Film
selected_path=American Beauty | Pulp Fiction -> National Board of Review Award for Best Film -> American Beauty
```

Leitura:

- para o usuário `69`
- na recomendação número `1`
- um dos caminhos candidatos para explicar o item `2858`
- foi construído a partir do item interagido `296`
- usando a propriedade `National Board of Review Award for Best Film`

## Relação com a seed

Quando a estratégia é `random`, a seed controla a amostragem dos caminhos.

Na prática, isso significa que, mantendo:

- o mesmo algoritmo
- o mesmo conjunto de usuários
- a mesma estratégia
- o mesmo número de recomendações
- o mesmo número de caminhos por recomendação
- a mesma seed

o CSV tende a ser reproduzido da mesma forma.

Se mudar o algoritmo, o arquivo base de caminhos muda também, então os resultados normalmente mudam.

## Arquivo de metadata

Cada pasta também contém um `selected_paths_metadata.json`.

Esse arquivo registra:

- os argumentos usados para gerar o CSV
- o escopo real de usuários resolvido pelo pipeline
- a pasta do split de usuários
- o número total de usuários processados
- o caminho do CSV gerado
- o tempo gasto para preparar os caminhos

## Como esses arquivos são usados no projeto

Os caminhos pré-selecionados podem ser consumidos por:

- `run_prompt_optimizer.py`
- `run_llm_explainability.py`

Os dois aceitam o argumento:

```bash
--selected_paths_input_path <caminho_para_selected_paths.csv>
```

Se esse argumento for passado, o pipeline reutiliza o CSV já pronto.

Se não for passado, o pipeline continua podendo gerar os caminhos internamente.

## Como gerar esses arquivos

O jeito recomendado é usar:

```bash
bash explainability-with-LLMs/bash/run_prepare_selected_paths.sh
```

Ou diretamente:

```bash
python explainability-with-LLMs/run_prepare_selected_paths.py \
  --datain ../datasets \
  --algorithm user_knn \
  --selection_strategy random \
  --num_recommendations 10 \
  --num_paths_per_recommendation 10 \
  --user_scope optimization \
  --seed 2026 \
  --out ../datasets/preselected_explanation_paths
```

## Resumo

Esta pasta guarda um estágio intermediário do pipeline:

- ainda não é a escolha final do LLM
- já não é o conjunto bruto completo de caminhos

Ela representa o subconjunto de caminhos candidatos que será oferecido ao modelo em execuções posteriores.
