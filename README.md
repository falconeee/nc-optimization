# nc-optimization

Otimização de redes neurais com meta-heurísticas de computação natural. O repositório reúne duas tarefas computacionais, uma em cada notebook. Nelas, algoritmos evolutivos e de enxame buscam hiperparâmetros e/ou subconjuntos de atributos para modelos neurais.

## Estrutura

```
nc-optimization/
├── notebooks/
│   ├── tc1_metaheuristicas_mlp_elm.ipynb   # TC1: DE/GA + MLP/ELM (Wine e Diabetes)
│   └── tc2_pso_autoencoder_mnist.ipynb     # TC2: PSO + Autoencoder (MNIST)
├── data/                                   # datasets locais (não versionados)
├── pyproject.toml                          # dependências (uv)
├── uv.lock
└── .python-version                         # Python 3.12
```

## Tarefa Computacional 1: seleção de atributos e hiperparâmetros

`notebooks/tc1_metaheuristicas_mlp_elm.ipynb`

Cada indivíduo codifica uma **máscara de atributos** e os **hiperparâmetros da rede**. O fitness é a acurácia média em validação cruzada de 3 folds no conjunto de treino. O modelo final é avaliado no conjunto de teste, separado de forma estratificada (75/25) e padronizado com z-score.

| Meta-heurística | Biblioteca | Modelo | Variáveis otimizadas |
|---|---|---|---|
| Differential Evolution (DE) | `scipy.optimize` | MLP (backpropagation) | atributos, nº de camadas ocultas, neurônios por camada, taxa de aprendizado |
| Differential Evolution (DE) | `scipy.optimize` | ELM | atributos, nº de neurônios ocultos |
| Algoritmo Genético (GA) | `pygad` | MLP (backpropagation) | atributos, nº de camadas ocultas, neurônios por camada, taxa de aprendizado |
| Algoritmo Genético (GA) | `pygad` | ELM | atributos, nº de neurônios ocultos |

O baseline é um `MLPClassifier` com todos os atributos e os parâmetros padrão.

**Datasets**

- **Wine**: 178 amostras, 13 atributos, 3 classes. Carregado via `sklearn.datasets.load_wine`.
- **Pima Indians Diabetes**: 768 amostras, 8 atributos, 2 classes. Lido de `data/diabetes.csv` (ver [Dados](#dados)).

**Resultados (acurácia no teste, execução salva no notebook)**

| Modelo | Wine | Diabetes |
|---|---|---|
| Baseline MLP | 0.9778 | 0.7135 |
| DE + MLP | 1.0000 | 0.7188 |
| DE + ELM | 0.9778 | 0.7240 |
| GA + MLP | 0.9556 | 0.7344 |
| GA + ELM | — ¹ | — ¹ |

> As meta-heurísticas não usam semente fixa. Por isso, novas execuções podem gerar valores diferentes.
>
> ¹ As células GA + ELM foram corrigidas: o espaço de busca é recriado na própria célula e o gene de neurônios usa o intervalo 10–1000. As saídas antigas foram apagadas. Execute essas células novamente para obter os resultados.

## Tarefa Computacional 2: PSO para arquitetura de autoencoder

`notebooks/tc2_pso_autoencoder_mnist.ipynb`

O PSO (`mealpy.PSO.OriginalPSO`) busca o número de neurônios das três camadas do encoder de um autoencoder denso aplicado ao MNIST. O espaço latente tem 32 dimensões e o decoder espelha o encoder.

| Variável | Intervalo |
|---|---|
| Neurônios camada 1 | 200 – 400 |
| Neurônios camada 2 | 100 – 200 |
| Neurônios camada 3 | 20 – 100 |

A avaliação de cada partícula segue três passos:

1. treina o autoencoder (MSE, Adam com lr = 1e-4);
2. congela o encoder e treina um classificador (`Dense(64) → Dense(10, softmax)`) sobre o espaço latente;
3. retorna `1 − acurácia` no conjunto de teste.

Depois disso, o notebook retreina o melhor modelo com mais épocas, reporta acurácia, precisão, recall e F1 (macro) e plota as imagens originais ao lado das reconstruídas.

> A execução é custosa: cada partícula treina duas redes no MNIST completo. Com `epoch=25` e `pop_size=15`, use GPU ou reduza esses valores.

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # cria .venv com Python 3.12 e instala as dependências
uv run jupyter lab           # ou abra os notebooks no VS Code usando o kernel .venv
```

Execute os notebooks a partir da pasta `notebooks/`, porque o caminho do dataset é relativo (`../data/diabetes.csv`).

### Dados

O dataset de diabetes não é versionado. Baixe o *Pima Indians Diabetes Database* (por exemplo, [no Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)) e salve o arquivo como `data/diabetes.csv`. Ele deve conter as colunas `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

Os datasets Wine e MNIST são baixados automaticamente pelo scikit-learn e pelo Keras.

## Principais dependências

`scikit-learn`, `scipy`, `pygad`, `tfelm` (ELM em TensorFlow), `mealpy==2.5.3`, `tensorflow`/`keras`, `numpy`, `pandas`, `matplotlib`. As versões estão em `pyproject.toml` e `uv.lock`.
