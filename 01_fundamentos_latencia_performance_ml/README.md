# Aula 01 — Fundamentos de Latência e Performance em ML

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Compreender** os conceitos fundamentais de latência e throughput em sistemas de ML
2. **Medir** e **perfilar** o tempo de execução de cada etapa do pipeline de ML (pré-processamento, inferência, pós-processamento)
3. **Analisar** percentis de latência (P50, P95, P99) e sua importância em ambientes de produção
4. **Comparar** a complexidade computacional de diferentes algoritmos de ML e seu impacto na latência
5. **Avaliar** o efeito do tamanho de batch na relação latência-throughput
6. **Implementar** benchmarks reproduzíveis para pipelines de ML
7. **Identificar** gargalos de performance usando técnicas de profiling

## 📖 Teoria-Chave

### Latência em Sistemas de ML

A **latência** é o tempo total decorrido desde que uma requisição é recebida até que a resposta seja
entregue. Em sistemas de ML em produção, a latência end-to-end é composta por:

$$L_{total} = L_{rede} + L_{pre} + L_{inferencia} + L_{pos} + L_{serializacao}$$

Onde cada componente contribui para o tempo total de resposta. Conforme Crankshaw et al. (2017),
a latência de inferência é frequentemente o componente dominante, mas otimizações no
pré-processamento podem render ganhos significativos.

### Throughput vs. Latência

O **throughput** (vazão) mede o número de predições por unidade de tempo. Existe um trade-off
fundamental entre latência e throughput:

- **Batching** aumenta throughput mas pode aumentar latência individual
- **Paralelismo** pode melhorar ambos, mas com retornos decrescentes
- O **ponto ótimo** depende dos SLAs (Service Level Agreements) da aplicação

### Percentis de Latência

Em produção, a **latência média** é insuficiente. Métricas críticas incluem:

| Métrica | Descrição | Uso típico |
|---------|-----------|------------|
| P50 | Mediana | Experiência típica do usuário |
| P95 | 95º percentil | Performance sob carga |
| P99 | 99º percentil | Worst-case para SLAs |
| P99.9 | 99.9º percentil | Tail latency |

Conforme Dean & Barroso (2013), "The Tail at Scale", tail latencies são particularmente
problemáticas em sistemas distribuídos devido ao efeito de amplificação.

### Complexidade de Modelos

A complexidade computacional de diferentes algoritmos impacta diretamente a latência:

| Algoritmo | Complexidade de Inferência | Observação |
|-----------|---------------------------|------------|
| Regressão Logística | O(d) | Linear no nº de features |
| Random Forest | O(t · log n) | t = nº árvores |
| Gradient Boosting | O(t · profundidade) | Sequencial por natureza |
| SVM (kernel RBF) | O(n_sv · d) | n_sv = support vectors |

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| Python | ≥3.10 | Linguagem base |
| NumPy | ≥1.24.0 | Computação numérica |
| Pandas | ≥2.0.0 | Manipulação de dados |
| scikit-learn | ≥1.3.0 | Modelos de ML |
| Matplotlib | ≥3.7.0 | Visualizações |
| Seaborn | ≥0.12.0 | Gráficos estatísticos |
| psutil | ≥5.9.0 | Monitoramento de recursos |
| pytest | ≥7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
01_fundamentos_latencia_performance_ml/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb          # EDA e análise do dataset
│   ├── 02_treinamento.ipynb         # Treinamento e profiling
│   └── 03_avaliacao.ipynb           # Avaliação de latência e métricas
├── src/
│   ├── __init__.py                  # Inicialização do pacote
│   ├── data_preprocessing.py        # Pré-processamento de dados
│   ├── model.py                     # Definição dos modelos com benchmark
│   ├── training.py                  # Lógica de treinamento e tuning
│   ├── evaluation.py                # Métricas de performance e latência
│   └── utils.py                     # Utilitários (save/load, configs)
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Dataset sintético de classificação
│   ├── processed/                   # Dados processados (gerado em runtime)
│   └── README.md                    # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py        # Testes do pré-processamento
│   ├── test_model.py                # Testes do modelo
│   └── test_evaluation.py           # Testes de avaliação
└── outputs/
    ├── models/                      # Modelos serializados
    ├── figures/                     # Gráficos gerados
    └── logs/                        # Logs de benchmark
```

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
cd 01_fundamentos_latencia_performance_ml
pip install -r requirements.txt
```

### 2. Gerar Dataset (se necessário)

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor.generate_synthetic_dataset('data/raw/dataset.csv')"
```

### 3. Executar Notebooks (sequencialmente)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Executar via Scripts Python

```bash
# Pipeline completo
python -m src.training
```

## 📊 Datasets

O dataset utilizado é **sintético**, gerado para simular um problema de classificação binária
com features de diferentes escalas e correlações. Consulte [data/README.md](data/README.md) para
detalhes completos.

- **Instâncias:** 10.000
- **Features:** 15 numéricas
- **Target:** Binário (0/1), balanceamento ~50/50

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ -v --tb=short
```

## 📚 Referências

1. **Crankshaw, D. et al.** (2017). "Clipper: A Low-Latency Online Prediction Serving System."
   *NSDI '17*.
2. **Dean, J. & Barroso, L. A.** (2013). "The Tail at Scale."
   *Communications of the ACM*, 56(2), 74–80.
3. **Reddi, V. J. et al.** (2020). "MLPerf Inference Benchmark."
   *ISCA '20*.
4. **Sculley, D. et al.** (2015). "Hidden Technical Debt in Machine Learning Systems."
   *NeurIPS '15*.
5. **Li, M. et al.** (2014). "Scaling Distributed Machine Learning with the Parameter Server."
   *OSDI '14*.

## 🎥 Vídeos Relacionados

- **Vídeo 1:** Introdução à latência em sistemas de ML
- **Vídeo 2:** Demonstração prática — Profiling de modelos
- **Vídeo 3:** Hands-on — Comparação de algoritmos e batch sizes
- **Vídeo 4:** Análise de resultados e melhores práticas

## 🔗 Links Úteis

- [scikit-learn: Model Complexity & Performance](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html)
- [MLPerf Inference Benchmark](https://mlcommons.org/en/inference-datacenter-11/)
- [Google Cloud: ML Latency Best Practices](https://cloud.google.com/architecture/minimizing-predictive-serving-latency-in-machine-learning)
- [Python Profiling Tools](https://docs.python.org/3/library/profile.html)
