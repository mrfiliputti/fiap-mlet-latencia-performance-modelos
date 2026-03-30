# Aula 5 — Transfer Learning Eficiente e Adaptação de Modelos

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, você será capaz de:

1. **Compreender** os fundamentos de Transfer Learning e a hipótese de subespaços de baixa dimensionalidade (Aghajanyan et al., 2021)
2. **Implementar** fine-tuning eficiente com LoRA (Hu et al., 2022) e QLoRA (Dettmers et al., 2023)
3. **Comparar** técnicas PEFT: LoRA, Adapters, Prefix Tuning e Prompt Tuning
4. **Exportar** modelos otimizados para ONNX com otimizações de grafo computacional
5. **Avaliar** trade-offs entre parâmetros treináveis, memória GPU, latência e desempenho
6. **Aplicar** estratégias de warm-up e caching para serving em produção

## 📖 Teoria-Chave

### Hipótese de Subespaços de Baixa Dimensionalidade

O fine-tuning efetivo de modelos pré-treinados ocorre em subespaços de dimensionalidade intrínseca
muito menor que o espaço total de parâmetros. Para muitas tarefas de NLP, esta dimensão é de
apenas algumas centenas a poucos milhares, mesmo para modelos com bilhões de parâmetros
(Li et al., 2018; Aghajanyan et al., 2021).

### Low-Rank Adaptation (LoRA)

LoRA injeta uma atualização de baixo rank parametrizada como:

```
h = W₀x + ΔWx = W₀x + BAx
```

onde B ∈ ℝ^(d×r) e A ∈ ℝ^(r×k) com r ≪ min(d, k). Após treinamento, as matrizes BA
podem ser mescladas com W₀ sem overhead de inferência (Hu et al., 2022).

### QLoRA

QLoRA combina LoRA com quantização NF4 de 4 bits, possibilitando fine-tuning de modelos
com bilhões de parâmetros em GPUs de consumo (Dettmers et al., 2023).

### Otimização de Grafos Computacionais

Ferramentas como ONNX Runtime, TorchScript e torch.compile aplicam otimizações de
constant folding, fusão de operadores e compilação JIT para inferência de baixa latência.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Descrição |
|-----------|--------|-----------|
| PyTorch | ≥ 2.0.0 | Framework de Deep Learning |
| Transformers | ≥ 4.35.0 | Modelos pré-treinados (Hugging Face) |
| PEFT | ≥ 0.7.0 | Parameter-Efficient Fine-Tuning |
| bitsandbytes | ≥ 0.41.0 | Quantização 4-bit / 8-bit |
| ONNX Runtime | ≥ 1.16.0 | Otimização e inferência de grafos |
| scikit-learn | ≥ 1.3.0 | Métricas e avaliação |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Visualizações |
| pytest | ≥ 7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
05_transfer_learning_eficiente/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb          # EDA + análise de modelos pré-treinados
│   ├── 02_treinamento.ipynb         # Fine-tuning com LoRA/PEFT
│   └── 03_avaliacao.ipynb           # Métricas, benchmarks e ONNX
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Pipeline de pré-processamento textual
│   ├── model.py                     # Wrapper PEFT (LoRA, QLoRA, Adapters)
│   ├── training.py                  # Treinamento com Hugging Face Trainer
│   ├── evaluation.py                # Métricas e benchmarking de latência
│   └── utils.py                     # Utilitários (save/load, config, timer)
├── data/
│   ├── raw/dataset.csv              # Dataset de classificação de domínio
│   ├── processed/                   # Dados tokenizados (gerado em runtime)
│   └── README.md                    # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py        # Testes do DataPreprocessor
│   ├── test_model.py                # Testes do PEFTModelWrapper
│   └── test_evaluation.py           # Testes de métricas
└── outputs/
    ├── models/                      # Checkpoints e adaptadores salvos
    ├── figures/                      # Gráficos gerados
    └── logs/                         # Logs de treinamento
```

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
cd 05_transfer_learning_eficiente
pip install -r requirements.txt
```

### 2. Gerar Dataset Sintético

```bash
python -c "from src.data_preprocessing import generate_synthetic_dataset; generate_synthetic_dataset(output_path='data/raw/dataset.csv')"
```

### 3. Execução dos Notebooks (sequencial)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Execução via Scripts Python

```python
from src.data_preprocessing import DataPreprocessor, generate_synthetic_dataset
from src.model import PEFTModelWrapper
from src.training import train_with_peft
from src.evaluation import calculate_metrics, benchmark_latency

# Gerar dados
df = generate_synthetic_dataset(n_samples=5000, output_path="data/raw/dataset.csv")

# Pré-processar
preprocessor = DataPreprocessor(max_length=128)
df = preprocessor.load_data("data/raw/dataset.csv")
df = preprocessor.clean_data()
df = preprocessor.prepare_features()
train_df, val_df, test_df = preprocessor.split_data()

# Treinar com LoRA
model_wrapper = PEFTModelWrapper(model_name="distilbert-base-uncased", num_labels=5)
model_wrapper.load_base_model()
model_wrapper.apply_lora(r=16, lora_alpha=32)

# Avaliar
metrics = calculate_metrics(y_true, y_pred, label_names=list(preprocessor.label_map.keys()))
```

## 📊 Datasets

Dataset sintético de classificação de domínio textual com 5 categorias:
jurídico, tecnologia, financeiro, saúde, educação.

Detalhes completos em [data/README.md](data/README.md).

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes específicos
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v
```

## 📚 Referências

- **Hu, E.J. et al. (2022)**. LoRA: Low-Rank Adaptation of Large Language Models. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Dettmers, T. et al. (2023)**. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Aghajanyan, A. et al. (2021)**. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. ACL.
- **Houlsby, N. et al. (2019)**. Parameter-Efficient Transfer Learning for NLP. ICML.
- **Lester, B. et al. (2021)**. The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP.
- **Li, X.L. & Liang, P. (2021)**. Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.
- **Pan, S.J. & Yang, Q. (2010)**. A Survey on Transfer Learning. IEEE TKDE, 22(10).

## 🎥 Vídeos Relacionados

- **Vídeo 1**: Introdução ao Transfer Learning e motivação para PEFT
- **Vídeo 2**: Decomposição de baixo rank e visualização do espaço de parâmetros
- **Vídeo 3**: Pipeline completo de fine-tuning QLoRA com análise de memória e performance
- **Vídeo 4**: Validação numérica e troubleshooting de exportação ONNX

## 🔗 Links Úteis

- [Hugging Face PEFT](https://github.com/huggingface/peft) — Biblioteca oficial de PEFT
- [ONNX Runtime](https://onnxruntime.ai/) — Runtime de inferência otimizada
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — Quantização para PyTorch
- [DeepLearning.AI – Efficient Fine-Tuning](https://www.deeplearning.ai/) — Curso recomendado
- [AdapterHub](https://adapterhub.ml/) — Framework de adaptadores (Pfeiffer et al., 2020)
