# Aula 06: Aceleração com Hardware: GPU, TPU e Aceleradores

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

---

## 🎯 Objetivos de Aprendizagem

Ao concluir esta aula, você será capaz de:

1. **Compreender a arquitetura de GPUs NVIDIA** — organização em Streaming Multiprocessors, modelo SIMT e hierarquia de memória (Nickolls & Dally, 2010)
2. **Aplicar o Modelo Roofline** para classificar operações como *compute-bound* ou *memory-bound* (Williams, Waterman & Patterson, 2009)
3. **Implementar Mixed Precision Training** com AMP (Automatic Mixed Precision) usando FP16/BF16 e loss scaling dinâmico (Micikevicius et al., 2018)
4. **Utilizar Gradient Checkpointing** para reduzir consumo de memória GPU em até 60% (Chen et al., 2016)
5. **Realizar profiling detalhado** com PyTorch Profiler e interpretar métricas de GPU (SM efficiency, memory throughput, occupancy)
6. **Comparar aceleradores** — GPU, TPU, NPU e AWS Inferentia — e selecionar hardware adequado para diferentes workloads

---

## 📖 Teoria-Chave

### Modelo Roofline

O Modelo Roofline (Williams et al., 2009) caracteriza performance de workloads via **intensidade aritmética**:

$$
\text{Performance}_{\text{atingível}} = \min\left(\text{Peak}_{\text{compute}},\ \text{BW}_{\text{mem}} \times \text{AI}\right)
$$

| Operação | Intensidade Aritmética | Classificação |
|---|---|---|
| Matmul (matrizes grandes) | ~100 FLOP/byte | Compute-bound |
| Convolução 3×3 | ~30 FLOP/byte | Balanceado |
| BatchNorm / Softmax | ~1-3 FLOP/byte | Memory-bound |

### Mixed Precision Training

Utiliza FP16/BF16 no forward/backward pass e FP32 para acumulação de gradientes, com **loss scaling** para evitar underflow:

$$
\text{Loss}_{\text{scaled}} = S \times \text{Loss}_{\text{original}} \qquad \nabla W_{\text{final}} = \frac{\nabla W_{\text{scaled}}}{S}
$$

### Gradient Checkpointing

Reduz memória de O(n) para O(√n) ao recomputar ativações intermediárias durante o backward pass, ao custo de ~33% de computação adicional (Chen et al., 2016).

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Propósito |
|---|---|---|
| PyTorch | ≥ 2.0.0 | Framework de deep learning com AMP |
| torchvision | ≥ 0.15.0 | Datasets e transforms |
| NumPy | ≥ 1.24.0 | Computação numérica |
| Pandas | ≥ 2.0.0 | Manipulação de dados tabulares |
| Matplotlib | ≥ 3.7.0 | Visualizações |
| Seaborn | ≥ 0.12.0 | Visualizações estatísticas |
| pytest | ≥ 7.4.0 | Testes unitários |
| psutil | ≥ 5.9.0 | Monitoramento de recursos |

---

## 📂 Estrutura de Arquivos

```
06_aceleracao_hardware_gpu_tpu/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb          # Exploração: hardware, roofline, benchmarks
│   ├── 02_treinamento.ipynb         # Treinamento: mixed precision, checkpointing
│   └── 03_avaliacao.ipynb           # Avaliação: profiling, métricas, comparações
├── src/
│   ├── __init__.py                  # Inicialização do pacote
│   ├── data_preprocessing.py        # Pré-processamento e geração de dados
│   ├── model.py                     # Vision Transformer com suporte a AMP
│   ├── training.py                  # Treinamento com mixed precision
│   ├── evaluation.py                # Métricas, profiling e roofline
│   └── utils.py                     # Utilitários: device, save/load, config
├── data/
│   ├── raw/                         # Dados brutos / gerados
│   │   └── dataset.csv              # Dataset sintético de classificação
│   ├── processed/                   # Dados processados
│   └── README.md                    # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py        # Testes de pré-processamento
│   ├── test_model.py                # Testes do modelo ViT
│   └── test_evaluation.py           # Testes de avaliação e métricas
└── outputs/
    ├── models/                      # Modelos salvos (.pt)
    ├── figures/                     # Gráficos gerados
    └── logs/                        # Logs de profiling
```

---

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
cd 06_aceleracao_hardware_gpu_tpu
pip install -r requirements.txt
```

> **Nota:** Para aceleração GPU, instale PyTorch com suporte CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Gerar Dataset Sintético

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_dataset()"
```

### 3. Executar Notebooks (sequencialmente)

```
notebooks/01_exploracao.ipynb   → Exploração de hardware e conceitos
notebooks/02_treinamento.ipynb  → Treinamento com mixed precision
notebooks/03_avaliacao.ipynb    → Profiling e avaliação de performance
```

### 4. Executar via Scripts Python

```bash
# Treinamento completo
python -m src.training

# Avaliação
python -m src.evaluation
```

---

## 📊 Datasets

Dataset sintético de classificação de imagens (formato tabular) para demonstração de técnicas de aceleração por hardware. Consulte [data/README.md](data/README.md) para detalhes completos.

- **Instâncias:** 10.000
- **Features:** 768 (patches de imagem 32×32 flattened)
- **Classes:** 10
- **Formato:** CSV

---

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes específicos
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v

# Com cobertura
pytest tests/ -v --tb=short
```

---

## 📚 Referências

Referências principais implementadas no código desta aula:

- **Micikevicius, P. et al.** (2018). Mixed Precision Training. *ICLR*. DOI: 10.48550/arXiv.1710.03740
- **Williams, S., Waterman, A., Patterson, D.** (2009). Roofline: An Insightful Visual Performance Model. *Communications of the ACM*, 52(4), 65-76.
- **Chen, T. et al.** (2016). Training Deep Nets with Sublinear Memory Cost. *arXiv*. DOI: 10.48550/arXiv.1604.06174
- **Dao, T.** (2023). FlashAttention-2: Faster Attention with Better Parallelism. *NeurIPS*. DOI: 10.48550/arXiv.2307.08691
- **Nickolls, J., Dally, W.J.** (2010). The GPU Computing Era. *IEEE Micro*, 30(2), 56-69.
- **Jouppi, N.P. et al.** (2017). In-Datacenter Performance Analysis of a Tensor Processing Unit. *ISCA*. DOI: 10.1145/3079856.3080246
- **Noune, B. et al.** (2022). FP8 Formats for Deep Learning. *arXiv*. DOI: 10.48550/arXiv.2209.05433

---

## 🎥 Vídeos Relacionados

- **Vídeo 1:** Fundamentos de arquitetura GPU e modelo SIMT
- **Vídeo 2:** Hierarquia de memória GPU e benchmarks de aceleradores
- **Vídeo 3:** Implementação prática de Mixed Precision Training e profiling
- **Vídeo 4:** Gradient Checkpointing e otimização de memória

---

## 🔗 Links Úteis

- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Ampere Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [Programming Massively Parallel Processors (Kirk & Hwu, 2022)](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0)
