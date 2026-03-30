# Aula 4 — Otimização de Modelos: Pruning e Quantização

Pipeline completo de compressão de redes neurais com técnicas de **pruning**,
**quantização INT8** e **knowledge distillation**, aplicadas a um modelo de
classificação de imagens (CIFAR-10) construído em PyTorch.

---

## 🎯 Objetivos de Aprendizagem

Ao final desta aula o aluno será capaz de:

1. Compreender a **Hipótese da Lottery Ticket** (Frankle & Carlin, 2019) e
   por que redes neurais podem ser comprimidas sem perda significativa.
2. Aplicar **pruning baseado em magnitude** (Han et al., 2015) — tanto
   não-estruturado quanto estruturado — a camadas lineares e convolucionais.
3. Implementar **quantização dinâmica e estática INT8** usando
   `torch.quantization`, incluindo calibração e Quantization-Aware Training.
4. Construir uma função de perda de **Knowledge Distillation** (Hinton et al.,
   2015) com soft labels e temperatura ajustável.
5. Montar um **pipeline integrado de compressão** (pruning → fine-tune →
   quantização) e avaliar trade-offs de acurácia, latência e memória.

---

## 📖 Teoria-Chave

### Lottery Ticket Hypothesis

> Uma rede neural densa, inicializada aleatoriamente, contém uma sub-rede
> esparsa que, quando treinada isoladamente desde a mesma inicialização, pode
> atingir acurácia comparável à rede completa (Frankle & Carlin, 2019).

### Critérios de Importância para Pruning

| Critério | Fórmula / Ideia |
|----------|-----------------|
| Magnitude-based | Remove pesos com \|w\| < τ |
| Gradient-based | Remove pesos com \|∂L/∂w\| pequeno |
| Taylor 1ª ordem | ΔL(wᵢ) ≈ \|∂L/∂wᵢ · wᵢ\| (Molchanov et al., 2017) |

### Quantização Uniforme

$$q = \text{round}\left(\frac{x - z}{s}\right), \quad \hat{x} = s \cdot q + z$$

### Knowledge Distillation

$$\mathcal{L} = \alpha \cdot T^2 \cdot D_{KL}(p^T_{student} \| p^T_{teacher}) + (1 - \alpha) \cdot \mathcal{L}_{CE}(p_{student}, y)$$

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão mínima | Uso |
|------------|---------------|-----|
| PyTorch | 2.0.0 | Modelos, pruning, quantização |
| torchvision | 0.15.0 | Dataset CIFAR-10, transforms |
| NumPy | 1.24.0 | Operações numéricas |
| Pandas | 2.0.0 | Tabulação de métricas |
| Matplotlib | 3.7.0 | Gráficos |
| Seaborn | 0.12.0 | Visualizações estatísticas |
| pytest | 7.4.0 | Testes unitários |

---

## 📂 Estrutura de Arquivos

```
04_otimizacao_modelos_pruning_quantizacao/
├── README.md                   ← este arquivo
├── requirements.txt            ← dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb     ← EDA do modelo base + análise de pesos
│   ├── 02_treinamento.ipynb    ← pruning, quantização, distillation
│   └── 03_avaliacao.ipynb      ← benchmarks de latência/memória/acurácia
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   ← carregamento CIFAR-10 e transforms
│   ├── model.py                ← CNN base + variantes comprimidas
│   ├── training.py             ← treinamento, fine-tune, distillation
│   ├── evaluation.py           ← métricas de acurácia, latência, memória
│   └── utils.py                ← serialização, logging, helpers
├── data/
│   ├── raw/                    ← (CIFAR-10 auto-download via torchvision)
│   ├── processed/
│   └── README.md
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_evaluation.py
└── outputs/
    ├── models/
    ├── figures/
    └── logs/
```

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
cd 04_otimizacao_modelos_pruning_quantizacao
pip install -r requirements.txt
```

### 2. Executar notebooks (ordem recomendada)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 3. Executar via scripts

```bash
# Treinar modelo base
python -m src.training

# Avaliar modelos comprimidos
python -m src.evaluation
```

---

## 📊 Datasets

O dataset utilizado é o **CIFAR-10** (Krizhevsky, 2009), baixado
automaticamente via `torchvision.datasets.CIFAR10`. Veja detalhes em
[data/README.md](data/README.md).

---

## 🧪 Testes

```bash
# Na raiz da aula:
cd 04_otimizacao_modelos_pruning_quantizacao
pytest tests/ -v
```

---

## 📚 Referências (código ↔ documento 04)

- **Han, S. et al. (2015).** Learning both Weights and Connections for
  Efficient Neural Networks. *NeurIPS*. → `src/model.py` (pruning)
- **Frankle, J. & Carlin, M. (2019).** The Lottery Ticket Hypothesis.
  *ICLR*. → Notebook 01 (análise de esparsidade)
- **Hinton, G. et al. (2015).** Distilling the Knowledge in a Neural
  Network. *NIPS Workshop*. → `src/training.py` (distillation loss)
- **Jacob, B. et al. (2018).** Quantization and Training of Neural Networks
  for Efficient Integer-Arithmetic-Only Inference. *CVPR*. →
  `src/model.py` (quantização)
- **Molchanov, P. et al. (2017).** Pruning CNNs for Resource Efficient
  Inference. *ICLR*. → `src/model.py` (Taylor pruning)
- **Blalock, D. et al. (2020).** What is the State of Neural Network
  Pruning? *MLSys*. → Notebook 03 (comparações)

---

## 🎥 Vídeos Relacionados

| Vídeo | Conteúdo |
|-------|----------|
| Vídeo 1 | Introdução ao pipeline de compressão |
| Vídeo 2 | Identificação de winning tickets e esparsidade iterativa |
| Vídeo 3 | Knowledge Distillation aplicada a BERT / classificadores |
| Vídeo 4 | Pipeline combinado com deployment |

---

## 🔗 Links Úteis

- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [Intel Neural Compressor](https://github.com/intel/neural-compressor)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/)

---

*Material elaborado para a disciplina de Engenharia de Machine Learning — FIAP Pós-Tech (2026)*
