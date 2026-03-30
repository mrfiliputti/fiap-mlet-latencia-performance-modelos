# Aula 02 — Desafios de Performance em NLP e Áudio

> Disciplina: Engenharia de Machine Learning — FIAP PósTech (Mestrado)

---

## 🎯 Objetivos de Aprendizagem

Ao final desta aula o aluno será capaz de:

1. **Compreender** a complexidade O(n²) do mecanismo de self-attention e identificar seus gargalos computacionais e de memória.
2. **Implementar** o mecanismo de self-attention vanilla (scaled dot-product) e compará-lo com variantes eficientes.
3. **Explicar** como Flash Attention substitui o paradigma de redução de FLOPs por IO-awareness, explorando a hierarquia de memória das GPUs.
4. **Classificar** abordagens de atenção eficiente (Sparse, Linformer, Performer, Local) usando a taxonomia de Tay et al. (2022).
5. **Implementar** KV-Cache para geração autoregressiva e analisar seus trade-offs de memória vs. computação.
6. **Extrair** representações de áudio (STFT, espectrogramas Mel, MFCCs) e compreender os parâmetros relevantes para modelos como Whisper.
7. **Avaliar** estratégias de batching (padding, packing, sorted batching) e seu impacto no throughput.
8. **Projetar** pipelines de transcrição de áudio otimizados com quantização e VAD.

---

## 📖 Teoria-Chave

### Self-Attention — Complexidade Quadrática

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

A matriz de atenção $QK^T$ possui dimensões $n \times n$, implicando complexidade de tempo $O(n^2 \cdot d)$ e **memória** $O(n^2)$. (Vaswani et al., 2017)

### Flash Attention — IO-Awareness

Em vez de minimizar FLOPs, Flash Attention minimiza acessos à HBM usando **tiling** e **recomputação**:

$$
O_i = \sum_j \frac{\exp(Q_i K_j^T - m_i)}{\sum_{j'} \exp(Q_i K_{j'}^T - m_i)} V_j
$$

Reduz memória de $O(n^2)$ para $O(n)$ com speedups de 2-4× em inferência. (Dao et al., 2022)

### Processamento de Áudio — Espectrograma Mel

Escala Mel perceptualmente motivada:

$$
m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)
$$

Whisper utiliza 80 filtros Mel, `n_fft=400` (25 ms @ 16 kHz), `hop_length=160` (10 ms). (Radford et al., 2023)

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca        | Versão   | Uso                                  |
|-------------------|----------|--------------------------------------|
| Python            | ≥ 3.10   | Linguagem base                       |
| NumPy             | ≥ 1.24   | Álgebra linear, geração de dados     |
| Pandas            | ≥ 2.0    | Manipulação tabular                  |
| PyTorch           | ≥ 2.0    | Tensores, autograd, modelos          |
| Librosa           | ≥ 0.10   | Processamento de áudio               |
| SoundFile         | ≥ 0.12   | Leitura/escrita de arquivos de áudio |
| Matplotlib        | ≥ 3.7    | Visualizações                        |
| Seaborn           | ≥ 0.12   | Gráficos estatísticos                |
| scikit-learn      | ≥ 1.3    | Métricas, utilidades                 |
| pytest            | ≥ 7.4    | Testes unitários                     |
| Jupyter           | ≥ 1.0    | Notebooks interativos                |

---

## 📂 Estrutura de Arquivos

```
02_desafios_performance_nlp_audio/
├── README.md                        # ← Você está aqui
├── requirements.txt                 # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb          # EDA: atenção, sinais de áudio
│   ├── 02_treinamento.ipynb         # Benchmark de mecanismos de atenção
│   └── 03_avaliacao.ipynb           # Métricas de latência e memória
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Tokenização, padding, Mel features
│   ├── model.py                     # Attention vanilla, eficiente, KV-Cache
│   ├── training.py                  # Benchmark e treinamento
│   ├── evaluation.py                # Métricas de performance
│   └── utils.py                     # Timers, I/O, configuração
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Sequências sintéticas + metadados
│   ├── processed/
│   └── README.md                    # Documentação do dataset
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
cd 02_desafios_performance_nlp_audio
pip install -r requirements.txt
```

### 2. Gerar dataset (se necessário)

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_dataset()"
```

### 3. Executar notebooks (ordem recomendada)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Executar via scripts Python

```python
from src.data_preprocessing import DataPreprocessor
from src.model import VanillaAttention, LinearAttention
from src.training import benchmark_attention_mechanisms
from src.evaluation import PerformanceEvaluator

# Pipeline completo
preprocessor = DataPreprocessor()
data = preprocessor.load_data("data/raw/dataset.csv")
results = benchmark_attention_mechanisms(seq_lengths=[64, 128, 256, 512])
evaluator = PerformanceEvaluator()
evaluator.plot_scaling_comparison(results)
```

---

## 📊 Datasets

Dataset sintético com sequências de texto tokenizadas e sinais de áudio simulados, projetado para demonstrar os conceitos de complexidade de atenção e processamento de áudio discutidos na aula.

→ Detalhes completos em [`data/README.md`](data/README.md)

---

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes por módulo
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v

# Com cobertura
pytest tests/ -v --tb=short
```

---

## 📚 Referências

Referências principais utilizadas no código (extraídas do material teórico da aula):

1. **Vaswani, A. et al.** (2017). Attention Is All You Need. *NeurIPS*, v. 30, p. 5998-6008.
2. **Dao, T. et al.** (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*, v. 35.
3. **Dao, T.** (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *arXiv:2307.08691*.
4. **Tay, Y. et al.** (2022). Efficient Transformers: A Survey. *ACM Computing Surveys*, v. 55, n. 6.
5. **Radford, A. et al.** (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*.
6. **Baevski, A. et al.** (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*, v. 33.
7. **Shazeer, N.** (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv:1911.02150*.
8. **Katharopoulos, A. et al.** (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML*.

---

## 🎥 Vídeos Relacionados

- **Vídeo 1** — Configuração do ambiente e overview da aula
- **Vídeo 2** — Pipeline de execução de Transformers e identificação de gargalos; Visualização interativa das transformações de áudio
- **Vídeo 3** — Pipeline completo de transcrição de áudio com Whisper otimizado
- **Vídeo 4** — Benchmark comparativo: Flash Attention vs. Vanilla Attention

---

## 🔗 Links Úteis

- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Flash Attention — GitHub](https://github.com/Dao-AILab/flash-attention)
- [Hugging Face Transformers — Documentação](https://huggingface.co/docs/transformers/)
- [Librosa — Documentação](https://librosa.org/doc/latest/)
- [faster-whisper — GitHub](https://github.com/SYSTRAN/faster-whisper)
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

---

*Material elaborado para a disciplina de Engenharia de Machine Learning — FIAP PósTech.*
*Última atualização: Março de 2026.*
