# Aula 07 — Inferência Distribuída e Paralelismo de Modelos

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Compreender** a taxonomia de estratégias de paralelismo (data, tensor e pipeline) e quando aplicar cada uma
2. **Implementar** simulações de tensor parallelism e pipeline parallelism para compreender particionamento de modelos
3. **Analisar** o impacto de PagedAttention no gerenciamento de KV-Cache, comparando alocação contígua vs. paginada
4. **Projetar** schedulers de continuous batching que maximizam throughput sob restrições de memória
5. **Avaliar** speculative decoding como estratégia de redução de latência com modelos draft
6. **Medir** métricas de serving (throughput, latência P50/P95/P99) e definir SLOs para sistemas de inferência
7. **Comparar** frameworks de serving distribuído (vLLM, Triton, TGI) quanto a trade-offs de performance

## 📖 Teoria-Chave

### Paralelismo em Deep Learning

A demanda computacional de modelos de linguagem de grande escala transformou fundamentalmente
a arquitetura de sistemas de inferência. Enquanto o GPT-2 (2019) operava em uma única GPU com
1.5B parâmetros, modelos como LLaMA-2 70B exigem dezenas de aceleradores em conjunto
(Brown et al., 2020).

**Estratégias de Paralelismo:**

| Estratégia | Caso de Uso | Overhead | Escalabilidade |
|---|---|---|---|
| Data Parallelism | Modelo cabe em 1 GPU | Baixo | Linear |
| Tensor Parallelism | Modelo grande, baixa latência | Médio | 8–16 GPUs |
| Pipeline Parallelism | Modelos muito grandes | Baixo-médio | Alta |
| ZeRO-Inference | Memória limitada | Médio | Alta |

### PagedAttention e KV-Cache

O mecanismo de KV-Cache é fundamental para inferência eficiente de Transformers autoregressivos.
PagedAttention (Kwon et al., 2023) gerencia memória em blocos de tamanho fixo, inspirando-se
em memória virtual de sistemas operacionais. Para uma sequência com $n$ tokens e blocos de
tamanho $B$, o número de blocos é $\lceil n/B \rceil$.

### Continuous Batching

Continuous batching (Yu et al., 2022) trata cada iteração de geração como unidade de scheduling
independente. Quando uma requisição completa, é imediatamente removida do batch e uma nova é
inserida, mantendo o batch maximamente preenchido.

### Speculative Decoding

Speculative decoding (Leviathan et al., 2023) utiliza um modelo draft menor para gerar
candidatos de múltiplos tokens, verificados em paralelo pelo modelo principal. O speedup
depende da acceptance rate $\alpha$:

$$\text{speedup} = \frac{K \cdot \alpha}{\text{cost}_{\text{draft}} + \text{cost}_{\text{verify}}}$$

### Lei de Amdahl e Gustafson

A Lei de Amdahl limita o speedup pela fração sequencial: se 10% do código é sequencial,
o speedup máximo é 10×. Gustafson (1988) propõe perspectiva complementar onde, com problemas
escaláveis, os ganhos de paralelização podem ser substancialmente maiores.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | ≥3.10 | Linguagem base |
| NumPy | ≥1.24.0 | Computação numérica |
| Pandas | ≥2.0.0 | Manipulação de dados |
| Matplotlib | ≥3.7.0 | Visualizações |
| Seaborn | ≥0.12.0 | Gráficos estatísticos |
| PyTorch | ≥2.1.0 | Framework de deep learning |
| pytest | ≥7.4.0 | Testes unitários |

> **Nota:** Esta aula utiliza simulações pedagógicas dos conceitos de paralelismo e serving
> distribuído. Para execução real com vLLM e multi-GPU, consulte a seção Hands On do
> documento acadêmico e os requisitos de hardware (2× A10G ou equivalente).

## 📂 Estrutura de Arquivos

```
07_inferencia_distribuida_paralelismo/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb          # Exploração de paralelismo e memória
│   ├── 02_treinamento.ipynb         # Simulação de serving distribuído
│   └── 03_avaliacao.ipynb           # Benchmarks e avaliação de SLOs
├── src/
│   ├── __init__.py                  # Inicialização do pacote
│   ├── data_preprocessing.py        # Geração e preparação de workloads
│   ├── model.py                     # Simuladores de paralelismo e serving
│   ├── training.py                  # Lógica de execução de inferência
│   ├── evaluation.py                # Métricas de latência e throughput
│   └── utils.py                     # Utilitários (save/load, configs)
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Workload sintético de requisições
│   ├── processed/                   # Dados processados (gerado em runtime)
│   └── README.md                    # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py        # Testes do pré-processamento
│   ├── test_model.py                # Testes dos simuladores
│   └── test_evaluation.py           # Testes de avaliação
└── outputs/
    ├── models/                      # Configurações salvas
    ├── figures/                     # Gráficos gerados
    └── logs/                        # Logs de benchmark
```

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
cd 07_inferencia_distribuida_paralelismo
pip install -r requirements.txt
```

### 2. Gerar Dataset (se necessário)

```bash
python -c "from src.data_preprocessing import WorkloadPreprocessor; WorkloadPreprocessor.generate_synthetic_workload('data/raw/dataset.csv')"
```

### 3. Executar Notebooks (sequencialmente)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Executar via Scripts Python

```bash
python -m src.training
```

## 📊 Datasets

O dataset utilizado é um **workload sintético** que simula requisições de inferência LLM com
propriedades variáveis de comprimento de prompt e geração. Consulte [data/README.md](data/README.md)
para detalhes completos.

- **Instâncias:** 5.000 requisições sintéticas
- **Features:** 8 (comprimento do prompt, tokens gerados, tipo de modelo, etc.)
- **Target:** Métricas de latência simuladas

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ -v --tb=short
```

## 📚 Referências

1. **Brown, T. et al.** (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*.
   DOI: 10.48550/arXiv.2005.14165.
2. **Kwon, W. et al.** (2023). "Efficient Memory Management for Large Language Model Serving
   with PagedAttention." *SOSP 2023*. DOI: 10.1145/3600006.3613165.
3. **Yu, G.-I. et al.** (2022). "Orca: A Distributed Serving System for Transformer-Based
   Generative Models." *OSDI 2022*.
4. **Leviathan, Y. et al.** (2023). "Fast Inference from Transformers via Speculative Decoding."
   *ICML 2023*.
5. **Shoeybi, M. et al.** (2019). "Megatron-LM: Training Multi-Billion Parameter Language
   Models Using Model Parallelism." DOI: 10.48550/arXiv.1909.08053.
6. **Rajbhandari, S. et al.** (2020). "ZeRO: Memory Optimizations Toward Training Trillion
   Parameter Models." *SC 2020*. DOI: 10.1109/SC41405.2020.00024.
7. **Huang, Y. et al.** (2019). "GPipe: Efficient Training of Giant Neural Networks using
   Pipeline Parallelism." *NeurIPS 2019*.
8. **Narayanan, D. et al.** (2019). "PipeDream: Generalized Pipeline Parallelism for DNN
   Training." *SOSP 2019*.
9. **Gustafson, J. L.** (1988). "Reevaluating Amdahl's Law." *Communications of the ACM*,
   31(5), 532–533.
10. **Pope, R. et al.** (2022). "Efficiently Scaling Transformer Inference." *MLSys 2022*.

## 🎥 Vídeos Relacionados

- **Vídeo 1:** Fundamentos de paralelismo — data, tensor e pipeline parallelism
- **Vídeo 2:** Visualização detalhada de PagedAttention gerenciando blocos de memória
- **Vídeo 3:** Configuração prática de vLLM com tensor parallelism em Azure ML
- **Vídeo 4:** Benchmark comparativo vLLM vs. Hugging Face Transformers (8× speedup)

## 🔗 Links Úteis

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Hugging Face Text Generation Inference](https://huggingface.co/docs/text-generation-inference/)
- [Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
