# Aula 03 — Desafios de Performance em Visão Computacional

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Compreender** a complexidade computacional de operações convolucionais e de self-attention em redes neurais de visão
2. **Calcular** FLOPs para camadas convolucionais ($\text{FLOPs} = 2 \times K^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out}$) e mecanismos de atenção ($\text{FLOPs} = 4nd^2 + 2n^2d$)
3. **Comparar** trade-offs entre CNNs e Vision Transformers sob a ótica de latência, FLOPs e acurácia
4. **Analisar** o impacto quadrático da resolução de entrada no custo computacional
5. **Avaliar** arquiteturas eficientes (MobileNet, EfficientNet, EfficientViT) para diferentes cenários de deployment
6. **Implementar** benchmarks de latência comparativos entre arquiteturas CNN e ViT
7. **Construir** pipelines de detecção em tempo real com YOLOv8
8. **Aplicar** técnicas de processamento de vídeo otimizado com keyframe extraction e temporal downsampling
9. **Selecionar** arquiteturas na fronteira de Pareto para restrições específicas de produção

## 📖 Teoria-Chave

### Complexidade Computacional em Visão

A **complexidade de operações convolucionais** escala com múltiplos fatores:

$$\text{FLOPs}_{\text{conv}} = 2 \times K^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out}$$

Para **Vision Transformers**, o mecanismo de self-attention apresenta complexidade quadrática:

$$\text{FLOPs}_{\text{attention}} = 4 \times n \times d^2 + 2 \times n^2 \times d$$

onde $n$ é o número de patches e $d$ a dimensionalidade do embedding (Dosovitskiy et al., 2021).

### Arquiteturas Eficientes

- **MobileNet** (Howard et al., 2017): Depthwise separable convolutions reduzem custo em ~8-9x
- **EfficientNet** (Tan & Le, 2019): Compound scaling com $d = \alpha^\phi$, $w = \beta^\phi$, $r = \gamma^\phi$
- **EfficientViT** (Cai et al., 2023): Atenção linear com complexidade $O(nd^2)$

### Detecção em Tempo Real

| Variante | Parâmetros (M) | FLOPs (G) | mAP@0.5:0.95 | Latência V100 (ms) |
|----------|----------------|-----------|--------------|---------------------|
| YOLOv8n  | 3.2            | 8.7       | 37.3         | 1.8                 |
| YOLOv8s  | 11.2           | 28.6      | 44.9         | 2.4                 |
| YOLOv8m  | 25.9           | 78.9      | 50.2         | 4.1                 |

*Fonte: Adaptado de Ultralytics (Jocher et al., 2023).*

### Fronteira de Pareto

A seleção de arquitetura é um problema de otimização multi-objetivo na **fronteira de Pareto** entre
acurácia, latência e custo. Cada ponto na fronteira representa um trade-off ótimo onde não é
possível melhorar um objetivo sem degradar outro (Tan & Le, 2019; Cai et al., 2020).

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão  | Uso |
|-----------|---------|-----|
| Python     | ≥3.10   | Linguagem base |
| PyTorch    | ≥2.0.0  | Framework de deep learning |
| torchvision | ≥0.15.0 | Modelos e transforms de visão |
| timm       | ≥0.9.0  | Biblioteca de modelos de visão |
| NumPy      | ≥1.24.0 | Computação numérica |
| Pandas     | ≥2.0.0  | Manipulação de dados |
| Matplotlib | ≥3.7.0  | Visualização |
| Seaborn    | ≥0.12.0 | Visualizações estatísticas |
| OpenCV     | ≥4.8.0  | Processamento de imagens e vídeo |
| scikit-learn | ≥1.3.0 | Métricas e utilitários |
| Ultralytics | ≥8.0.0 | YOLOv8 (opcional) |

## 📂 Estrutura de Arquivos

```
03_desafios_performance_visao_computacional/
├── README.md                          # Este documento
├── requirements.txt                   # Dependências específicas
├── notebooks/
│   ├── 01_exploracao.ipynb            # EDA de benchmarks de arquiteturas
│   ├── 02_treinamento.ipynb           # Benchmarking e treinamento de modelos
│   └── 03_avaliacao.ipynb             # Avaliação, Pareto e processamento de vídeo
├── src/
│   ├── __init__.py                    # Inicialização do pacote
│   ├── data_preprocessing.py          # Geração e pré-processamento de dados
│   ├── model.py                       # Modelos de visão e cálculo de FLOPs
│   ├── training.py                    # Treinamento e benchmarking
│   ├── evaluation.py                  # Métricas e visualizações de performance
│   └── utils.py                       # Utilitários: timer, save/load, config
├── data/
│   ├── raw/
│   │   └── dataset.csv                # Benchmark de arquiteturas (~5000 observações)
│   ├── processed/
│   └── README.md                      # Descrição dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Testes do pré-processamento
│   ├── test_model.py                  # Testes dos modelos e FLOPs
│   └── test_evaluation.py             # Testes de métricas e avaliação
└── outputs/
    ├── models/                        # Modelos salvos
    ├── figures/                       # Gráficos gerados
    └── logs/                          # Logs de execução
```

## 🚀 Como Executar

### Instalação de Dependências

```bash
cd 03_desafios_performance_visao_computacional
pip install -r requirements.txt
```

### Geração do Dataset

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor.generate_benchmark_dataset('data/raw/dataset.csv')"
```

### Execução dos Notebooks (sequencial)

```bash
jupyter notebook notebooks/01_exploracao.ipynb     # Passo 1: Exploração
jupyter notebook notebooks/02_treinamento.ipynb    # Passo 2: Benchmarking
jupyter notebook notebooks/03_avaliacao.ipynb      # Passo 3: Avaliação
```

### Execução via Scripts

```bash
# Gerar dataset de benchmarks
python -c "from src.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.generate_benchmark_dataset()"

# Benchmark de arquiteturas
python -c "from src.training import benchmark_architectures; benchmark_architectures()"

# Análise de Pareto
python -c "from src.evaluation import plot_pareto_frontier; plot_pareto_frontier()"
```

## 📊 Datasets

Dataset de benchmarks de arquiteturas de visão computacional com ~5000 observações.
Cada registro representa uma execução de benchmark com modelo, resolução, hardware e métricas.

Detalhes completos em [`data/README.md`](data/README.md).

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes específicos
pytest tests/test_model.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_evaluation.py -v
```

## 📚 Referências

Referências principais relacionadas ao código implementado:

1. **Dosovitskiy, A. et al. (2021).** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021.*
2. **Howard, A. G. et al. (2017).** MobileNets: Efficient CNNs for Mobile Vision Applications. *arXiv:1704.04861.*
3. **Tan, M. & Le, Q. V. (2019).** EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019.*
4. **Cai, H. et al. (2023).** EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction. *CVPR 2023.*
5. **Jocher, G. et al. (2023).** Ultralytics YOLOv8. *GitHub repository.*
6. **Lv, W. et al. (2023).** RT-DETR: DETRs Beat YOLOs on Real-time Object Detection. *arXiv:2304.08069.*
7. **He, K. & Sun, J. (2015).** Convolutional Neural Networks at Constrained Time Cost. *CVPR 2015.*
8. **Zhu, X. et al. (2017).** Deep Feature Flow for Video Recognition. *CVPR 2017.*

## 🎥 Vídeos Relacionados

- **Vídeo 1:** Introdução teórica a complexidade computacional em visão computacional
- **Vídeo 2:** Benchmark comparativo CNN vs ViT — Demonstração prática com análise de mapas de atenção
- **Vídeo 3:** Detecção em tempo real com YOLOv8 e processamento de vídeo otimizado
- **Vídeo 4:** Seleção de arquiteturas na fronteira de Pareto para cenários de produção

## 🔗 Links Úteis

- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/)
- [torchvision.transforms.v2](https://pytorch.org/vision/stable/transforms.html)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
