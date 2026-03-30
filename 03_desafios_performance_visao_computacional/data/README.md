# Datasets — Aula 03: Desafios de Performance em Visão Computacional

## Dataset Principal: Benchmarks de Arquiteturas de Visão

### Descrição

Dataset sintético de benchmarks de performance para arquiteturas de visão computacional.
Cada registro representa uma execução de benchmark contendo o modelo avaliado, resolução
de entrada, hardware utilizado e métricas de performance medidas. O dataset permite análise
exploratória dos trade-offs entre acurácia, latência e custo computacional discutidos na
seção "Saiba Mais" da aula.

### Estrutura

| Coluna              | Tipo   | Descrição                                        |
|---------------------|--------|--------------------------------------------------|
| model_name          | str    | Nome da arquitetura (e.g., `efficientnet_b0`)    |
| model_family        | str    | Família: `cnn`, `vit` ou `hybrid`                |
| input_resolution    | int    | Resolução de entrada (lado do quadrado, pixels)  |
| hardware            | str    | Hardware de execução: `v100`, `rtx3090`, `cpu`   |
| batch_size          | int    | Tamanho do batch de inferência                   |
| params_millions     | float  | Número de parâmetros (milhões)                   |
| flops_giga          | float  | FLOPs de inferência (bilhões)                    |
| top1_accuracy       | float  | Acurácia Top-1 ImageNet (%)                      |
| latency_ms          | float  | Latência de inferência (ms) — batch_size=1       |
| throughput_imgs_sec | float  | Throughput (imagens/segundo)                     |
| memory_mb           | float  | Uso de memória de pico (MB)                      |
| run_id              | int    | Identificador da execução (para repetições)      |

### Estatísticas

- **Instâncias:** ~5.000
- **Modelos únicos:** 20 arquiteturas
- **Resoluções:** 5 (128, 224, 320, 384, 512)
- **Hardwares:** 3 (V100, RTX 3090, CPU)
- **Repetições:** ~17 execuções por configuração
- **Missing values:** Nenhum
- **Seed de reprodução:** 42

### Modelos Incluídos

| Modelo             | Família | Parâmetros (M) | Referência                   |
|--------------------|---------|----------------|------------------------------|
| MobileNetV2        | CNN     | 3.4            | Sandler et al. (2018)        |
| MobileNetV3-Large  | CNN     | 5.4            | Howard et al. (2019)         |
| EfficientNet-B0    | CNN     | 5.3            | Tan & Le (2019)              |
| EfficientNet-B4    | CNN     | 19.0           | Tan & Le (2019)              |
| EfficientNetV2-S   | CNN     | 21.5           | Tan & Le (2021)              |
| ResNet-18          | CNN     | 11.7           | He et al. (2016)             |
| ResNet-50          | CNN     | 25.6           | He et al. (2016)             |
| VGG-16             | CNN     | 138.4          | Simonyan & Zisserman (2015)  |
| ConvNeXt-Tiny      | CNN     | 28.6           | Liu et al. (2022)            |
| ConvNeXt-Small     | CNN     | 50.2           | Liu et al. (2022)            |
| DeiT-Tiny          | ViT     | 5.7            | Touvron et al. (2021)        |
| DeiT-Small         | ViT     | 22.1           | Touvron et al. (2021)        |
| DeiT-Base          | ViT     | 86.6           | Touvron et al. (2021)        |
| ViT-Base/16        | ViT     | 86.6           | Dosovitskiy et al. (2021)    |
| Swin-Tiny          | Hybrid  | 28.3           | Liu et al. (2021)            |
| Swin-Small         | Hybrid  | 49.6           | Liu et al. (2021)            |
| EfficientViT-B1    | Hybrid  | 9.1            | Cai et al. (2023)            |
| YOLOv8n (backbone) | CNN     | 3.2            | Jocher et al. (2023)         |
| YOLOv8s (backbone) | CNN     | 11.2           | Jocher et al. (2023)         |
| YOLOv8m (backbone) | CNN     | 25.9           | Jocher et al. (2023)         |

### Fonte

Dataset **sintético** gerado deterministicamente com valores-âncora extraídos de papers
e benchmarks publicados (Tan & Le, 2019; Dosovitskiy et al., 2021; Jocher et al., 2023).
Variabilidade entre execuções simulada com ruído gaussiano controlado (σ = 3-5%).

### Reprodução

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor.generate_benchmark_dataset('data/raw/dataset.csv')"
```

### Propósito Pedagógico

O dataset foi projetado especificamente para:

1. **Análise comparativa** — Comparar famílias de modelos (CNN vs ViT vs Híbridos)
2. **Impacto da resolução** — Observar crescimento quadrático de FLOPs com resolução
3. **Fronteira de Pareto** — Identificar trade-offs ótimos entre acurácia e latência
4. **Seleção de hardware** — Comparar performance em diferentes aceleradores
5. **Análise estatística** — Múltiplas execuções permitem intervalos de confiança
