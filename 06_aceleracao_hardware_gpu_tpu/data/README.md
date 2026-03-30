# Datasets — Aula 06: Aceleração com Hardware: GPU, TPU e Aceleradores

## Dataset Principal

### Descrição

Dataset **sintético** de classificação de imagens, projetado para demonstrar técnicas de
aceleração por hardware (Mixed Precision Training, Gradient Checkpointing, Profiling GPU).
Cada amostra representa uma imagem 32×32 pixels em 3 canais (RGB) flattened em vetor de 3072 features,
simulando um cenário de classificação visual compatível com a arquitetura Vision Transformer
discutida na aula.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `feature_0` a `feature_3071` | float32 | Pixels da imagem 32×32×3 normalizados [0, 1] |
| `label` | int | Classe da imagem (0-9) |

### Estatísticas

- **Instâncias:** 10.000
- **Features:** 3.072 (32 × 32 × 3)
- **Classes:** 10 (balanceamento uniforme: ~1.000 por classe)
- **Tamanho em disco:** ~120 MB (CSV)

### Fonte

Dataset **sintético** gerado via `src/data_preprocessing.py`. As distribuições de features
são criadas com clusters gaussianos para simular padrões de imagem classificáveis,
permitindo que os algoritmos de aceleração demonstrem ganhos mensuráveis.

### Reprodução

```bash
# Gerar dataset completo
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_dataset()"

# Ou com parâmetros customizados
python -c "
from src.data_preprocessing import DataPreprocessor
dp = DataPreprocessor(n_samples=10000, img_size=32, n_channels=3, n_classes=10, seed=42)
dp.generate_dataset()
"
```

### Notas

- O dataset é propositalmente grande o suficiente para demonstrar diferenças de performance
  entre FP32 e FP16/BF16, conforme discutido na seção sobre Mixed Precision Training
  (Micikevicius et al., 2018).
- Para experimentos rápidos, use o subconjunto de 1.000 amostras gerado automaticamente
  em `data/processed/`.
