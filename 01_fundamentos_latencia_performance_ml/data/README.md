# Datasets — Aula 01: Fundamentos de Latência e Performance em ML

## Dataset Principal

### Descrição

Dataset sintético de classificação binária projetado para estudar o impacto de diferentes
algoritmos e configurações na latência e throughput de modelos de ML. O dataset simula um
cenário de detecção de fraude em transações financeiras, com features numéricas de diferentes
escalas e distribuições.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| feature_0 | float | Feature informativa (distribuição normal) |
| feature_1 | float | Feature informativa (distribuição normal) |
| feature_2 | float | Feature informativa (distribuição normal) |
| feature_3 | float | Feature informativa (distribuição normal) |
| feature_4 | float | Feature informativa (distribuição normal) |
| feature_5 | float | Feature informativa (distribuição normal) |
| feature_6 | float | Feature informativa (distribuição normal) |
| feature_7 | float | Feature informativa (distribuição normal) |
| feature_8 | float | Feature informativa (distribuição normal) |
| feature_9 | float | Feature informativa (distribuição normal) |
| feature_10 | float | Feature redundante (combinação linear) |
| feature_11 | float | Feature redundante (combinação linear) |
| feature_12 | float | Feature ruidosa (sem poder preditivo) |
| feature_13 | float | Feature ruidosa (sem poder preditivo) |
| feature_14 | float | Feature ruidosa (sem poder preditivo) |
| target | int | Classe binária (0 = legítima, 1 = fraude) |

### Estatísticas

- **Instâncias:** 10.000
- **Features:** 15 numéricas (10 informativas, 2 redundantes, 3 ruidosas)
- **Classes:** 2 (balanceamento: ~50/50)
- **Missing values:** Nenhum
- **Seed de reprodução:** 42

### Fonte

Dataset **sintético** gerado com `sklearn.datasets.make_classification`.

### Reprodução

```bash
# Gerar dataset via módulo src
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor.generate_synthetic_dataset('data/raw/dataset.csv')"
```

### Propósito Pedagógico

O dataset foi projetado especificamente para:

1. **15 features** — tamanho suficiente para demonstrar diferenças de latência entre algoritmos
2. **10K instâncias** — permite medir throughput significativo sem tempo excessivo
3. **Features informativas + ruidosas** — demonstra impacto de feature selection na performance
4. **Balanceamento 50/50** — foco na latência, não no desbalanceamento
