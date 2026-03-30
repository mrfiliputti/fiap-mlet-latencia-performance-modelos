# Datasets — Aula 5: Transfer Learning Eficiente e Adaptação de Modelos

## Dataset Principal: Classificação de Domínio Textual

### Descrição

Dataset sintético de textos em português brasileiro distribuídos em **5 domínios**
temáticos. Projetado para demonstrar Transfer Learning e fine-tuning eficiente
com técnicas PEFT (LoRA, QLoRA), conforme discutido na seção "Fundamentos do
Transfer Learning" (Pan & Yang, 2010) do material da Aula 5.

A tarefa de **classificação de domínio** simula o cenário real de adaptar um
modelo pré-treinado (ex: BERT, DistilBERT) a domínios especializados — como
jurídico ou financeiro — utilizando fine-tuning eficiente com LoRA (Hu et al., 2022).

### Estrutura

| Coluna | Tipo | Descrição |
|-----------|--------|---------------------------------------------|
| `text` | string | Texto do documento (50–200 caracteres) |
| `label` | string | Categoria do domínio (5 classes) |
| `domain_id`| int | Identificador numérico do domínio (0–4) |

### Classes

| domain_id | label | Descrição |
|-----------|-------------|---------------------------------------|
| 0 | educacao | Textos sobre educação e pedagogia |
| 1 | financeiro | Textos sobre finanças e mercado |
| 2 | juridico | Textos jurídicos e legislação |
| 3 | saude | Textos sobre saúde e medicina |
| 4 | tecnologia | Textos sobre tecnologia e software |

### Estatísticas

- **Instâncias:** 5.000
- **Features textuais:** 1 (coluna `text`)
- **Classes:** 5 (balanceamento uniforme: ~1.000 por classe)
- **Comprimento médio:** ~100 caracteres por documento

### Fonte

**Sintético** — gerado programaticamente com vocabulário temático por domínio.

### Reprodução

```bash
# Gerar dataset completo
python -c "
import sys; sys.path.insert(0, '..')
from src.data_preprocessing import generate_synthetic_dataset
generate_synthetic_dataset(n_samples=5000, output_path='raw/dataset.csv')
"
```

Ou via Python:

```python
from src.data_preprocessing import generate_synthetic_dataset

df = generate_synthetic_dataset(
    n_samples=5000,
    random_state=42,
    output_path="data/raw/dataset.csv"
)
print(df.shape)        # (5000, 3)
print(df["label"].value_counts())
```

### Conexão com o Material Teórico

Este dataset permite demonstrar empiricamente que:

1. **Transfer Learning** (Pan & Yang, 2010): modelos pré-treinados em corpora
   genéricos transferem representações úteis para classificação de domínio.
2. **LoRA** (Hu et al., 2022): basta treinar ~0,06% dos parâmetros com adaptadores
   de baixo rank para atingir 97–99% do desempenho de fine-tuning completo.
3. **QLoRA** (Dettmers et al., 2023): quantização NF4 permite carregar modelos
   maiores com memória reduzida sem perda significativa de qualidade.
