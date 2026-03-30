# Datasets — Aula 02: Desafios de Performance em NLP e Áudio

## Dataset Principal: `raw/dataset.csv`

### Descrição

Dataset **sintético** projetado para demonstrar os desafios de performance em
mecanismos de self-attention e processamento de sinais de áudio, conforme
discutido na seção *Saiba Mais* do material teórico.

Cada registro representa uma amostra com:
- Metadados de sequência (comprimento, tipo) para experimentos de atenção.
- Parâmetros de sinal de áudio sintético para extração de features.
- Métricas de referência para validação de implementações.

### Estrutura

| Coluna              | Tipo   | Descrição                                                     |
|---------------------|--------|---------------------------------------------------------------|
| `sample_id`         | int    | Identificador único da amostra                                |
| `seq_length`        | int    | Comprimento da sequência de tokens (32–1024)                  |
| `d_model`           | int    | Dimensão do modelo (64, 128, 256)                             |
| `num_heads`         | int    | Número de cabeças de atenção (1, 2, 4, 8)                     |
| `signal_freq_hz`    | float  | Frequência fundamental do sinal de áudio sintético (Hz)       |
| `signal_duration_s` | float  | Duração do sinal de áudio (segundos)                          |
| `sample_rate`       | int    | Taxa de amostragem (8000, 16000, 22050)                       |
| `n_mels`            | int    | Número de filtros Mel (40, 80, 128)                           |
| `noise_level`       | float  | Nível de ruído aditivo (0.0–0.5)                              |
| `attention_type`    | str    | Tipo de atenção: vanilla, linear, sparse, local               |
| `vanilla_latency_ms`| float  | Latência de atenção vanilla (referência)                      |
| `memory_mb`         | float  | Uso de memória estimado para a configuração                   |
| `category`          | str    | Categoria: short_seq, medium_seq, long_seq                    |

### Estatísticas

- **Instâncias:** 5 000
- **Features numéricas:** 9
- **Features categóricas:** 3
- **Categorias de sequência:** 3 (short ≤128, medium 129–512, long >512)
- **Sem valores nulos**

### Fonte

Dataset **100 % sintético**, gerado pelo script `src/data_preprocessing.py`
com seed fixa (`RANDOM_SEED = 42`) para reprodutibilidade.

### Reprodução

```bash
# Dentro da pasta da aula
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_dataset()"
```

O arquivo será (re)criado em `data/raw/dataset.csv`.
