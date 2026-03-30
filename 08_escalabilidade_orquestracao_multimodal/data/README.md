# Datasets — Aula 08: Escalabilidade e Orquestração Multimodal em Produção

## Dataset Principal

### Descrição

Dataset **sintético** que simula logs de métricas de um sistema de inferência multimodal
em produção. Cada linha representa uma requisição de inferência processada por um
pipeline multimodal composto por workers de texto, imagem e áudio, conforme a
arquitetura de referência discutida na seção "Saiba Mais" (Paleyes et al., 2022).

O dataset inclui campos que permitem analisar:
- Distribuições de latência por percentil (P50, P95, P99).
- Cálculo de SLIs, SLOs e error budgets (Beyer et al., 2016).
- Comparação de versões de modelo para canary deployments.
- Métricas por modalidade (texto, imagem, áudio).

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `request_id` | str | Identificador único da requisição |
| `timestamp` | datetime | Momento da requisição (ISO 8601) |
| `modality` | str | Modalidade processada: `text`, `image`, `audio` |
| `model_version` | str | Versão do modelo: `v1-stable`, `v2-canary` |
| `instance_type` | str | Tipo de instância: `cpu`, `gpu` |
| `preprocessing_ms` | float | Tempo de pré-processamento (ms) |
| `inference_ms` | float | Tempo de inferência (ms) |
| `postprocessing_ms` | float | Tempo de pós-processamento (ms) |
| `total_latency_ms` | float | Latência total da requisição (ms) |
| `queue_wait_ms` | float | Tempo de espera na fila (ms) |
| `confidence` | float | Confiança da predição [0, 1] |
| `is_error` | int | Se houve erro (0/1) |
| `error_type` | str | Tipo de erro (timeout/oom/model_error/none) |
| `cpu_utilization` | float | Uso de CPU no momento [0, 1] |
| `memory_utilization` | float | Uso de memória no momento [0, 1] |
| `replica_count` | int | Número de réplicas ativas |

### Estatísticas

- **Instâncias:** 10 000
- **Features:** 16
- **Modalidades:** 3 (text ~40 %, image ~35 %, audio ~25 %)
- **Versões de modelo:** 2 (v1-stable ~90 %, v2-canary ~10 %)
- **Taxa de erro global:** ~2 %
- **Período simulado:** 24 horas

### Fonte

Dataset **sintético** gerado programaticamente para fins pedagógicos.

### Reprodução

```bash
cd 08_escalabilidade_orquestracao_multimodal
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_and_save()"
```

Ou via notebook `01_exploracao.ipynb` (células iniciais).
