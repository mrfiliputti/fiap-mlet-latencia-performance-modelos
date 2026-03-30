# Datasets — Aula 07: Inferência Distribuída e Paralelismo de Modelos

## Dataset Principal: Workload Sintético de Inferência LLM

### Descrição

Dataset sintético que simula um workload de requisições de inferência para Large Language Models.
Cada linha representa uma requisição individual com características típicas de ambientes de
serving LLM em produção, conforme descrito na seção "Saiba Mais" do material da aula,
particularmente nas subseções sobre Continuous Batching (Yu et al., 2022) e PagedAttention
(Kwon et al., 2023).

O dataset permite experimentar com:
- Análise de distribuições de comprimento de prompt e geração
- Simulação de scheduling e continuous batching
- Análise de uso de memória KV-Cache com PagedAttention
- Benchmarking de estratégias de paralelismo

### Estrutura

| Coluna | Tipo | Descrição |
|---|---|---|
| request_id | int | Identificador único da requisição |
| arrival_time_ms | float | Tempo de chegada em milissegundos desde o início do trace |
| prompt_tokens | int | Número de tokens no prompt de entrada |
| max_gen_tokens | int | Máximo de tokens a gerar (saída) |
| actual_gen_tokens | int | Tokens efetivamente gerados (≤ max_gen_tokens) |
| model_size_b | float | Tamanho do modelo em bilhões de parâmetros |
| num_attention_heads | int | Número de cabeças de atenção do modelo |
| hidden_dim | int | Dimensão oculta do modelo |

### Estatísticas

- **Instâncias:** 5.000 requisições
- **Features:** 8
- **Distribuição de prompt_tokens:** lognormal (μ=5.5, σ=1.0), range ~30–2048
- **Distribuição de actual_gen_tokens:** lognormal (μ=4.0, σ=1.2), range ~10–512
- **Modelos simulados:** 7B, 13B, 70B (proporção 50%/35%/15%)
- **Taxa de chegada:** Processo de Poisson (~50 req/s)

### Fonte

**Sintético** — gerado programaticamente para fins pedagógicos.

As distribuições foram calibradas para reproduzir padrões descritos em:
- Kwon et al. (2023) — distribuições de comprimento de sequência do dataset ShareGPT
- Yu et al. (2022) — padrões de chegada em sistemas Orca

### Reprodução

```bash
python -c "from src.data_preprocessing import WorkloadPreprocessor; WorkloadPreprocessor.generate_synthetic_workload('data/raw/dataset.csv')"
```
