# Aula 08 — Escalabilidade e Orquestração Multimodal em Produção

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Aplicar princípios de Site Reliability Engineering (SRE)** a sistemas de ML, definindo SLIs, SLOs e error budgets conforme formalizado por Beyer et al. (2016).
2. **Comparar arquiteturas de serving** (microserviços vs. monolito) à luz do Teorema CAP (Brewer, 2000; Gilbert & Lynch, 2002).
3. **Configurar orquestradores de ML serving** (KServe, Seldon Core, BentoML) sobre Kubernetes.
4. **Implementar canary deployments** com validação estatística (teste-t, qui-quadrado) para promoção/rollback de modelos.
5. **Instrumentar observabilidade** (métricas, logs, traces) para pipelines multimodais em produção.
6. **Calcular error budgets** e modelar latência usando teoria de filas (Erlang-C).
7. **Reconhecer e mitigar technical debt** em sistemas de ML conforme Sculley et al. (2015).

---

## 📖 Teoria-Chave

### SLIs, SLOs e Error Budgets
- **SLIs** — métricas quantitativas: latência P50/P95/P99, throughput, taxa de erros, accuracy em produção.
- **SLOs** — metas sobre SLIs: "99 % das inferências < 200 ms em janela de 30 dias".
- **Error Budget** — recurso gerenciável derivado do SLO (ex.: 0,1 % ≈ 43 min/mês).

### Teoria de Filas (Erlang-C)
A latência média de espera em um sistema M/M/c:

$$W_q = \\frac{C(c, \\rho)}{c\\mu - \\lambda}$$

### Canary Deployment — Testes Estatísticos
- **Latência**: teste-t de Student.
- **Taxa de erro**: teste qui-quadrado de independência.

### Technical Debt em ML
Sculley et al. (2015) mostram que < 5 % de um sistema real de ML é código de modelo; o restante é infraestrutura de dados, serving e monitoramento.

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão mín. | Uso |
|------------|-------------|-----|
| Python | 3.11+ | Runtime |
| numpy | ≥ 1.24.0 | Cálculos numéricos |
| pandas | ≥ 2.0.0 | Manipulação de dados |
| scikit-learn | ≥ 1.3.0 | Modelos auxiliares |
| matplotlib | ≥ 3.7.0 | Visualizações |
| seaborn | ≥ 0.12.0 | Gráficos estatísticos |
| scipy | ≥ 1.11.0 | Testes estatísticos e Erlang-C |
| pytest | ≥ 7.4.0 | Testes unitários |
| jupyter | ≥ 1.0.0 | Notebooks |

---

## 📂 Estrutura de Arquivos

```
08_escalabilidade_orquestracao_multimodal/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências
├── notebooks/
│   ├── 01_exploracao.ipynb          # EDA de métricas de serving
│   ├── 02_treinamento.ipynb         # Simulação de auto-scaling e canary
│   └── 03_avaliacao.ipynb           # Avaliação de SLOs e testes estatísticos
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Geração e pré-processamento de dados de inferência
│   ├── model.py                     # Simuladores: MultimodalOrchestrator, SLOMonitor
│   ├── training.py                  # Auto-scaling e canary deployment
│   ├── evaluation.py                # Error budgets, testes estatísticos, métricas SRE
│   └── utils.py                     # Serialização, configuração, logging
├── data/
│   ├── raw/                         # Dados brutos de simulação
│   │   └── dataset.csv
│   ├── processed/                   # Dados processados
│   └── README.md                    # Descrição dos dados
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
cd 08_escalabilidade_orquestracao_multimodal
pip install -r requirements.txt
```

### 2. Gerar dataset sintético

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_and_save()"
```

### 3. Executar notebooks (sequencial)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
# depois  → notebooks/02_treinamento.ipynb
# depois  → notebooks/03_avaliacao.ipynb
```

### 4. Executar via scripts

```python
from src.data_preprocessing import DataPreprocessor
from src.model import MultimodalOrchestrator
from src.training import run_autoscaling_simulation
from src.evaluation import evaluate_slos

dp = DataPreprocessor()
df = dp.load_data("data/raw/dataset.csv")
df_clean = dp.clean_data(df)
```

---

## 📊 Datasets

Dataset sintético simulando métricas de um sistema de inferência multimodal em produção. Veja detalhes em [`data/README.md`](data/README.md).

---

## 🧪 Testes

```bash
cd 08_escalabilidade_orquestracao_multimodal
pytest tests/ -v
```

---

## 📚 Referências

- BEYER, B. et al. **Site Reliability Engineering: How Google Runs Production Systems**. O'Reilly, 2016.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. **NeurIPS**, 2015.
- PALEYES, A.; URMA, R.-G.; LAWRENCE, N. D. Challenges in Deploying Machine Learning. **ACM Computing Surveys**, 2022.
- BREWER, E. Towards Robust Distributed Systems. **PODC**, 2000.
- GILBERT, S.; LYNCH, N. Brewer's Conjecture … **ACM SIGACT News**, 2002.
- SRIDHARAN, C. **Distributed Systems Observability**. O'Reilly, 2018.
- BURKOV, A. **Machine Learning Engineering**. True Positive Inc., 2020.

---

## 🎥 Vídeos Relacionados

- **Vídeo 1** — Introdução a SRE para ML: SLIs, SLOs e Error Budgets.
- **Vídeo 2** — Dashboard de monitoramento de SLOs em ação.
- **Vídeo 3** — Canary deployment completo com análise estatística.
- **Vídeo 4** — Pipeline de observabilidade com Application Insights.

---

## 🔗 Links Úteis

- [KServe Documentation](https://kserve.github.io/website/)
- [Seldon Core Documentation](https://docs.seldon.io/projects/seldon-core/)
- [BentoML Documentation](https://docs.bentoml.com)
- [Azure ML Managed Online Endpoints](https://learn.microsoft.com/azure/machine-learning/concept-endpoints)
- [OpenTelemetry](https://opentelemetry.io/)
- [Erlang-C Calculator Reference](https://en.wikipedia.org/wiki/Erlang_(unit)#Erlang_C_formula)
