"""
Aula 07 — Inferência Distribuída e Paralelismo de Modelos.

Pacote contendo simuladores pedagógicos de paralelismo, PagedAttention,
continuous batching, speculative decoding e métricas de serving para
Large Language Models.

Referências principais:
    Kwon, W. et al. (2023). Efficient Memory Management for Large Language
        Model Serving with PagedAttention. SOSP 2023.
    Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.
    Shoeybi, M. et al. (2019). Megatron-LM: Training Multi-Billion Parameter
        Language Models Using Model Parallelism.
"""

from src.data_preprocessing import WorkloadPreprocessor
from src.model import (
    ContinuousBatchingScheduler,
    PagedKVCacheManager,
    PipelineParallelismSimulator,
    SpeculativeDecoder,
    TensorParallelismSimulator,
)
from src.evaluation import ServingMetricsEvaluator
from src.training import InferenceRunner
from src.utils import save_metrics, load_metrics, save_config, load_config
