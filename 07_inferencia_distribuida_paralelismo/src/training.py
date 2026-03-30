"""
Módulo de execução e orquestração de inferência distribuída simulada.

Implementa a lógica de execução que integra os simuladores de paralelismo,
PagedAttention e continuous batching para demonstrar o pipeline completo
de serving de LLMs em produção.

Conecta-se diretamente com as seções 'Arquitetura de Sistemas de Serving
Distribuído' e 'Estratégias de Auto-scaling e SLOs' do material da aula,
onde é discutido o fluxo: load balancer → router → model replicas → GPU workers.

Referências:
    Kwon, W. et al. (2023). Efficient Memory Management for Large Language
        Model Serving with PagedAttention. SOSP 2023.
    Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.model import (
    ContinuousBatchingScheduler,
    PagedKVCacheManager,
    ScheduledRequest,
    StaticBatchingScheduler,
    TensorParallelismSimulator,
    PipelineParallelismSimulator,
    SpeculativeDecoder,
)


# =========================================================================
# Configuração de simulação
# =========================================================================

@dataclass
class ServingConfig:
    """Configuração para simulação de serving.

    Mapeia parâmetros descritos na seção 'Arquitetura de Sistemas de
    Serving Distribuído': batch size, limite de tokens e número de
    blocos KV-Cache.

    Attributes:
        max_batch_size: Tamanho máximo do batch ativo.
        max_total_tokens: Limite de tokens simultâneos.
        kv_cache_blocks: Blocos de KV-Cache disponíveis.
        kv_block_size: Tokens por bloco PagedAttention.
        num_gpus: GPUs para tensor parallelism.
    """

    max_batch_size: int = 32
    max_total_tokens: int = 8192
    kv_cache_blocks: int = 512
    kv_block_size: int = 16
    num_gpus: int = 2


# =========================================================================
# Runner de Inferência
# =========================================================================

class InferenceRunner:
    """
    Executor de simulações de inferência distribuída.

    Orquestra os diferentes componentes (scheduler, KV-Cache manager,
    simuladores de paralelismo) para executar cenários end-to-end e
    coletar métricas de serving.

    Implementa o pipeline discutido na seção 'Arquitetura de Sistemas
    de Serving Distribuído' (Figura 1 do material da aula):
    Load Balancer → Scheduler (Continuous Batching) → PagedAttention
    KV-Cache → GPU Workers (Tensor Parallel).

    Attributes:
        config: Configuração de serving.
        scheduler: Scheduler de continuous batching.
        kv_manager: Gerenciador de KV-Cache paginado.
    """

    def __init__(self, config: Optional[ServingConfig] = None) -> None:
        self.config = config or ServingConfig()
        self.scheduler = ContinuousBatchingScheduler(
            max_batch_size=self.config.max_batch_size,
            max_total_tokens=self.config.max_total_tokens,
        )
        self.kv_manager = PagedKVCacheManager(
            num_blocks=self.config.kv_cache_blocks,
            block_size=self.config.kv_block_size,
        )

    def run_continuous_batching(
        self,
        requests_df: pd.DataFrame,
        max_steps: int = 5000,
    ) -> dict:
        """Executa simulação de serving com continuous batching.

        Simula o processo completo de serving:
        1. Requisições chegam conforme arrival_time
        2. Scheduler admite no batch quando há espaço
        3. KV-Cache é alocado/liberado dinamicamente
        4. Métricas de latência e throughput são coletadas

        Implementa conceitos das seções 'Continuous Batching' e
        'PagedAttention' do material da aula.

        Args:
            requests_df: DataFrame com workload (precisa de colunas
                prompt_tokens e actual_gen_tokens).
            max_steps: Limite de iterações.

        Returns:
            Dicionário com métricas de serving.
        """
        # Prepara requisições
        for _, row in requests_df.iterrows():
            req = ScheduledRequest(
                request_id=int(row["request_id"]),
                prompt_tokens=int(row["prompt_tokens"]),
                max_gen_tokens=int(row["actual_gen_tokens"]),
            )
            self.scheduler.add_request(req)

        # Executa
        start_time = time.perf_counter()
        history = self.scheduler.run_to_completion(max_steps=max_steps)
        elapsed = time.perf_counter() - start_time

        summary = self.scheduler.get_summary()
        summary["wall_clock_time_s"] = elapsed
        summary["kv_cache_stats"] = self.kv_manager.get_memory_stats()
        summary["history"] = history
        return summary

    def run_static_batching(
        self, requests_df: pd.DataFrame
    ) -> dict:
        """Executa simulação com static batching (baseline).

        Para comparação com continuous batching, conforme seção
        'Continuous Batching e Scheduling Dinâmico' que demonstra
        a superioridade do scheduling dinâmico.

        Args:
            requests_df: DataFrame com workload.

        Returns:
            Dicionário com métricas.
        """
        scheduler = StaticBatchingScheduler(
            max_batch_size=self.config.max_batch_size
        )

        requests = [
            ScheduledRequest(
                request_id=int(row["request_id"]),
                prompt_tokens=int(row["prompt_tokens"]),
                max_gen_tokens=int(row["actual_gen_tokens"]),
            )
            for _, row in requests_df.iterrows()
        ]

        start = time.perf_counter()
        history = scheduler.process_requests(requests)
        elapsed = time.perf_counter() - start

        # Calcula latências
        latencies = []
        for req in scheduler.completed:
            total_latency = req.finish_step - req.arrival_step
            latencies.append(total_latency)

        latencies_arr = np.array(latencies) if latencies else np.array([0])

        return {
            "total_completed": len(scheduler.completed),
            "total_steps": scheduler.total_steps,
            "avg_latency_steps": float(np.mean(latencies_arr)),
            "p50_latency_steps": float(np.percentile(latencies_arr, 50)),
            "p95_latency_steps": float(np.percentile(latencies_arr, 95)),
            "p99_latency_steps": float(np.percentile(latencies_arr, 99)),
            "wall_clock_time_s": elapsed,
            "history": history,
        }

    def run_tensor_parallelism_benchmark(
        self,
        matrix_sizes: Optional[list[int]] = None,
        gpu_counts: Optional[list[int]] = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Executa benchmark de tensor parallelism com diferentes configurações.

        Compara speedup real vs. teórico para diferentes tamanhos de
        matriz e números de GPUs, conforme discutido na seção
        'Tensor Parallelism para Large Language Models' (Shoeybi et al., 2019).

        Args:
            matrix_sizes: Dimensões de matrizes a testar.
            gpu_counts: Números de GPUs a testar.
            batch_size: Tamanho do batch.

        Returns:
            DataFrame com resultados do benchmark.
        """
        if matrix_sizes is None:
            matrix_sizes = [256, 512, 1024, 2048, 4096]
        if gpu_counts is None:
            gpu_counts = [1, 2, 4, 8]

        rng = np.random.default_rng(42)
        results = []

        for dim in matrix_sizes:
            X = rng.standard_normal((batch_size, dim)).astype(np.float32)
            W = rng.standard_normal((dim, dim)).astype(np.float32)

            # Baseline: 1 GPU
            start = time.perf_counter()
            _ = X @ W
            baseline_time = time.perf_counter() - start

            for n_gpus in gpu_counts:
                if dim % n_gpus != 0:
                    continue

                sim = TensorParallelismSimulator(
                    num_devices=n_gpus, comm_latency_factor=0.05
                )
                _, metrics = sim.simulate_forward_column_parallel(X, W)

                results.append(
                    {
                        "matrix_dim": dim,
                        "num_gpus": n_gpus,
                        "baseline_time_ms": baseline_time * 1000,
                        "parallel_time_ms": metrics["total_time"] * 1000,
                        "speedup": baseline_time / max(metrics["total_time"], 1e-12),
                        "comm_overhead_ms": metrics["comm_time_simulated"] * 1000,
                        "efficiency": (
                            baseline_time
                            / max(metrics["total_time"] * n_gpus, 1e-12)
                        ),
                    }
                )

        return pd.DataFrame(results)

    def run_pipeline_benchmark(
        self,
        stage_counts: Optional[list[int]] = None,
        micro_batch_counts: Optional[list[int]] = None,
        batch_size: int = 64,
        hidden_dim: int = 64,
    ) -> pd.DataFrame:
        """Benchmark de pipeline parallelism variando estágios e micro-batches.

        Valida empiricamente a fórmula de bubble fraction:
        (P-1)/(M+P-1), conforme Huang et al. (2019) — GPipe.

        Args:
            stage_counts: Números de estágios a testar.
            micro_batch_counts: Números de micro-batches a testar.
            batch_size: Tamanho total do batch.
            hidden_dim: Dimensão oculta.

        Returns:
            DataFrame com resultados.
        """
        if stage_counts is None:
            stage_counts = [2, 4, 8]
        if micro_batch_counts is None:
            micro_batch_counts = [1, 2, 4, 8, 16, 32]

        rng = np.random.default_rng(42)
        results = []

        for num_stages in stage_counts:
            for num_mb in micro_batch_counts:
                if batch_size < num_mb:
                    continue

                sim = PipelineParallelismSimulator(
                    num_stages=num_stages,
                    stage_compute_ms=5.0,
                    hidden_dim=hidden_dim,
                )
                batch = rng.standard_normal(
                    (batch_size, hidden_dim)
                ).astype(np.float32)

                _, metrics = sim.simulate_pipeline(batch, num_micro_batches=num_mb)
                results.append(
                    {
                        "num_stages": num_stages,
                        "num_micro_batches": num_mb,
                        "bubble_fraction": metrics["bubble_fraction"],
                        "utilization": metrics["utilization"],
                        "compute_time_s": metrics["compute_time_s"],
                        "total_pipeline_steps": metrics["total_pipeline_steps"],
                    }
                )

        return pd.DataFrame(results)


# =========================================================================
# Ponto de entrada quando executado como módulo
# =========================================================================

def main() -> None:
    """Executa pipeline completo de demonstração."""
    from src.data_preprocessing import WorkloadPreprocessor

    print("=" * 60)
    print("Aula 07 — Inferência Distribuída e Paralelismo de Modelos")
    print("=" * 60)

    # 1. Gera workload
    print("\n[1/4] Gerando workload sintético...")
    preprocessor = WorkloadPreprocessor(seed=42)
    df = preprocessor.generate_workload(n_requests=200)
    print(f"  → {len(df)} requisições geradas")

    # 2. Continuous vs. Static Batching
    print("\n[2/4] Comparando continuous vs. static batching...")
    runner = InferenceRunner(ServingConfig(max_batch_size=16, max_total_tokens=4096))

    continuous_result = runner.run_continuous_batching(df.head(100))
    print(f"  Continuous batching: {continuous_result['total_completed']} reqs, "
          f"avg latency = {continuous_result['avg_latency_steps']:.1f} steps")

    runner2 = InferenceRunner(ServingConfig(max_batch_size=16))
    static_result = runner2.run_static_batching(df.head(100))
    print(f"  Static batching:    {static_result['total_completed']} reqs, "
          f"avg latency = {static_result['avg_latency_steps']:.1f} steps")

    # 3. Tensor Parallelism Benchmark
    print("\n[3/4] Benchmark de tensor parallelism...")
    tp_results = runner.run_tensor_parallelism_benchmark(
        matrix_sizes=[256, 512, 1024],
        gpu_counts=[1, 2, 4],
    )
    print(tp_results.to_string(index=False))

    # 4. Pipeline Parallelism
    print("\n[4/4] Benchmark de pipeline parallelism...")
    pp_results = runner.run_pipeline_benchmark(
        stage_counts=[2, 4],
        micro_batch_counts=[1, 4, 16],
    )
    print(pp_results.to_string(index=False))

    print("\n✓ Pipeline completo executado com sucesso.")


if __name__ == "__main__":
    main()
