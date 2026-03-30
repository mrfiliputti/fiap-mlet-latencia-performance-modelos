"""
Módulo de avaliação e métricas para sistemas de serving de LLMs.

Implementa métricas de latência e throughput conforme discutidas na
seção 'Estratégias de Auto-scaling e SLOs' do material da aula:
P50, P95, P99 latência, throughput (req/s), e conformidade com SLOs.

Também inclui visualizações para análise comparativa de estratégias
de paralelismo e scheduling.

Referências:
    Kwon, W. et al. (2023). Efficient Memory Management for Large Language
        Model Serving with PagedAttention. SOSP 2023.
    Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ServingMetricsEvaluator:
    """
    Avaliador de métricas para sistemas de serving de LLMs.

    Calcula, visualiza e analisa métricas de latência e throughput
    relevantes para SLOs de produção, conforme seção 'Estratégias
    de Auto-scaling e SLOs' do material da aula.

    Métricas implementadas:
    - Percentis de latência (P50, P95, P99)
    - Throughput (requisições por segundo / por step)
    - Utilização de batch
    - Conformidade com SLOs
    - Análise de pipeline bubbles (Huang et al., 2019)

    Attributes:
        output_dir: Diretório para salvar figuras.
    """

    def __init__(self, output_dir: str = "outputs/figures") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Métricas de Latência
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_latency_percentiles(
        latencies: np.ndarray,
        percentiles: Optional[list[float]] = None,
    ) -> dict[str, float]:
        """Calcula percentis de latência para análise de SLOs.

        Conforme seção 'Estratégias de Auto-scaling e SLOs':
        'P99 latência tipicamente < 500ms para aplicações interativas'.

        Args:
            latencies: Array de latências observadas.
            percentiles: Lista de percentis (default: [50, 95, 99, 99.9]).

        Returns:
            Dicionário com percentis calculados.
        """
        if percentiles is None:
            percentiles = [50.0, 95.0, 99.0, 99.9]

        result: dict[str, float] = {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }
        for p in percentiles:
            result[f"p{p:.0f}" if p == int(p) else f"p{p}"] = float(
                np.percentile(latencies, p)
            )
        return result

    @staticmethod
    def check_slo_compliance(
        latencies: np.ndarray,
        slo_p99_ms: float = 500.0,
        slo_throughput_min: float = 10.0,
        total_time_s: float = 1.0,
    ) -> dict:
        """Verifica conformidade com SLOs de serving.

        Implementa verificação de SLOs conforme seção sobre Auto-scaling:
        'P99 latência < 500ms, throughput mínimo, disponibilidade 99.9%+'.

        Args:
            latencies: Array de latências (ms ou steps).
            slo_p99_ms: SLO de P99 latência.
            slo_throughput_min: Throughput mínimo (req/s).
            total_time_s: Tempo total de observação.

        Returns:
            Dicionário com métricas de compliance.
        """
        p99 = float(np.percentile(latencies, 99))
        throughput = len(latencies) / max(total_time_s, 1e-6)

        return {
            "p99_latency": p99,
            "slo_p99_target": slo_p99_ms,
            "p99_compliant": p99 <= slo_p99_ms,
            "throughput": throughput,
            "slo_throughput_target": slo_throughput_min,
            "throughput_compliant": throughput >= slo_throughput_min,
            "overall_compliant": (p99 <= slo_p99_ms) and (throughput >= slo_throughput_min),
        }

    # ------------------------------------------------------------------
    # Visualizações
    # ------------------------------------------------------------------

    def plot_latency_distribution(
        self,
        latencies: np.ndarray,
        title: str = "Distribuição de Latência",
        slo_p99: Optional[float] = None,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Plota distribuição de latência com percentis e SLO.

        Visualiza métricas discutidas na seção 'Estratégias de Auto-scaling
        e SLOs': P50, P95, P99 com linha de SLO.

        Args:
            latencies: Array de latências.
            title: Título do gráfico.
            slo_p99: Linha de SLO P99 (se fornecido).
            save_name: Nome do arquivo para salvar.

        Returns:
            Figura matplotlib.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histograma
        ax = axes[0]
        ax.hist(latencies, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        ax.axvline(p50, color="green", linestyle="--", label=f"P50={p50:.1f}")
        ax.axvline(p95, color="orange", linestyle="--", label=f"P95={p95:.1f}")
        ax.axvline(p99, color="red", linestyle="--", label=f"P99={p99:.1f}")
        if slo_p99 is not None:
            ax.axvline(slo_p99, color="darkred", linestyle="-", linewidth=2,
                       label=f"SLO P99={slo_p99:.0f}")
        ax.set_xlabel("Latência (steps)")
        ax.set_ylabel("Frequência")
        ax.set_title(f"{title} — Histograma")
        ax.legend(fontsize=9)

        # CDF
        ax = axes[1]
        sorted_lat = np.sort(latencies)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        ax.plot(sorted_lat, cdf, color="steelblue", linewidth=2)
        ax.axhline(0.50, color="green", linestyle=":", alpha=0.5)
        ax.axhline(0.95, color="orange", linestyle=":", alpha=0.5)
        ax.axhline(0.99, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Latência (steps)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{title} — CDF")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_batching_comparison(
        self,
        continuous_history: list[dict],
        static_summary: dict,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Compara continuous vs. static batching.

        Visualiza a diferença de utilização de batch entre os métodos,
        conforme discutido na seção 'Continuous Batching e Scheduling
        Dinâmico' (Yu et al., 2022): continuous batching mantém o batch
        maximamente preenchido.

        Args:
            continuous_history: Histórico step-by-step do continuous batching.
            static_summary: Resumo do static batching.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Utilização do batch ao longo do tempo (continuous)
        ax = axes[0]
        steps = [h["step"] for h in continuous_history]
        batch_sizes = [h["active_batch_size"] for h in continuous_history]
        ax.fill_between(steps, batch_sizes, alpha=0.4, color="steelblue")
        ax.plot(steps, batch_sizes, color="steelblue", linewidth=1,
                label="Continuous Batching")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tamanho do Batch Ativo")
        ax.set_title("Utilização do Batch — Continuous Batching")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Comparação de métricas
        ax = axes[1]
        cb_steps = len(continuous_history)
        sb_steps = static_summary.get("total_steps", 0)
        labels = ["Continuous\nBatching", "Static\nBatching"]
        values = [cb_steps, sb_steps]
        colors = ["steelblue", "coral"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_ylabel("Total de Steps")
        ax.set_title("Steps para Completar Workload")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f"{val}", ha="center", va="bottom", fontsize=11)

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_bubble_fraction(
        self,
        pipeline_results: pd.DataFrame,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Plota fração de pipeline bubble vs. micro-batches.

        Valida empiricamente a fórmula (P-1)/(M+P-1) de Huang et al. (2019)
        — GPipe, discutida na seção 'Pipeline Parallelism e Micro-batching'.

        Args:
            pipeline_results: DataFrame com resultados do benchmark.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for num_stages in sorted(pipeline_results["num_stages"].unique()):
            subset = pipeline_results[pipeline_results["num_stages"] == num_stages]
            ax.plot(
                subset["num_micro_batches"],
                subset["bubble_fraction"],
                marker="o",
                linewidth=2,
                label=f"P={num_stages} estágios",
            )

        ax.set_xlabel("Número de Micro-batches (M)", fontsize=12)
        ax.set_ylabel("Fração de Pipeline Bubble", fontsize=12)
        ax.set_title(
            "Pipeline Bubble: (P-1)/(M+P-1)\n"
            "Conforme GPipe (Huang et al., 2019)",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_tensor_parallelism_speedup(
        self,
        tp_results: pd.DataFrame,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Plota speedup de tensor parallelism vs. ideal.

        Visualiza eficiência de tensor parallelism conforme seção
        'Tensor Parallelism para Large Language Models' (Shoeybi et al., 2019):
        eficiência depende da relação computação/comunicação.

        Args:
            tp_results: DataFrame com resultados do benchmark.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Speedup
        ax = axes[0]
        for dim in sorted(tp_results["matrix_dim"].unique()):
            subset = tp_results[tp_results["matrix_dim"] == dim]
            ax.plot(
                subset["num_gpus"],
                subset["speedup"],
                marker="o",
                linewidth=2,
                label=f"dim={dim}",
            )
        # Ideal linear
        max_gpus = tp_results["num_gpus"].max()
        ax.plot(
            [1, max_gpus],
            [1, max_gpus],
            "k--",
            alpha=0.5,
            label="Ideal (linear)",
        )
        ax.set_xlabel("Número de GPUs", fontsize=12)
        ax.set_ylabel("Speedup", fontsize=12)
        ax.set_title("Tensor Parallelism — Speedup\n(Shoeybi et al., 2019)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Eficiência
        ax = axes[1]
        for dim in sorted(tp_results["matrix_dim"].unique()):
            subset = tp_results[tp_results["matrix_dim"] == dim]
            ax.plot(
                subset["num_gpus"],
                subset["efficiency"],
                marker="s",
                linewidth=2,
                label=f"dim={dim}",
            )
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.3, label="Ideal (100%)")
        ax.set_xlabel("Número de GPUs", fontsize=12)
        ax.set_ylabel("Eficiência Paralela", fontsize=12)
        ax.set_title("Tensor Parallelism — Eficiência")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_speculative_decoding_analysis(
        self,
        results: list[dict],
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Plota análise de speculative decoding.

        Visualiza speedup e eficiência do speculative decoding conforme
        Leviathan et al. (2023), seção 'Speculative Decoding para
        Redução de Latência'.

        Args:
            results: Lista de dicts com métricas de cada configuração.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        tokens = [r["total_tokens"] for r in results]
        speedups = [r["speedup"] for r in results]
        auto_lat = [r["autoregressive_latency_ms"] for r in results]
        spec_lat = [r["speculative_latency_ms"] for r in results]

        # Speedup
        ax = axes[0]
        ax.bar(range(len(tokens)), speedups, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([str(t) for t in tokens])
        ax.set_xlabel("Tokens Gerados")
        ax.set_ylabel("Speedup")
        ax.set_title("Speculative Decoding — Speedup\n(Leviathan et al., 2023)")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Baseline (1×)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Latência comparativa
        ax = axes[1]
        x = np.arange(len(tokens))
        width = 0.35
        ax.bar(x - width / 2, auto_lat, width, label="Autoregressivo", color="coral",
               alpha=0.8)
        ax.bar(x + width / 2, spec_lat, width, label="Speculative", color="steelblue",
               alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in tokens])
        ax.set_xlabel("Tokens Gerados")
        ax.set_ylabel("Latência Estimada (ms)")
        ax.set_title("Latência: Autoregressivo vs. Speculative")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_kv_cache_comparison(
        self,
        sequence_lengths: np.ndarray,
        block_size: int = 16,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Compara desperdício entre alocação contígua e PagedAttention.

        Visualiza o benefício de PagedAttention vs. alocação contígua fixa,
        conforme seção 'PagedAttention e Gerenciamento de KV-Cache':
        'Alocação contígua: ~50% utilização → PagedAttention: >95%'.

        Args:
            sequence_lengths: Comprimentos de sequências observados.
            block_size: Tamanho de bloco PagedAttention.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        max_seq = int(np.max(sequence_lengths))

        # Contígua: aloca max_seq para todas
        contig_allocated = np.full_like(sequence_lengths, max_seq)
        contig_waste = contig_allocated - sequence_lengths
        contig_utilization = sequence_lengths / contig_allocated

        # Paged: aloca ceil(n/B) * B
        paged_allocated = np.ceil(sequence_lengths / block_size).astype(int) * block_size
        paged_waste = paged_allocated - sequence_lengths
        paged_utilization = sequence_lengths / paged_allocated

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Boxplot de utilização
        ax = axes[0]
        data = pd.DataFrame({
            "Contígua (max_seq)": contig_utilization,
            f"PagedAttention (B={block_size})": paged_utilization,
        })
        sns.boxplot(data=data, ax=ax, palette=["coral", "steelblue"])
        ax.set_ylabel("Utilização de Memória")
        ax.set_title("Utilização: Contígua vs. PagedAttention\n(Kwon et al., 2023)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        # Desperdício total
        ax = axes[1]
        labels = ["Contígua", "PagedAttention"]
        waste_totals = [np.sum(contig_waste), np.sum(paged_waste)]
        colors = ["coral", "steelblue"]
        bars = ax.bar(labels, waste_totals, color=colors, alpha=0.8)
        ax.set_ylabel("Tokens Desperdiçados (Total)")
        ax.set_title("Desperdício de Memória")
        for bar, val in zip(bars, waste_totals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig

    def plot_amdahl_gustafson(
        self,
        serial_fractions: Optional[list[float]] = None,
        max_processors: int = 64,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Compara Lei de Amdahl vs. Gustafson.

        Visualiza os limites de speedup discutidos na seção
        'Fundamentos de Paralelismo em Deep Learning': Amdahl representa
        o limite pessimista, Gustafson o otimista para workloads escaláveis.

        Args:
            serial_fractions: Frações seriais a plotar.
            max_processors: Máximo de processadores no eixo x.
            save_name: Nome para salvar.

        Returns:
            Figura matplotlib.
        """
        from src.model import amdahl_speedup, gustafson_speedup

        if serial_fractions is None:
            serial_fractions = [0.01, 0.05, 0.10, 0.25]

        processors = np.arange(1, max_processors + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Amdahl
        ax = axes[0]
        for s in serial_fractions:
            speedups = [amdahl_speedup(s, p) for p in processors]
            ax.plot(processors, speedups, linewidth=2, label=f"s={s:.0%}")
        ax.plot(processors, processors, "k--", alpha=0.3, label="Ideal")
        ax.set_xlabel("Número de Processadores")
        ax.set_ylabel("Speedup")
        ax.set_title("Lei de Amdahl")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Gustafson
        ax = axes[1]
        for s in serial_fractions:
            speedups = [gustafson_speedup(s, p) for p in processors]
            ax.plot(processors, speedups, linewidth=2, label=f"s={s:.0%}")
        ax.plot(processors, processors, "k--", alpha=0.3, label="Ideal")
        ax.set_xlabel("Número de Processadores")
        ax.set_ylabel("Speedup (Gustafson)")
        ax.set_title("Lei de Gustafson (1988)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            path = os.path.join(self.output_dir, save_name)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig
