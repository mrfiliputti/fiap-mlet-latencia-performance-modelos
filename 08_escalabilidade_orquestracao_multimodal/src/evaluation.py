"""
Avaliação de SLOs, error budgets e métricas SRE.

Funções de avaliação alinhadas com a seção 'Fundamentos de SRE para
Machine Learning' e 'Monitoramento de Latência e Observabilidade em
Produção' do material da aula.

Inclui visualizações de dashboards de SLO (inspiradas na Figura 2),
gráficos de latência por percentil e análise de error budgets.

Referências:
    Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
    Sridharan, C. (2018). Distributed Systems Observability. O'Reilly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model import SLOMonitor, SLOResult


# ================================================================== #
#  Cálculo de Métricas                                                 #
# ================================================================== #


def calculate_metrics(
    df: pd.DataFrame,
    slo_latency_p99_ms: float = 200.0,
    slo_availability: float = 0.999,
) -> Dict[str, Any]:
    """Calcula métricas abrangentes de SRE para serving de ML.

    Produz SLIs, avaliação de SLOs e error budgets conforme o
    framework SRE de Beyer et al. (2016).

    Args:
        df: DataFrame com métricas de serving.
        slo_latency_p99_ms: Meta de latência P99 em ms.
        slo_availability: Meta de disponibilidade.

    Returns:
        Dicionário com SLIs, SLOs, error budgets e métricas auxiliares.
    """
    monitor = SLOMonitor(
        slo_latency_p99_ms=slo_latency_p99_ms,
        slo_availability=slo_availability,
    )
    slis = monitor.calculate_slis(df)
    slos = monitor.evaluate_slos(df)

    # Métricas por modalidade
    per_modality: Dict[str, Dict[str, float]] = {}
    for mod in df["modality"].unique():
        mod_df = df[df["modality"] == mod]
        per_modality[mod] = {
            "count": len(mod_df),
            "latency_p50": float(np.percentile(mod_df["total_latency_ms"], 50)),
            "latency_p99": float(np.percentile(mod_df["total_latency_ms"], 99)),
            "error_rate": float(mod_df["is_error"].mean()),
        }

    # Métricas por versão
    per_version: Dict[str, Dict[str, float]] = {}
    for ver in df["model_version"].unique():
        ver_df = df[df["model_version"] == ver]
        per_version[ver] = {
            "count": len(ver_df),
            "latency_p50": float(np.percentile(ver_df["total_latency_ms"], 50)),
            "latency_p99": float(np.percentile(ver_df["total_latency_ms"], 99)),
            "error_rate": float(ver_df["is_error"].mean()),
        }

    return {
        "slis": slis,
        "slos": [
            {
                "name": s.name,
                "target": s.target,
                "actual": round(s.actual, 4),
                "met": s.met,
                "error_budget_remaining_pct": round(s.error_budget_pct, 2),
            }
            for s in slos
        ],
        "per_modality": per_modality,
        "per_version": per_version,
        "total_requests": len(df),
    }


def calculate_error_budget(
    total_requests: int,
    error_count: int,
    slo_availability: float = 0.999,
    window_days: int = 30,
) -> Dict[str, float]:
    """Calcula error budget restante.

    O error budget é a diferença entre o máximo de erros permitido
    pelo SLO e o número de erros observados, conforme descrito na
    seção 'Fundamentos de SRE para Machine Learning':

        Error Budget = SLO_target × total — erros observados

    Args:
        total_requests: Total de requisições no período.
        error_count: Erros observados.
        slo_availability: Meta de disponibilidade.
        window_days: Janela de avaliação.

    Returns:
        Dicionário com budget total, consumido e restante.
    """
    allowed_errors = total_requests * (1 - slo_availability)
    consumed = min(error_count, allowed_errors)
    remaining = max(0.0, allowed_errors - error_count)
    pct_remaining = (remaining / allowed_errors * 100) if allowed_errors > 0 else 0.0

    total_minutes = window_days * 24 * 60
    downtime_budget_min = total_minutes * (1 - slo_availability)
    downtime_consumed_min = downtime_budget_min * (consumed / allowed_errors) if allowed_errors > 0 else 0.0

    return {
        "allowed_errors": round(allowed_errors, 2),
        "observed_errors": error_count,
        "budget_remaining": round(remaining, 2),
        "budget_remaining_pct": round(pct_remaining, 2),
        "downtime_budget_minutes": round(downtime_budget_min, 2),
        "downtime_consumed_minutes": round(downtime_consumed_min, 2),
    }


# ================================================================== #
#  Visualizações                                                       #
# ================================================================== #


def plot_latency_distribution(
    df: pd.DataFrame,
    slo_target_ms: float = 200.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota distribuição de latência com linha de SLO.

    Inspirado no dashboard de SLO da Figura 2 do material da aula.

    Args:
        df: DataFrame com coluna ``total_latency_ms``.
        slo_target_ms: Linha vertical do SLO.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma geral
    ax = axes[0]
    ax.hist(df["total_latency_ms"], bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(slo_target_ms, color="red", linestyle="--", linewidth=2, label=f"SLO P99 = {slo_target_ms} ms")
    p99 = np.percentile(df["total_latency_ms"], 99)
    ax.axvline(p99, color="orange", linestyle=":", linewidth=2, label=f"P99 real = {p99:.1f} ms")
    ax.set_xlabel("Latência (ms)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Latência — Todas as Modalidades")
    ax.legend()

    # Box-plot por modalidade
    ax2 = axes[1]
    df.boxplot(column="total_latency_ms", by="modality", ax=ax2)
    ax2.axhline(slo_target_ms, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Modalidade")
    ax2.set_ylabel("Latência (ms)")
    ax2.set_title("Latência por Modalidade")
    plt.suptitle("")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_budget_gauge(
    budget_remaining_pct: float,
    slo_name: str = "Disponibilidade 99.9 %",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota indicador de error budget estilo gauge.

    Implementa visualização similar ao painel 'ERROR BUDGET' da
    Figura 2 do material da aula.

    Args:
        budget_remaining_pct: Percentual restante do error budget.
        slo_name: Nome do SLO para título.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    consumed = 100 - budget_remaining_pct

    colors = ["#e74c3c" if consumed > 80 else "#f39c12" if consumed > 50 else "#2ecc71"]
    ax.barh(["Error Budget"], [consumed], color=colors, height=0.4, label="Consumido")
    ax.barh(["Error Budget"], [budget_remaining_pct], left=[consumed],
            color="#ecf0f1", height=0.4, edgecolor="#bdc3c7", label="Restante")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% do Error Budget")
    ax.set_title(f"Error Budget — {slo_name}")
    ax.legend(loc="lower right")

    ax.text(consumed / 2, 0, f"{consumed:.1f}%", ha="center", va="center", fontweight="bold", fontsize=12)
    ax.text(consumed + budget_remaining_pct / 2, 0, f"{budget_remaining_pct:.1f}%",
            ha="center", va="center", fontsize=11, color="#7f8c8d")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_canary_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota comparação de latência entre versões stable e canary.

    Visualiza as distribuições lado a lado para apoiar a decisão
    de canary deployment (seção 'Canary Deployments e Testes
    Estatísticos em Produção').

    Args:
        df: DataFrame com colunas ``model_version``, ``total_latency_ms``.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma sobreposto
    ax = axes[0]
    for ver, color in [("v1-stable", "steelblue"), ("v2-canary", "coral")]:
        subset = df[df["model_version"] == ver]
        ax.hist(subset["total_latency_ms"], bins=60, alpha=0.6,
                label=ver, color=color, edgecolor="white")
    ax.set_xlabel("Latência (ms)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Latência — Stable vs Canary")
    ax.legend()

    # Violin plot
    ax2 = axes[1]
    sns.violinplot(data=df, x="model_version", y="total_latency_ms", ax=ax2, palette="Set2")
    ax2.set_xlabel("Versão do Modelo")
    ax2.set_ylabel("Latência (ms)")
    ax2.set_title("Violin Plot — Stable vs Canary")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scaling_timeline(
    df: pd.DataFrame,
    events: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota timeline de auto-scaling com eventos de scale up/down.

    Conecta o conceito de auto-scaling baseado em métricas customizadas
    (seção sobre Azure ML Managed Endpoints) com visualização prática.

    Args:
        df: DataFrame com colunas ``timestamp``, ``simulated_replicas``,
            ``cpu_utilization``, ``total_latency_ms``.
        events: Lista de ScalingEvent.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Réplicas
    ax = axes[0]
    ax.plot(df["timestamp"], df["simulated_replicas"], color="steelblue", linewidth=1.5)
    for ev in events:
        if ev.direction == "scale_up":
            ax.axvline(df["timestamp"].iloc[ev.timestamp_idx], color="green", alpha=0.3, linewidth=0.8)
        elif ev.direction == "scale_down":
            ax.axvline(df["timestamp"].iloc[ev.timestamp_idx], color="red", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Réplicas")
    ax.set_title("Auto-Scaling — Réplicas ao Longo do Tempo")

    # CPU
    ax2 = axes[1]
    ax2.plot(df["timestamp"], df["cpu_utilization"], color="orange", alpha=0.6, linewidth=0.8)
    ax2.axhline(0.70, color="red", linestyle="--", label="Threshold scale-up (70%)")
    ax2.axhline(0.30, color="green", linestyle="--", label="Threshold scale-down (30%)")
    ax2.set_ylabel("CPU Utilization")
    ax2.legend(fontsize=8)

    # Latência
    ax3 = axes[2]
    ax3.plot(df["timestamp"], df["total_latency_ms"], color="purple", alpha=0.4, linewidth=0.5)
    ax3.axhline(200, color="red", linestyle="--", label="SLO P99 = 200ms")
    ax3.set_ylabel("Latência (ms)")
    ax3.set_xlabel("Tempo")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_slo_dashboard(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota dashboard de SLOs inspirado na Figura 2 do material.

    Apresenta visão consolidada de SLIs, SLOs e error budgets.

    Args:
        metrics: Resultado de ``calculate_metrics()``.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    slis = metrics["slis"]
    slos = metrics["slos"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Painel 1: Percentis de latência
    ax = axes[0, 0]
    percentiles = ["P50", "P95", "P99", "Max"]
    values = [slis["latency_p50"], slis["latency_p95"], slis["latency_p99"], slis["latency_max"]]
    colors = ["#2ecc71" if v < 200 else "#e74c3c" for v in values]
    ax.bar(percentiles, values, color=colors, edgecolor="white")
    ax.axhline(200, color="red", linestyle="--", label="SLO = 200ms")
    ax.set_ylabel("Latência (ms)")
    ax.set_title("Percentis de Latência")
    ax.legend()

    # Painel 2: Error rate e availability
    ax2 = axes[0, 1]
    avail_slo = next((s for s in slos if s["name"] == "availability"), None)
    if avail_slo:
        ax2.bar(["Disponibilidade"], [avail_slo["actual"] * 100],
                color="#2ecc71" if avail_slo["met"] else "#e74c3c")
        ax2.axhline(avail_slo["target"] * 100, color="red", linestyle="--",
                     label=f"SLO = {avail_slo['target']*100:.1f}%")
        ax2.set_ylim(99, 100.05)
        ax2.set_ylabel("Disponibilidade (%)")
        ax2.set_title("Disponibilidade")
        ax2.legend()

    # Painel 3: Latência por modalidade
    ax3 = axes[1, 0]
    mod_data = metrics["per_modality"]
    mods = list(mod_data.keys())
    p99s = [mod_data[m]["latency_p99"] for m in mods]
    ax3.barh(mods, p99s, color=["#3498db", "#e67e22", "#9b59b6"])
    ax3.axvline(200, color="red", linestyle="--")
    ax3.set_xlabel("Latência P99 (ms)")
    ax3.set_title("P99 por Modalidade")

    # Painel 4: Error budget
    ax4 = axes[1, 1]
    for slo in slos:
        remaining = slo["error_budget_remaining_pct"]
        consumed = 100 - remaining
        color = "#e74c3c" if consumed > 80 else "#f39c12" if consumed > 50 else "#2ecc71"
        ax4.barh([slo["name"]], [consumed], color=color, label="Consumido")
        ax4.barh([slo["name"]], [remaining], left=[consumed], color="#ecf0f1", edgecolor="#bdc3c7")
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Error Budget (%)")
    ax4.set_title("Error Budget Consumido")

    plt.suptitle("SLO Dashboard — Inferência Multimodal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
