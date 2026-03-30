"""
Módulo de avaliação de performance e latência de modelos de ML.

Implementa métricas de qualidade (accuracy, F1, AUC-ROC) e métricas de
performance (latência, throughput, percentis), conforme discutidas na
seção 'Percentis de Latência' e 'Complexidade de Modelos' da aula.

Referência:
    Dean, J. & Barroso, L. A. (2013). The Tail at Scale.
    Communications of the ACM, 56(2), 74-80.
    Reddi, V. J. et al. (2020). MLPerf Inference Benchmark. ISCA '20.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model import LatencyBenchmarkModel


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> dict[str, float]:
    """
    Calcula métricas de qualidade de classificação.

    Implementa as métricas padrão de avaliação de classificadores conforme
    discutido na seção 'Avaliação de Modelos'. Inclui métricas que balanceiam
    precision e recall (F1-score).

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        y_prob: Probabilidades preditas (para AUC-ROC).

    Returns:
        Dicionário com accuracy, precision, recall, f1 e opcionalmente auc_roc.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
    }

    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def calculate_latency_metrics(
    model: LatencyBenchmarkModel,
    X: np.ndarray,
    n_iterations: int = 100,
) -> dict[str, float]:
    """
    Calcula métricas de latência de inferência.

    Implementa o protocolo de medição de latência conforme o MLPerf
    Inference Benchmark (Reddi et al., 2020): aquecimento + múltiplas
    iterações + cálculo de percentis.

    Percentis são fundamentais em produção conforme Dean & Barroso (2013):
    - P50: Experiência do usuário típico
    - P95: Performance sob carga
    - P99: Worst-case para SLAs

    Args:
        model: Modelo treinado com benchmarking.
        X: Features para inferência.
        n_iterations: Número de iterações.

    Returns:
        Dicionário com estatísticas de latência (ms) e throughput.
    """
    return model.benchmark_inference(X, n_iterations=n_iterations)


def benchmark_batch_sizes(
    model: LatencyBenchmarkModel,
    X: np.ndarray,
    batch_sizes: Optional[list[int]] = None,
    n_iterations: int = 50,
) -> pd.DataFrame:
    """
    Avalia o impacto do batch size na latência e throughput.

    Implementa a análise de trade-off entre batch size, latência e throughput.
    Conforme discutido na seção 'Throughput vs. Latência', batching aumenta
    throughput mas pode aumentar a latência individual.

    Args:
        model: Modelo treinado.
        X: Features completas para amostragem de batches.
        batch_sizes: Lista de tamanhos de batch a testar.
        n_iterations: Iterações por batch size.

    Returns:
        DataFrame com batch_size, mean_latency_ms, p95_ms, throughput.
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000]

    results = []
    for bs in batch_sizes:
        if bs > X.shape[0]:
            continue

        X_batch = X[:bs]
        latencies: list[float] = []

        # Warm-up
        for _ in range(3):
            model.model.predict(X_batch)

        for _ in range(n_iterations):
            start = time.perf_counter()
            model.model.predict(X_batch)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)

        arr = np.array(latencies)
        results.append(
            {
                "batch_size": bs,
                "mean_latency_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "throughput_samples_per_sec": bs / (float(np.mean(arr)) / 1000),
                "latency_per_sample_ms": float(np.mean(arr)) / bs,
            }
        )

    return pd.DataFrame(results)


def compare_models_latency(
    models: list[LatencyBenchmarkModel],
    X: np.ndarray,
    n_iterations: int = 100,
) -> pd.DataFrame:
    """
    Compara latência de inferência entre múltiplos modelos.

    Implementa o conceito de benchmarking comparativo conforme a seção
    'Complexidade de Modelos', onde diferentes algoritmos são avaliados
    sob as mesmas condições para comparação justa.

    Args:
        models: Lista de modelos treinados.
        X: Features para inferência.
        n_iterations: Número de iterações por modelo.

    Returns:
        DataFrame comparativo com métricas de latência por modelo.
    """
    results = []
    for model in models:
        benchmark = model.benchmark_inference(X, n_iterations=n_iterations)
        results.append(benchmark)

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Funções de Visualização
# ------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gera matriz de confusão com visualização.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Classe 0", "Classe 1"],
        yticklabels=["Classe 0", "Classe 1"],
        ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusão", fontsize=14)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Modelo",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gera curva ROC com AUC.

    Args:
        y_true: Labels verdadeiros.
        y_prob: Probabilidades preditas para classe positiva.
        model_name: Nome do modelo para legenda.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Curva ROC", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latency_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gera gráfico comparativo de latência entre modelos.

    Visualiza P50, P95 e P99 lado a lado, implementando a análise de
    percentis conforme Dean & Barroso (2013), "The Tail at Scale".

    Args:
        comparison_df: DataFrame de compare_models_latency().
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = comparison_df["model"].values
    x = np.arange(len(models))

    # Gráfico 1: Latência média com barras de erro
    axes[0].bar(x, comparison_df["mean_ms"], yerr=comparison_df["std_ms"],
                capsize=5, color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].set_ylabel("Latência (ms)")
    axes[0].set_title("Latência Média de Inferência")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Gráfico 2: Percentis de latência
    width = 0.25
    axes[1].bar(x - width, comparison_df["p50_ms"], width, label="P50", alpha=0.8)
    axes[1].bar(x, comparison_df["p95_ms"], width, label="P95", alpha=0.8)
    axes[1].bar(x + width, comparison_df["p99_ms"], width, label="P99", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    axes[1].set_ylabel("Latência (ms)")
    axes[1].set_title("Percentis de Latência (P50/P95/P99)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Gráfico 3: Throughput
    axes[2].barh(x, comparison_df["throughput_samples_per_sec"],
                 color="seagreen", alpha=0.8)
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(models)
    axes[2].set_xlabel("Amostras/segundo")
    axes[2].set_title("Throughput de Inferência")
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle("Comparação de Performance entre Modelos", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_batch_size_analysis(
    batch_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gera análise visual do impacto do batch size na latência e throughput.

    Implementa a visualização do trade-off latência vs. throughput conforme
    discutido na seção 'Throughput vs. Latência'.

    Args:
        batch_df: DataFrame de benchmark_batch_sizes().
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Latência total vs batch size
    axes[0].plot(
        batch_df["batch_size"],
        batch_df["mean_latency_ms"],
        "o-",
        color="steelblue",
        linewidth=2,
        label="Latência Média",
    )
    axes[0].fill_between(
        batch_df["batch_size"],
        batch_df["p50_ms"],
        batch_df["p99_ms"],
        alpha=0.2,
        color="steelblue",
        label="P50-P99",
    )
    axes[0].set_xlabel("Batch Size", fontsize=12)
    axes[0].set_ylabel("Latência Total (ms)", fontsize=12)
    axes[0].set_title("Latência vs. Batch Size", fontsize=13)
    axes[0].legend()
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    # Gráfico 2: Throughput vs batch size
    axes[1].plot(
        batch_df["batch_size"],
        batch_df["throughput_samples_per_sec"],
        "s-",
        color="seagreen",
        linewidth=2,
    )
    axes[1].set_xlabel("Batch Size", fontsize=12)
    axes[1].set_ylabel("Throughput (amostras/s)", fontsize=12)
    axes[1].set_title("Throughput vs. Batch Size", fontsize=13)
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Trade-off Latência vs. Throughput — Efeito do Batch Size",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
