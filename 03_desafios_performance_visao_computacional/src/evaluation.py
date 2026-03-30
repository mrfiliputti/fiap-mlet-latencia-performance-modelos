"""
Módulo de avaliação e visualização de performance de visão computacional.

Implementa métricas de avaliação, análise de fronteira de Pareto e
visualizações de trade-offs entre acurácia, latência e custo computacional,
conforme discutido nas seções 4 ('Impacto da Resolução') e 10 ('Fronteira
de Pareto e Seleção de Arquiteturas') do material.

Referência:
    Tan, M. & Le, Q. V. (2019). EfficientNet. ICML 2019.
    Cai, H. et al. (2020). Once-for-All. ICLR 2020.
    Cai, H. et al. (2023). EfficientViT. CVPR 2023.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# =========================================================================
# Métricas de classificação
# =========================================================================


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Calcula métricas de classificação completas.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Predições do modelo.
        class_names: Nomes das classes (opcional).

    Returns:
        Dicionário com accuracy, precision, recall, f1 e
        classification_report.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        ),
    }
    return metrics


# =========================================================================
# Visualizações de performance — seção 10 (Fronteira de Pareto)
# =========================================================================


def plot_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "latency_mean_ms",
    y_col: str = "top1_accuracy",
    color_col: str = "model_family",
    size_col: str = "params_millions",
    title: str = "Fronteira de Pareto: Acurácia vs Latência",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota fronteira de Pareto entre acurácia e latência.

    Implementa visualização da fronteira de Pareto conforme seção 10
    do material: identifica o conjunto de soluções onde não é possível
    melhorar acurácia sem aumentar latência (Tan & Le, 2019).

    Args:
        df: DataFrame com dados agregados de benchmark.
        x_col: Coluna para eixo x (latência).
        y_col: Coluna para eixo y (acurácia).
        color_col: Coluna para cor/família de modelo.
        size_col: Coluna para tamanho do ponto (parâmetros).
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Scatter plot por família
    families = df[color_col].unique()
    palette = {"cnn": "#2196F3", "vit": "#F44336", "hybrid": "#4CAF50"}
    markers = {"cnn": "o", "vit": "s", "hybrid": "D"}

    for family in families:
        subset = df[df[color_col] == family]
        sizes = subset[size_col] * 3 if size_col in df.columns else 80
        ax.scatter(
            subset[x_col], subset[y_col],
            s=sizes, alpha=0.6,
            color=palette.get(family, "#9E9E9E"),
            marker=markers.get(family, "o"),
            label=family.upper(),
            edgecolors="white", linewidth=0.5,
        )

    # Calcular e plotar fronteira de Pareto
    pareto_points = _find_pareto_points(
        df[x_col].values, df[y_col].values, maximize_y=True,
    )
    pareto_df = df.iloc[pareto_points].sort_values(x_col)
    ax.plot(
        pareto_df[x_col], pareto_df[y_col],
        "k--", alpha=0.5, linewidth=1.5, label="Fronteira de Pareto",
    )

    # Anotar pontos relevantes
    if "model_name" in df.columns:
        for _, row in pareto_df.iterrows():
            ax.annotate(
                row["model_name"],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.8,
            )

    ax.set_xlabel("Latência (ms)", fontsize=12)
    ax.set_ylabel("Acurácia Top-1 (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figura salva em %s", save_path)

    return fig


def _find_pareto_points(
    x: np.ndarray,
    y: np.ndarray,
    maximize_y: bool = True,
) -> list[int]:
    """
    Encontra pontos na fronteira de Pareto.

    Um ponto é Pareto-ótimo se nenhum outro ponto tem x menor (latência)
    E y maior (acurácia) simultaneamente.

    Args:
        x: Valores do eixo x (a minimizar, e.g., latência).
        y: Valores do eixo y (a maximizar, e.g., acurácia).
        maximize_y: Se True, maximiza y; se False, minimiza.

    Returns:
        Índices dos pontos Pareto-ótimos.
    """
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maximize_y:
                # j domina i se j tem menor x E maior y
                if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                    pareto_mask[i] = False
                    break
            else:
                if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                    pareto_mask[i] = False
                    break

    return list(np.where(pareto_mask)[0])


def plot_resolution_impact(
    df: pd.DataFrame,
    model_name: str = "efficientnet_b0",
    hardware: str = "v100",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza impacto da resolução em FLOPs e latência para um modelo.

    Conforme seção 4 do material: dobrar a resolução aumenta FLOPs
    por fator ~4 (crescimento quadrático O(r²)) em camadas convolucionais
    (He & Sun, 2015; Howard et al., 2017).

    Args:
        df: DataFrame com dados de benchmark.
        model_name: Modelo a analisar.
        hardware: Hardware a filtrar.
        save_path: Caminho para salvar figura.

    Returns:
        Figura matplotlib.
    """
    subset = df[(df["model_name"] == model_name) & (df["hardware"] == hardware)]

    if subset.empty:
        raise ValueError(f"Sem dados para {model_name} em {hardware}.")

    agg = subset.groupby("input_resolution").agg(
        flops_giga=("flops_giga", "mean"),
        latency_ms=("latency_ms", "mean"),
        memory_mb=("memory_mb", "mean"),
        top1_accuracy=("top1_accuracy", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FLOPs vs Resolução
    axes[0, 0].plot(agg["input_resolution"], agg["flops_giga"], "o-", color="#2196F3")
    axes[0, 0].set_xlabel("Resolução de Entrada")
    axes[0, 0].set_ylabel("FLOPs (GFLOPs)")
    axes[0, 0].set_title("FLOPs vs Resolução (crescimento ≈ O(r²))")
    axes[0, 0].grid(True, alpha=0.3)

    # Latência vs Resolução
    axes[0, 1].plot(agg["input_resolution"], agg["latency_ms"], "s-", color="#F44336")
    axes[0, 1].set_xlabel("Resolução de Entrada")
    axes[0, 1].set_ylabel("Latência (ms)")
    axes[0, 1].set_title("Latência vs Resolução")
    axes[0, 1].grid(True, alpha=0.3)

    # Memória vs Resolução
    axes[1, 0].plot(agg["input_resolution"], agg["memory_mb"], "D-", color="#4CAF50")
    axes[1, 0].set_xlabel("Resolução de Entrada")
    axes[1, 0].set_ylabel("Memória (MB)")
    axes[1, 0].set_title("Memória vs Resolução")
    axes[1, 0].grid(True, alpha=0.3)

    # Acurácia vs Resolução
    axes[1, 1].plot(agg["input_resolution"], agg["top1_accuracy"], "^-", color="#FF9800")
    axes[1, 1].set_xlabel("Resolução de Entrada")
    axes[1, 1].set_ylabel("Acurácia Top-1 (%)")
    axes[1, 1].set_title("Acurácia vs Resolução (retornos decrescentes)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Impacto da Resolução — {model_name} ({hardware})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_family_comparison(
    df: pd.DataFrame,
    metric: str = "latency_mean_ms",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compara distribuições de uma métrica por família de modelo.

    Visualiza diferenças de performance entre CNNs, ViTs e Híbridos,
    conforme discussão das seções 2 e 3 do material.

    Args:
        df: DataFrame agregado com coluna 'model_family'.
        metric: Métrica para comparação.
        save_path: Caminho para salvar figura.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {"cnn": "#2196F3", "vit": "#F44336", "hybrid": "#4CAF50"}
    sns.boxplot(data=df, x="model_family", y=metric, palette=palette, ax=ax)
    sns.stripplot(
        data=df, x="model_family", y=metric,
        color="black", alpha=0.3, size=3, ax=ax,
    )

    labels_map = {
        "latency_mean_ms": "Latência Média (ms)",
        "flops_giga": "FLOPs (GFLOPs)",
        "top1_accuracy": "Acurácia Top-1 (%)",
        "params_millions": "Parâmetros (M)",
        "throughput_mean": "Throughput (imgs/s)",
    }
    ax.set_ylabel(labels_map.get(metric, metric), fontsize=12)
    ax.set_xlabel("Família de Modelo", fontsize=12)
    ax.set_title(
        f"Comparação CNN vs ViT vs Híbrido — {labels_map.get(metric, metric)}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Matriz de Confusão",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza matriz de confusão como heatmap.

    Args:
        cm: Matriz de confusão (numpy array).
        class_names: Nomes das classes.
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if class_names is None:
        class_names = [f"Classe {i}" for i in range(len(cm))]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict[str, list],
    title: str = "Histórico de Treinamento",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza curvas de treinamento (loss e accuracy).

    Args:
        history: Dicionário retornado por train_model().
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Treino", markersize=4)
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "r-s", label="Validação", markersize=4)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss por Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Treino", markersize=4)
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "r-s", label="Validação", markersize=4)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.set_title("Acurácia por Época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_benchmark_comparison(
    results: list[dict[str, Any]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza resultados de benchmark comparativo de arquiteturas.

    Args:
        results: Lista de dicionários retornados por benchmark_architectures().
        save_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Latência por modelo e resolução
    for model in df["model_name"].unique():
        subset = df[df["model_name"] == model]
        axes[0].plot(
            subset["input_size"], subset["latency_mean_ms"],
            "o-", label=model, markersize=6,
        )
    axes[0].set_xlabel("Resolução de Entrada")
    axes[0].set_ylabel("Latência (ms)")
    axes[0].set_title("Latência vs Resolução")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # FLOPs por modelo e resolução
    if "flops_gflops" in df.columns:
        for model in df["model_name"].unique():
            subset = df[df["model_name"] == model]
            axes[1].plot(
                subset["input_size"], subset["flops_gflops"],
                "s-", label=model, markersize=6,
            )
    axes[1].set_xlabel("Resolução de Entrada")
    axes[1].set_ylabel("FLOPs (GFLOPs)")
    axes[1].set_title("FLOPs vs Resolução")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Throughput por modelo e resolução
    for model in df["model_name"].unique():
        subset = df[df["model_name"] == model]
        axes[2].plot(
            subset["input_size"], subset["throughput_imgs_sec"],
            "^-", label=model, markersize=6,
        )
    axes[2].set_xlabel("Resolução de Entrada")
    axes[2].set_ylabel("Throughput (imgs/s)")
    axes[2].set_title("Throughput vs Resolução")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_architecture_report(
    df: pd.DataFrame,
    latency_constraint_ms: float = 10.0,
    min_accuracy: float = 70.0,
) -> pd.DataFrame:
    """
    Gera relatório de seleção de arquiteturas com base em restrições.

    Filtra e classifica arquiteturas que atendem restrições de latência
    e acurácia mínima, conforme metodologia da seção 10 do material.

    Args:
        df: DataFrame agregado com métricas de benchmark.
        latency_constraint_ms: Latência máxima aceitável (ms).
        min_accuracy: Acurácia mínima aceitável (%).

    Returns:
        DataFrame filtrado e ordenado por eficiência.
    """
    if "latency_mean_ms" not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'latency_mean_ms'.")

    filtered = df[
        (df["latency_mean_ms"] <= latency_constraint_ms) &
        (df["top1_accuracy"] >= min_accuracy)
    ].copy()

    # Score de eficiência: acurácia / latência
    filtered["efficiency_score"] = (
        filtered["top1_accuracy"] / filtered["latency_mean_ms"]
    )
    filtered = filtered.sort_values("efficiency_score", ascending=False)

    return filtered.reset_index(drop=True)
