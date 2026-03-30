"""
Avaliação e benchmarking para Transfer Learning Eficiente.

Implementa métricas de classificação, benchmarking de latência e
visualizações para comparar técnicas PEFT (LoRA, QLoRA, Adapters),
conforme discutido na Tabela 2 e seção 'Otimização de Grafos
Computacionais' do material da Aula 5.

Referências:
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation. ICLR.
    Dettmers, T. et al. (2023). QLoRA. NeurIPS.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Estilo padrão
sns.set_theme(style="whitegrid", palette="deep")


# ---------------------------------------------------------------------------
# Métricas de classificação
# ---------------------------------------------------------------------------


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas completas de classificação.

    Computa accuracy, precision, recall e F1 (macro e por classe),
    métricas padrão para avaliar modelos fine-tunados com PEFT
    conforme benchmarks GLUE/SuperGLUE citados na Tabela 2 do material.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        label_names: Nomes das classes (opcional).

    Returns:
        Dicionário com accuracy, precision, recall, f1 (macro e por classe).

    Referência:
        Hu, E.J. et al. (2022). Tabela 2 — Desempenho relativo de PEFT
        em benchmarks GLUE/SuperGLUE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Métricas por classe
    if label_names is not None:
        report = classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0,
        )
        metrics["per_class"] = {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in label_names
            if name in report
        }

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> str:
    """
    Gera relatório de classificação formatado.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        label_names: Nomes das classes.

    Returns:
        String com relatório formatado.
    """
    return classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0,
    )


# ---------------------------------------------------------------------------
# Benchmarking de latência
# ---------------------------------------------------------------------------


def benchmark_latency(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Mede latência de inferência do modelo.

    Implementa benchmarking conforme discutido na seção 'Estratégias de
    Warm-up e Caching de Modelos' do material da Aula 5. Warm-up
    inicial elimina overhead de compilação JIT e alocação de memória.

    A fórmula de speedup esperado (material, seção TorchScript):
        Speedup = T_eager / T_compiled

    Args:
        model: Modelo PyTorch em modo eval.
        input_ids: Input IDs de exemplo.
        attention_mask: Attention mask de exemplo.
        num_warmup: Rodadas de warm-up (descartadas).
        num_runs: Rodadas de medição.

    Returns:
        Dicionário com mean_ms, std_ms, p50_ms, p95_ms, p99_ms.

    Referência:
        Material da Aula 5, seção 'Estratégias de Warm-up e Caching'.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Warm-up: elimina cold start (compilação JIT, cache miss)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)

    # Medição
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies_np = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies_np)),
        "std_ms": float(np.std(latencies_np)),
        "p50_ms": float(np.percentile(latencies_np, 50)),
        "p95_ms": float(np.percentile(latencies_np, 95)),
        "p99_ms": float(np.percentile(latencies_np, 99)),
        "num_runs": num_runs,
    }


def benchmark_onnx_latency(
    onnx_path: str,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Mede latência de inferência com ONNX Runtime.

    Utiliza ONNX Runtime como Execution Provider, conforme discutido
    na seção 'ONNX: Interoperabilidade e Otimização Cross-Platform'
    do material da Aula 5 (Tabela 1 — Níveis de Otimização).

    Args:
        onnx_path: Caminho do modelo ONNX.
        input_ids: Input IDs numpy (int64).
        attention_mask: Attention mask numpy (int64).
        num_warmup: Rodadas de warm-up.
        num_runs: Rodadas de medição.

    Returns:
        Dicionário com mean_ms, std_ms, p50_ms, p95_ms.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    feed = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64),
    }

    # Warm-up
    for _ in range(num_warmup):
        session.run(None, feed)

    # Medição
    latencies: List[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, feed)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies_np = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies_np)),
        "std_ms": float(np.std(latencies_np)),
        "p50_ms": float(np.percentile(latencies_np, 50)),
        "p95_ms": float(np.percentile(latencies_np, 95)),
        "num_runs": num_runs,
    }


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    title: str = "Matriz de Confusão",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plota matriz de confusão com heatmap.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        label_names: Nomes das classes.
        title: Título do gráfico.
        save_path: Caminho para salvar figura.
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names or "auto",
        yticklabels=label_names or "auto",
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Matriz de confusão salva em %s", save_path)

    return fig


def plot_parameter_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Comparação de Parâmetros Treináveis — Full FT vs LoRA",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota comparação de parâmetros treináveis entre métodos.

    Visualiza dados equivalentes à Tabela 2 do material da Aula 5,
    comparando parâmetros treináveis de Full Fine-Tuning vs LoRA.

    Args:
        comparison_df: DataFrame com colunas [method, trainable_params].
        title: Título do gráfico.
        save_path: Caminho para salvar figura.

    Returns:
        Objeto Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de barras — parâmetros treináveis
    colors = ["#e74c3c", "#2ecc71"]
    axes[0].bar(
        comparison_df["method"],
        comparison_df["trainable_params"],
        color=colors[: len(comparison_df)],
    )
    axes[0].set_title("Parâmetros Treináveis", fontsize=12)
    axes[0].set_ylabel("Número de Parâmetros")
    axes[0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Gráfico de barras — percentual
    axes[1].bar(
        comparison_df["method"],
        comparison_df["trainable_pct"],
        color=colors[: len(comparison_df)],
    )
    axes[1].set_title("% Parâmetros Treináveis", fontsize=12)
    axes[1].set_ylabel("Percentual (%)")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latency_comparison(
    pytorch_latency: Dict[str, float],
    onnx_latency: Optional[Dict[str, float]] = None,
    title: str = "Comparação de Latência — PyTorch vs ONNX Runtime",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota comparação de latência entre PyTorch e ONNX Runtime.

    Visualiza o speedup obtido com otimização de grafo computacional
    conforme discutido na seção 'Otimização de Grafos Computacionais'
    do material: Speedup = T_eager / T_compiled.

    Args:
        pytorch_latency: Dict com 'mean_ms' do benchmark PyTorch.
        onnx_latency: Dict com 'mean_ms' do benchmark ONNX.
        title: Título do gráfico.
        save_path: Caminho para salvar figura.

    Returns:
        Objeto Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["PyTorch"]
    means = [pytorch_latency["mean_ms"]]
    stds = [pytorch_latency.get("std_ms", 0)]

    if onnx_latency is not None:
        methods.append("ONNX Runtime")
        means.append(onnx_latency["mean_ms"])
        stds.append(onnx_latency.get("std_ms", 0))

    colors = ["#3498db", "#e67e22"]
    bars = ax.bar(methods, means, yerr=stds, color=colors[: len(methods)], capsize=5)

    # Anotar valores
    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )

    if onnx_latency is not None and onnx_latency["mean_ms"] > 0:
        speedup = pytorch_latency["mean_ms"] / onnx_latency["mean_ms"]
        ax.set_title(f"{title}\n(Speedup ONNX: {speedup:.2f}x)", fontsize=13)
    else:
        ax.set_title(title, fontsize=13)

    ax.set_ylabel("Latência (ms)", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "Curvas de Treinamento — Fine-Tuning com LoRA",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota curvas de loss e accuracy durante treinamento.

    Args:
        train_losses: Lista de perdas de treinamento por step/época.
        val_losses: Lista de perdas de validação (opcional).
        val_accuracies: Lista de accuracies de validação (opcional).
        title: Título do gráfico.
        save_path: Caminho para salvar figura.

    Returns:
        Objeto Figure.
    """
    n_axes = 1 + (1 if val_accuracies else 0)
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    # Loss
    axes[0].plot(train_losses, label="Train Loss", color="#3498db")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", color="#e74c3c", linestyle="--")
    axes[0].set_xlabel("Step / Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    if val_accuracies and n_axes > 1:
        axes[1].plot(val_accuracies, label="Val Accuracy", color="#2ecc71", marker="o")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_peft_comparison_table() -> pd.DataFrame:
    """
    Cria tabela comparativa de técnicas PEFT.

    Reproduz a Tabela 2 do material da Aula 5 — 'Comparação de Técnicas
    PEFT para LLaMA-7B' (Hu et al., 2022; Dettmers et al., 2023).

    Returns:
        DataFrame com colunas: technique, trainable_params, trainable_pct,
        gpu_memory, latency_overhead, relative_performance.
    """
    data = [
        {
            "technique": "Full Fine-Tuning",
            "trainable_params": "7B (100%)",
            "gpu_memory_gb": 28,
            "latency_overhead": "0%",
            "relative_performance": "100%",
        },
        {
            "technique": "LoRA (r=16)",
            "trainable_params": "4.2M (0.06%)",
            "gpu_memory_gb": 8,
            "latency_overhead": "0%",
            "relative_performance": "97-99%",
        },
        {
            "technique": "QLoRA (4-bit, r=16)",
            "trainable_params": "4.2M (0.06%)",
            "gpu_memory_gb": 6,
            "latency_overhead": "0%",
            "relative_performance": "96-98%",
        },
        {
            "technique": "Adapter",
            "trainable_params": "3.5M (0.05%)",
            "gpu_memory_gb": 9,
            "latency_overhead": "5-10%",
            "relative_performance": "95-97%",
        },
        {
            "technique": "Prefix Tuning",
            "trainable_params": "0.5M (0.007%)",
            "gpu_memory_gb": 7,
            "latency_overhead": "3-5%",
            "relative_performance": "90-95%",
        },
        {
            "technique": "Prompt Tuning",
            "trainable_params": "0.1M (0.001%)",
            "gpu_memory_gb": 7,
            "latency_overhead": "1-2%",
            "relative_performance": "85-93%",
        },
    ]
    return pd.DataFrame(data)
