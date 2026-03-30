"""
Módulo de avaliação e métricas de compressão.

Funções para medir acurácia, latência de inferência, consumo de memória
e gerar gráficos comparativos entre modelos denso, podado e quantizado.

Os trade-offs avaliados seguem a Tabela 2 do documento 04 (Jacob et al.,
2018; Blalock et al., 2020).

Referências:
    Jacob, B. et al. (2018). Quantization and Training of Neural Networks
        for Efficient Integer-Arithmetic-Only Inference. CVPR.
    Blalock, D. et al. (2020). What is the State of Neural Network
        Pruning? MLSys.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# 1. Métricas de classificação
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calculate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    """Calcula acurácia top-1 no DataLoader fornecido.

    Args:
        model: Modelo treinado (FP32 ou INT8).
        loader: DataLoader de avaliação.
        device: Dispositivo.

    Returns:
        Acurácia como fração (0-1).
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total


@torch.no_grad()
def calculate_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 10,
    device: torch.device | str = "cpu",
) -> dict[int, float]:
    """Calcula acurácia por classe.

    Args:
        model: Modelo treinado.
        loader: DataLoader de avaliação.
        num_classes: Quantidade de classes.
        device: Dispositivo.

    Returns:
        Dicionário ``{class_idx: accuracy}``.
    """
    model.eval()
    class_correct: dict[int, int] = {i: 0 for i in range(num_classes)}
    class_total: dict[int, int] = {i: 0 for i in range(num_classes)}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        for cls in range(num_classes):
            mask = labels == cls
            class_total[cls] += mask.sum().item()
            class_correct[cls] += (predicted[mask] == cls).sum().item()

    return {
        cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0)
        for cls in range(num_classes)
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Métricas de eficiência
# ═══════════════════════════════════════════════════════════════════════════

def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    device: torch.device | str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Mede latência de inferência em milissegundos.

    Conforme discutido no documento 04, a latência é uma das três
    métricas fundamentais de trade-off (junto com memória e acurácia).

    Args:
        model: Modelo a medir.
        input_shape: Shape do tensor de entrada.
        device: Dispositivo.
        n_warmup: Iterações de aquecimento (ignoradas).
        n_runs: Iterações de medição.

    Returns:
        Dicionário com ``mean_ms``, ``std_ms``, ``p50_ms``, ``p99_ms``.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(n_warmup):
        model(dummy)

    # Medição
    latencies: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(dummy)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Retorna tamanho do modelo em MB (pesos + buffers).

    Conforme a Tabela 2 do documento 04, modelos INT8 ocupam ~4× menos
    memória que seus equivalentes FP32.
    """
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)


def calculate_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Calcula métricas completas: acurácia, latência e tamanho.

    Implementa a avaliação tríplice discutida na seção *Trade-offs* do
    documento 04: acurácia, latência e memória.

    Args:
        model: Modelo treinado.
        test_loader: DataLoader de teste.
        device: Dispositivo.

    Returns:
        Dicionário com ``accuracy``, ``latency_mean_ms``,
        ``latency_p99_ms``, ``size_mb``.
    """
    acc = calculate_accuracy(model, test_loader, device)
    latency = measure_latency(model, device=device)
    size = get_model_size_mb(model)

    return {
        "accuracy": acc,
        "latency_mean_ms": latency["mean_ms"],
        "latency_p99_ms": latency["p99_ms"],
        "size_mb": size,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Visualizações
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """Plota curvas de loss e acurácia de treinamento/validação.

    Args:
        history: Dicionário retornado por ``train_model()``.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura (opcional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_compression_comparison(
    results: dict[str, dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Gráfico de barras comparando modelos (acurácia, latência, tamanho).

    Reproduz a análise de trade-offs discutida na seção *Trade-offs:
    Acurácia, Latência e Memória* do documento 04.

    Args:
        results: ``{model_name: {accuracy, latency_mean_ms, size_mb}}``.
        save_path: Caminho para salvar a figura.
    """
    df = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("accuracy", "Acurácia", "Blues_d"),
        ("latency_mean_ms", "Latência (ms)", "Oranges_d"),
        ("size_mb", "Tamanho (MB)", "Greens_d"),
    ]

    for ax, (col, ylabel, palette) in zip(axes, metrics):
        sns.barplot(x=df.index, y=df[col], ax=ax, palette=palette)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.suptitle(
        "Comparação de Compressão — Trade-offs (Doc 04, Tabela 2)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sparsity_distribution(
    sparsity: dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """Plota distribuição de esparsidade por camada.

    Conforme Frankle & Carlin (2019), camadas diferentes possuem
    sensibilidades distintas ao pruning.

    Args:
        sparsity: Dicionário retornado por ``compute_sparsity()``.
        save_path: Caminho para salvar a figura.
    """
    layers = [k for k in sparsity if k != "global"]
    values = [sparsity[k] for k in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(layers, values, color="steelblue")
    ax.axvline(sparsity.get("global", 0), color="red", linestyle="--", label="Global")
    ax.set_xlabel("Esparsidade (%)")
    ax.set_title("Esparsidade por Camada (Lottery Ticket Analysis)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weight_distribution(
    model: nn.Module,
    title: str = "Weight Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Histogramas da distribuição de pesos por camada.

    Permite visualizar o efeito do pruning (pico em zero) e da
    quantização (discretização) nas distribuições de pesos.

    Args:
        model: Modelo a analisar.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.
    """
    weights: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data.cpu().numpy().flatten()
            weights[name] = w

    n = len(weights)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (name, w) in zip(axes_flat, weights.items()):
        ax.hist(w, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Peso")
        ax.set_ylabel("Frequência")
        non_zero_pct = 100.0 * np.count_nonzero(w) / w.size
        ax.text(
            0.95, 0.95, f"non-zero: {non_zero_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
        )

    # Esconde eixos extras
    for ax in axes_flat[len(weights):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
