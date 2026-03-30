"""
Módulo de treinamento e benchmarking de arquiteturas de visão.

Implementa lógica de treinamento para modelos simples de classificação
de imagens e funções de benchmarking comparativo entre arquiteturas,
seguindo a metodologia de avaliação discutida na seção 10 ('Fronteira
de Pareto e Seleção de Arquiteturas').

Referência:
    Tan, M. & Le, Q. V. (2019). EfficientNet. ICML 2019.
    Cai, H. et al. (2020). Once-for-All: Train One Network and
    Specialize it for Efficient Deployment. ICLR 2020.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# =========================================================================
# Treinamento de modelos
# =========================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Treina modelo de classificação de imagens com monitoramento de métricas.

    Implementa loop de treinamento padrão com loss cross-entropy e
    otimizador Adam. Registra métricas por época para análise
    posterior do trade-off entre convergência e custo computacional.

    Args:
        model: Modelo PyTorch (nn.Module).
        train_loader: DataLoader de treinamento.
        val_loader: DataLoader de validação (opcional).
        n_epochs: Número de épocas.
        lr: Learning rate.
        device: Dispositivo de execução.
        verbose: Se True, exibe progresso.

    Returns:
        Dicionário com histórico de treinamento:
        - 'train_loss': Lista de loss por época
        - 'train_acc': Lista de acurácia por época
        - 'val_loss', 'val_acc': Métricas de validação (se val_loader)
        - 'epoch_times': Tempo por época em segundos
        - 'total_time': Tempo total de treinamento
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    total_start = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()

        # --- Treinamento ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # --- Validação ---
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_times"].append(epoch_time)

        if verbose:
            msg = f"Época {epoch + 1}/{n_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}"
            if val_loader is not None:
                msg += f" | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}"
            msg += f" | Tempo: {epoch_time:.2f}s"
            logger.info(msg)
            print(msg)

    history["total_time"] = time.perf_counter() - total_start
    return history


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Avalia modelo em um DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# =========================================================================
# Preparação de DataLoaders
# =========================================================================


def create_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders de treino e validação a partir de arrays numpy.

    Args:
        images: Array de imagens (N, C, H, W), float32.
        labels: Array de rótulos (N,), int64.
        batch_size: Tamanho do batch.
        val_split: Fração de validação.
        random_state: Seed para split reprodutível.

    Returns:
        Tupla (train_loader, val_loader).
    """
    n = len(images)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    split = int(n * (1 - val_split))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = TensorDataset(
        torch.from_numpy(images[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(images[val_idx]),
        torch.from_numpy(labels[val_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# =========================================================================
# Benchmarking comparativo de arquiteturas
# =========================================================================


def benchmark_architectures(
    model_names: Optional[list[str]] = None,
    input_sizes: Optional[list[int]] = None,
    n_runs: int = 30,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """
    Executa benchmark comparativo de múltiplas arquiteturas.

    Implementa a metodologia de benchmarking sistemático para
    caracterização da fronteira de Pareto, conforme discutido na
    seção 10 do material (Tan & Le, 2019; Cai et al., 2020).

    Args:
        model_names: Lista de nomes de modelos para benchmark.
            Default: ['simple_cnn', 'mobilenet_style'].
        input_sizes: Lista de resoluções a testar.
            Default: [128, 224].
        n_runs: Execuções por configuração.
        device: Dispositivo de execução.

    Returns:
        Lista de dicionários com resultados de benchmark.
    """
    from src.model import VisionBenchmarkModel

    if model_names is None:
        model_names = ["simple_cnn", "mobilenet_style"]
    if input_sizes is None:
        input_sizes = [128, 224]

    all_results: list[dict[str, Any]] = []

    for name in model_names:
        logger.info("Benchmarking: %s", name)
        print(f"\n--- Benchmarking: {name} ---")

        try:
            bm = VisionBenchmarkModel(
                model_name=name,
                pretrained=False,
                device=device,
            )
            params = bm.count_parameters()

            for res in input_sizes:
                try:
                    stats = bm.benchmark_latency(
                        input_size=res,
                        n_warmup=3,
                        n_runs=n_runs,
                    )
                    flops_info = bm.estimate_flops(input_size=res)

                    result = {
                        "model_name": name,
                        "input_size": res,
                        "params_millions": params["total_millions"],
                        "flops_gflops": flops_info["total_gflops"],
                        "latency_mean_ms": stats["mean_ms"],
                        "latency_p50_ms": stats["p50_ms"],
                        "latency_p95_ms": stats["p95_ms"],
                        "throughput_imgs_sec": stats["throughput_imgs_sec"],
                        "device": device,
                    }
                    all_results.append(result)
                    print(
                        f"  {name} @ {res}: "
                        f"{stats['mean_ms']:.2f} ms | "
                        f"{flops_info['total_gflops']:.4f} GFLOPs | "
                        f"{params['total_millions']}M params"
                    )
                except RuntimeError as e:
                    logger.warning("Falha %s @ %d: %s", name, res, e)
        except ValueError as e:
            logger.warning("Modelo %s não disponível: %s", name, e)

    return all_results


def cross_validate_training(
    model_class: type,
    images: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 3,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """
    Validação cruzada para treinamento de modelos de visão.

    Args:
        model_class: Classe do modelo PyTorch.
        images: Array de imagens (N, C, H, W).
        labels: Array de rótulos (N,).
        n_folds: Número de folds.
        n_epochs: Épocas por fold.
        batch_size: Tamanho do batch.
        lr: Learning rate.
        device: Dispositivo.
        **model_kwargs: Argumentos para model_class.

    Returns:
        Dicionário com métricas médias por fold.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        model = model_class(**model_kwargs)

        train_ds = TensorDataset(
            torch.from_numpy(images[train_idx]),
            torch.from_numpy(labels[train_idx]),
        )
        val_ds = TensorDataset(
            torch.from_numpy(images[val_idx]),
            torch.from_numpy(labels[val_idx]),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        history = train_model(
            model, train_loader, val_loader,
            n_epochs=n_epochs, lr=lr, device=device, verbose=False,
        )

        fold_results.append({
            "fold": fold + 1,
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "total_time": history["total_time"],
        })
        print(
            f"    Val Acc: {fold_results[-1]['final_val_acc']:.4f} | "
            f"Tempo: {fold_results[-1]['total_time']:.2f}s"
        )

    # Média dos folds
    mean_acc = np.mean([r["final_val_acc"] for r in fold_results if r["final_val_acc"]])
    std_acc = np.std([r["final_val_acc"] for r in fold_results if r["final_val_acc"]])

    return {
        "folds": fold_results,
        "mean_val_acc": float(mean_acc),
        "std_val_acc": float(std_acc),
        "n_folds": n_folds,
    }
