"""
Testes unitários para o módulo evaluation.

Verifica métricas de acurácia, latência e tamanho de modelo.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    calculate_accuracy,
    calculate_per_class_accuracy,
    get_model_size_mb,
    measure_latency,
)
from src.model import SimpleCNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model() -> SimpleCNN:
    torch.manual_seed(42)
    return SimpleCNN(num_classes=10)


@pytest.fixture
def dummy_loader() -> DataLoader:
    """DataLoader com dados sintéticos (100 amostras, 10 classes)."""
    torch.manual_seed(42)
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    return DataLoader(TensorDataset(X, y), batch_size=32)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestAccuracy:
    """Testes de cálculo de acurácia."""

    def test_accuracy_range(self, simple_model: SimpleCNN, dummy_loader: DataLoader) -> None:
        acc = calculate_accuracy(simple_model, dummy_loader, device="cpu")
        assert 0.0 <= acc <= 1.0

    def test_per_class_accuracy_keys(
        self, simple_model: SimpleCNN, dummy_loader: DataLoader
    ) -> None:
        per_class = calculate_per_class_accuracy(
            simple_model, dummy_loader, num_classes=10, device="cpu"
        )
        assert len(per_class) == 10
        for cls in range(10):
            assert cls in per_class

    def test_per_class_accuracy_range(
        self, simple_model: SimpleCNN, dummy_loader: DataLoader
    ) -> None:
        per_class = calculate_per_class_accuracy(
            simple_model, dummy_loader, num_classes=10, device="cpu"
        )
        for acc in per_class.values():
            assert 0.0 <= acc <= 1.0


class TestLatency:
    """Testes de medição de latência."""

    def test_latency_positive(self, simple_model: SimpleCNN) -> None:
        lat = measure_latency(simple_model, n_warmup=2, n_runs=5)
        assert lat["mean_ms"] > 0
        assert lat["p50_ms"] > 0

    def test_latency_keys(self, simple_model: SimpleCNN) -> None:
        lat = measure_latency(simple_model, n_warmup=2, n_runs=5)
        for key in ["mean_ms", "std_ms", "p50_ms", "p99_ms"]:
            assert key in lat


class TestModelSize:
    """Testes de tamanho de modelo."""

    def test_size_positive(self, simple_model: SimpleCNN) -> None:
        size = get_model_size_mb(simple_model)
        assert size > 0

    def test_empty_model_small(self) -> None:
        model = nn.Linear(1, 1)
        size = get_model_size_mb(model)
        assert size < 0.01  # < 10 KB
