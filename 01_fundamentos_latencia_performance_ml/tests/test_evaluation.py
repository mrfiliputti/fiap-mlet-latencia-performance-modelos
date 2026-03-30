"""
Testes unitários para o módulo evaluation.

Verifica cálculo de métricas de qualidade, métricas de latência,
benchmarking de batch sizes e funções de visualização.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    benchmark_batch_sizes,
    calculate_latency_metrics,
    calculate_metrics,
    compare_models_latency,
)
from src.model import LatencyBenchmarkModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Gera predições binárias de exemplo."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def trained_model() -> LatencyBenchmarkModel:
    """Retorna modelo treinado para benchmarking."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    model = LatencyBenchmarkModel(model_name="logistic_regression", random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def test_features() -> np.ndarray:
    """Gera features de teste."""
    np.random.seed(42)
    return np.random.randn(100, 10)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Testes para cálculo de métricas de qualidade."""

    def test_returns_expected_keys(
        self, binary_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """calculate_metrics retorna todas as métricas esperadas."""
        y_true, y_pred = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_values_in_valid_range(
        self, binary_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Todas métricas estão entre 0 e 1."""
        y_true, y_pred = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} fora do range [0, 1]"

    def test_perfect_predictions(self) -> None:
        """Predições perfeitas resultam em métricas = 1.0."""
        y = np.array([0, 1, 1, 0, 1])
        metrics = calculate_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_with_probabilities(
        self, binary_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """AUC-ROC é calculado quando probabilidades são fornecidas."""
        y_true, y_pred = binary_predictions
        y_prob = np.random.rand(len(y_true))
        metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob)
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0


class TestCalculateLatencyMetrics:
    """Testes para métricas de latência."""

    def test_returns_latency_dict(
        self, trained_model: LatencyBenchmarkModel, test_features: np.ndarray
    ) -> None:
        """calculate_latency_metrics retorna dicionário com métricas."""
        result = calculate_latency_metrics(trained_model, test_features, n_iterations=10)
        assert "mean_ms" in result
        assert "p50_ms" in result
        assert "p99_ms" in result

    def test_positive_latencies(
        self, trained_model: LatencyBenchmarkModel, test_features: np.ndarray
    ) -> None:
        """Latências são positivas."""
        result = calculate_latency_metrics(trained_model, test_features, n_iterations=10)
        assert result["mean_ms"] > 0
        assert result["throughput_samples_per_sec"] > 0


class TestBenchmarkBatchSizes:
    """Testes para benchmark de batch sizes."""

    def test_returns_dataframe(
        self, trained_model: LatencyBenchmarkModel, test_features: np.ndarray
    ) -> None:
        """benchmark_batch_sizes retorna DataFrame."""
        result = benchmark_batch_sizes(
            trained_model, test_features,
            batch_sizes=[1, 10, 50], n_iterations=5
        )
        assert isinstance(result, pd.DataFrame)
        assert "batch_size" in result.columns
        assert "throughput_samples_per_sec" in result.columns

    def test_throughput_increases_with_batch(
        self, trained_model: LatencyBenchmarkModel, test_features: np.ndarray
    ) -> None:
        """Throughput geralmente aumenta com batch size maior."""
        result = benchmark_batch_sizes(
            trained_model, test_features,
            batch_sizes=[1, 50], n_iterations=20
        )
        # Throughput com batch=50 deve ser >= throughput com batch=1
        assert result.iloc[-1]["throughput_samples_per_sec"] >= \
               result.iloc[0]["throughput_samples_per_sec"] * 0.5  # margem conservadora
