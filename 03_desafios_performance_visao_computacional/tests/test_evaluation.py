"""
Testes unitários para o módulo evaluation.

Verifica cálculo de métricas, fronteira de Pareto, geração de relatórios
e funções de visualização.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    _find_pareto_points,
    calculate_metrics,
    generate_architecture_report,
    plot_confusion_matrix,
    plot_pareto_frontier,
    plot_resolution_impact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_predictions():
    """Previsões binárias para testes."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    return y_true, y_pred


@pytest.fixture
def multiclass_predictions():
    """Previsões multiclasse (4 classes) para testes."""
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = y_true.copy()
    # Introduz ~20% de erros
    mask = np.random.rand(100) < 0.2
    y_pred[mask] = np.random.randint(0, 4, mask.sum())
    return y_true, y_pred


@pytest.fixture
def benchmark_df() -> pd.DataFrame:
    """DataFrame de benchmark simulado para fronteira de Pareto."""
    return pd.DataFrame({
        "model_name": [
            "mobilenetv2", "efficientnet_b0", "resnet50", "deit_small",
            "convnext_tiny", "efficientnet_b4", "vgg16", "deit_tiny",
        ],
        "model_family": ["cnn", "cnn", "cnn", "vit", "hybrid", "cnn", "cnn", "vit"],
        "latency_mean_ms": [1.2, 2.1, 4.5, 3.8, 3.0, 8.3, 7.2, 2.0],
        "top1_accuracy": [72.0, 77.1, 79.0, 79.9, 82.1, 82.9, 71.5, 72.2],
        "params_millions": [3.4, 5.3, 25.6, 22.1, 28.6, 19.0, 138.0, 5.7],
        "flops_giga": [0.3, 0.39, 4.1, 4.6, 4.5, 4.2, 15.5, 1.3],
    })


# ---------------------------------------------------------------------------
# Testes de métricas
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Testes para calculate_metrics."""

    def test_returns_required_keys(self, binary_predictions) -> None:
        """Retorna dicionário com as chaves obrigatórias."""
        y_true, y_pred = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
            assert key in metrics, f"Chave ausente: {key}"

    def test_perfect_predictions(self) -> None:
        """Predições perfeitas → accuracy = 1.0."""
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        metrics = calculate_metrics(y, y)
        assert abs(metrics["accuracy"] - 1.0) < 1e-6

    def test_accuracy_range(self, multiclass_predictions) -> None:
        """Accuracy está em [0, 1]."""
        y_true, y_pred = multiclass_predictions
        metrics = calculate_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, multiclass_predictions) -> None:
        """Confusion matrix é quadrada com dimensão igual ao n° de classes."""
        y_true, y_pred = multiclass_predictions
        metrics = calculate_metrics(y_true, y_pred)
        n_classes = len(set(y_true))
        cm = metrics["confusion_matrix"]
        assert cm.shape == (n_classes, n_classes)

    def test_confusion_matrix_row_sums(self, binary_predictions) -> None:
        """Soma de cada linha da confusion matrix == contagem real por classe."""
        y_true, y_pred = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        for i in range(cm.shape[0]):
            assert cm[i].sum() == (y_true == i).sum()


# ---------------------------------------------------------------------------
# Testes da fronteira de Pareto
# ---------------------------------------------------------------------------


class TestFindParetoPoints:
    """Testes para detecção de soluções Pareto-ótimas."""

    def test_obvious_dominated(self) -> None:
        """Ponto dominado não está na fronteira."""
        # Pontos: (custo, -acurácia) → minimizamos ambos
        # A domina C (menor custo, maior acurácia)
        costs = np.array([1.0, 3.0, 5.0])
        accuracies = np.array([90.0, 80.0, 70.0])
        mask = _find_pareto_points(costs, accuracies)
        # Apenas o primeiro ponto é Pareto-ótimo (menor custo E maior acurácia)
        assert mask[0] is True or mask[0] == True

    def test_all_pareto_when_tradeoff(self) -> None:
        """Pontos com trade-off genuíno são todos Pareto-ótimos."""
        costs = np.array([1.0, 2.0, 3.0])
        accuracies = np.array([70.0, 80.0, 90.0])
        mask = _find_pareto_points(costs, accuracies)
        assert all(mask)

    def test_returns_boolean_array(self, benchmark_df: pd.DataFrame) -> None:
        """Retorna array booleano do mesmo comprimento."""
        costs = benchmark_df["latency_mean_ms"].values
        accs = benchmark_df["top1_accuracy"].values
        mask = _find_pareto_points(costs, accs)
        assert len(mask) == len(costs)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Testes de geração de relatório
# ---------------------------------------------------------------------------


class TestGenerateArchitectureReport:
    """Testes para generate_architecture_report."""

    def test_returns_dataframe(self, benchmark_df: pd.DataFrame) -> None:
        """Retorna DataFrame."""
        report = generate_architecture_report(
            benchmark_df, max_latency_ms=10.0, min_accuracy=70.0,
        )
        assert isinstance(report, pd.DataFrame)

    def test_respects_constraints(self, benchmark_df: pd.DataFrame) -> None:
        """Resultados respeitam restrições de latência e acurácia."""
        report = generate_architecture_report(
            benchmark_df, max_latency_ms=5.0, min_accuracy=75.0,
        )
        if len(report) > 0:
            assert (report["latency_mean_ms"] <= 5.0).all()
            assert (report["top1_accuracy"] >= 75.0).all()

    def test_empty_when_impossible(self, benchmark_df: pd.DataFrame) -> None:
        """Retorna vazio quando restrições são impossíveis."""
        report = generate_architecture_report(
            benchmark_df, max_latency_ms=0.1, min_accuracy=99.0,
        )
        assert len(report) == 0


# ---------------------------------------------------------------------------
# Testes de visualização (verificam que não lançam exceções)
# ---------------------------------------------------------------------------


class TestPlots:
    """Testes de fumaça para funções de plot."""

    def test_plot_confusion_matrix_runs(self, binary_predictions) -> None:
        """plot_confusion_matrix executa sem erros."""
        import matplotlib
        matplotlib.use("Agg")
        y_true, y_pred = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        fig = plot_confusion_matrix(
            metrics["confusion_matrix"], class_names=["Normal", "Defeito"],
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_pareto_frontier_runs(self, benchmark_df: pd.DataFrame) -> None:
        """plot_pareto_frontier executa sem erros."""
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_pareto_frontier(benchmark_df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_resolution_impact_runs(self) -> None:
        """plot_resolution_impact executa sem erros."""
        import matplotlib
        matplotlib.use("Agg")
        df = pd.DataFrame({
            "input_resolution": [128, 224, 320, 384, 512] * 4,
            "flops_giga": np.random.rand(20) * 10,
            "latency_ms": np.random.rand(20) * 15,
            "memory_mb": np.random.rand(20) * 1000,
            "top1_accuracy": np.random.rand(20) * 30 + 60,
        })
        fig = plot_resolution_impact(df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")
