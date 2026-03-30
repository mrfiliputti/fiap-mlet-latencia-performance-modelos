"""
Testes unitários para o módulo model.

Verifica a inicialização, treinamento, predição e benchmarking de latência
dos modelos de classificação.
"""

import numpy as np
import pytest

from src.model import LatencyBenchmarkModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Gera dados de exemplo para testes."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    return X, y


@pytest.fixture
def trained_model(sample_data: tuple[np.ndarray, np.ndarray]) -> LatencyBenchmarkModel:
    """Retorna modelo treinado com Logistic Regression."""
    X, y = sample_data
    model = LatencyBenchmarkModel(model_name="logistic_regression", random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestModelInit:
    """Testes de inicialização do modelo."""

    def test_supported_models(self) -> None:
        """Todos os modelos suportados podem ser inicializados."""
        for name in LatencyBenchmarkModel.SUPPORTED_MODELS:
            model = LatencyBenchmarkModel(model_name=name, random_state=42)
            assert model.model_name == name

    def test_unsupported_model_raises(self) -> None:
        """Modelo não suportado levanta ValueError."""
        with pytest.raises(ValueError, match="não suportado"):
            LatencyBenchmarkModel(model_name="modelo_inexistente")

    def test_initial_state(self) -> None:
        """Estado inicial: não treinado, sem histórico."""
        model = LatencyBenchmarkModel(model_name="logistic_regression")
        assert model.is_fitted is False
        assert len(model.latency_history) == 0


class TestModelFit:
    """Testes de treinamento."""

    def test_fit_marks_as_fitted(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fit() marca modelo como treinado."""
        X, y = sample_data
        model = LatencyBenchmarkModel(model_name="logistic_regression", random_state=42)
        model.fit(X, y)
        assert model.is_fitted is True

    def test_fit_logs_latency(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fit() registra latência."""
        X, y = sample_data
        model = LatencyBenchmarkModel(model_name="logistic_regression", random_state=42)
        model.fit(X, y)
        assert len(model.latency_history) == 1
        assert model.latency_history[0]["operation"] == "fit"

    def test_fit_returns_self(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fit() retorna self para encadeamento."""
        X, y = sample_data
        model = LatencyBenchmarkModel(model_name="logistic_regression", random_state=42)
        result = model.fit(X, y)
        assert result is model


class TestModelPredict:
    """Testes de predição."""

    def test_predict_returns_array(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict() retorna array de predições."""
        X, _ = sample_data
        y_pred = trained_model.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (X.shape[0],)

    def test_predict_before_fit_raises(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict() sem fit() levanta RuntimeError."""
        X, _ = sample_data
        model = LatencyBenchmarkModel(model_name="logistic_regression")
        with pytest.raises(RuntimeError, match="não treinado"):
            model.predict(X)

    def test_predict_single_returns_tuple(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict_single() retorna tupla (predição, latência_ms)."""
        X, _ = sample_data
        prediction, latency_ms = trained_model.predict_single(X[0])
        assert isinstance(prediction, int)
        assert isinstance(latency_ms, float)
        assert latency_ms > 0

    def test_predict_values_are_binary(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Predições são 0 ou 1 para classificação binária."""
        X, _ = sample_data
        y_pred = trained_model.predict(X)
        assert set(np.unique(y_pred)).issubset({0, 1})


class TestModelScore:
    """Testes de score."""

    def test_score_returns_float(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """score() retorna float entre 0 e 1."""
        X, y = sample_data
        score = trained_model.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestBenchmarkInference:
    """Testes de benchmarking de latência."""

    def test_benchmark_returns_expected_keys(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """benchmark_inference() retorna todas as métricas esperadas."""
        X, _ = sample_data
        result = trained_model.benchmark_inference(X, n_iterations=10)
        expected_keys = {
            "model", "n_samples", "n_iterations",
            "mean_ms", "std_ms", "min_ms", "max_ms",
            "p50_ms", "p95_ms", "p99_ms", "throughput_samples_per_sec",
        }
        assert set(result.keys()) == expected_keys

    def test_benchmark_percentiles_ordered(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Percentis estão em ordem: P50 ≤ P95 ≤ P99."""
        X, _ = sample_data
        result = trained_model.benchmark_inference(X, n_iterations=20)
        assert result["p50_ms"] <= result["p95_ms"]
        assert result["p95_ms"] <= result["p99_ms"]

    def test_benchmark_positive_throughput(
        self, trained_model: LatencyBenchmarkModel, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Throughput é positivo."""
        X, _ = sample_data
        result = trained_model.benchmark_inference(X, n_iterations=10)
        assert result["throughput_samples_per_sec"] > 0
