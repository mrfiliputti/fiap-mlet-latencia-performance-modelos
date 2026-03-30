"""
Testes unitários para o módulo de avaliação e profiling.

Valida implementação do Modelo Roofline (Williams et al., 2009),
funções de profiling, e métricas de classificação para experimentos
de aceleração por hardware (Aula 06).

Referências testadas:
    Williams, S. et al. (2009). Roofline: An Insightful Visual Performance Model.
    Dao, T. et al. (2022). FlashAttention.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    RooflineModel,
    calculate_metrics,
    profile_model,
    evaluate_model,
)
from src.model import create_model


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def roofline_a100():
    """Cria modelo roofline para A100."""
    return RooflineModel.from_hardware("A100_80GB")


@pytest.fixture
def small_model():
    """Modelo pequeno para testes de profiling."""
    return create_model(
        img_size=8, patch_size=2, embed_dim=32, depth=2,
        n_heads=2, n_classes=5,
    )


# -----------------------------------------------------------------------
# Testes do Modelo Roofline — Williams, Waterman & Patterson (2009)
# -----------------------------------------------------------------------

class TestRooflineModel:
    """Testes do Modelo Roofline."""

    def test_from_hardware_valid(self):
        """Verifica criação a partir de perfil de hardware válido."""
        roofline = RooflineModel.from_hardware("A100_80GB")
        assert roofline.peak_compute_tflops == 312.0
        assert roofline.memory_bandwidth_gb_s == 2039.0

    def test_from_hardware_invalid(self):
        """Verifica erro para hardware inválido."""
        with pytest.raises(ValueError, match="não encontrado"):
            RooflineModel.from_hardware("GPU_INEXISTENTE")

    def test_ridge_point_positive(self, roofline_a100):
        """Verifica ridge point positivo."""
        assert roofline_a100.ridge_point > 0

    def test_attainable_performance_memory_bound(self, roofline_a100):
        """Verifica que operações de baixa AI são memory-bound."""
        # Softmax: ~1 FLOP/byte — abaixo do ridge point
        perf_low_ai = roofline_a100.attainable_performance(1.0)
        assert perf_low_ai < roofline_a100.peak_compute_tflops

    def test_attainable_performance_compute_bound(self, roofline_a100):
        """Verifica que operações de alta AI atingem pico computacional."""
        # Matmul: ~1000 FLOP/byte — bem acima do ridge point
        perf_high_ai = roofline_a100.attainable_performance(1000.0)
        assert perf_high_ai == pytest.approx(roofline_a100.peak_compute_tflops)

    def test_classify_operation_matmul(self, roofline_a100):
        """Verifica classificação de matmul como compute-bound."""
        cls = roofline_a100.classify_operation(100.0)
        assert cls == "compute-bound"

    def test_classify_operation_softmax(self, roofline_a100):
        """Verifica classificação de softmax como memory-bound."""
        cls = roofline_a100.classify_operation(1.0)
        assert cls == "memory-bound"

    def test_analyze_operations_returns_all(self, roofline_a100):
        """Verifica que analyze_operations retorna todas as operações típicas."""
        results = roofline_a100.analyze_operations()
        assert len(results) == len(RooflineModel.TYPICAL_AI)
        for r in results:
            assert "operation" in r
            assert "classification" in r
            assert r["classification"] in ("compute-bound", "memory-bound", "balanced")

    def test_get_roofline_data_shape(self, roofline_a100):
        """Verifica formato dos dados de roofline para plotting."""
        data = roofline_a100.get_roofline_data(n_points=100)
        assert len(data["arithmetic_intensity"]) == 100
        assert len(data["attainable_performance"]) == 100
        assert data["ridge_point"] > 0
        assert data["peak_compute"] == 312.0


# -----------------------------------------------------------------------
# Testes de Métricas de Classificação
# -----------------------------------------------------------------------

class TestCalculateMetrics:
    """Testes de cálculo de métricas."""

    def test_perfect_accuracy(self):
        """Verifica acurácia perfeita (100%) com predições corretas."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        metrics = calculate_metrics(y_true, y_pred, n_classes=5)
        assert metrics["accuracy"] == 1.0
        assert metrics["n_correct"] == 5

    def test_zero_accuracy(self):
        """Verifica acurácia zero com predições erradas."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4, 0])
        metrics = calculate_metrics(y_true, y_pred, n_classes=5)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        """Verifica shape da matriz de confusão."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 2])
        metrics = calculate_metrics(y_true, y_pred, n_classes=3)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Verifica diagonal da matriz de confusão com acurácia perfeita."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        metrics = calculate_metrics(y_true, y_pred, n_classes=3)
        cm = np.array(metrics["confusion_matrix"])
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert cm.sum() - np.trace(cm) == 0  # Nenhum erro

    def test_per_class_accuracy(self):
        """Verifica acurácia por classe."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = calculate_metrics(y_true, y_pred, n_classes=2)
        assert metrics["per_class_accuracy"]["class_0_accuracy"] == 0.5
        assert metrics["per_class_accuracy"]["class_1_accuracy"] == 1.0


# -----------------------------------------------------------------------
# Testes de Profiling
# -----------------------------------------------------------------------

class TestProfiling:
    """Testes de profiling do modelo."""

    def test_profile_model_returns_table(self, small_model):
        """Verifica que profiling retorna tabela de resultados."""
        results = profile_model(
            small_model, input_shape=(2, 3, 8, 8),
            device=torch.device("cpu"), n_iterations=2, warmup=1,
        )
        assert "table" in results
        assert "events" in results
        assert isinstance(results["table"], str)
        assert len(results["events"]) > 0

    def test_profile_model_events_format(self, small_model):
        """Verifica formato dos eventos de profiling."""
        results = profile_model(
            small_model, input_shape=(2, 3, 8, 8),
            device=torch.device("cpu"), n_iterations=2, warmup=1,
        )
        for evt in results["events"]:
            assert "name" in evt
            assert "cpu_time_ms" in evt
            assert "count" in evt


# -----------------------------------------------------------------------
# Testes de Evaluate Model
# -----------------------------------------------------------------------

class TestEvaluateModel:
    """Testes de avaliação completa do modelo."""

    def test_evaluate_model_metrics(self, small_model):
        """Verifica que evaluate_model retorna métricas esperadas."""
        from src.data_preprocessing import SyntheticImageDataset
        from torch.utils.data import DataLoader

        features = np.random.rand(40, 192).astype(np.float32)
        labels = np.random.randint(0, 5, 40).astype(np.int64)
        dataset = SyntheticImageDataset(features, labels, img_size=8, n_channels=3)
        loader = DataLoader(dataset, batch_size=10)

        results = evaluate_model(small_model, loader, torch.device("cpu"), n_classes=5)

        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert "avg_inference_time_ms" in results
        assert results["avg_inference_time_ms"] > 0
        assert "throughput_samples_per_sec" in results
        assert results["throughput_samples_per_sec"] > 0
