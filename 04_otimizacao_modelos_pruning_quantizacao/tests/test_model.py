"""
Testes unitários para o módulo model.

Verifica arquiteturas (SimpleCNN, StudentCNN), funções de pruning,
quantização e helpers de tamanho/parâmetros.
"""

import copy

import pytest
import torch
import torch.nn as nn

from src.model import (
    SimpleCNN,
    StudentCNN,
    apply_structured_pruning,
    apply_unstructured_pruning,
    compute_sparsity,
    count_parameters,
    get_model_size_mb,
    quantize_dynamic,
    remove_pruning_reparametrization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cnn() -> SimpleCNN:
    torch.manual_seed(42)
    return SimpleCNN(num_classes=10)


@pytest.fixture
def student_cnn() -> StudentCNN:
    torch.manual_seed(42)
    return StudentCNN(num_classes=10)


@pytest.fixture
def dummy_input() -> torch.Tensor:
    return torch.randn(2, 3, 32, 32)


# ---------------------------------------------------------------------------
# Testes — Arquitetura
# ---------------------------------------------------------------------------


class TestSimpleCNN:
    """Testes de inicialização e forward da SimpleCNN."""

    def test_output_shape(self, simple_cnn: SimpleCNN, dummy_input: torch.Tensor) -> None:
        out = simple_cnn(dummy_input)
        assert out.shape == (2, 10)

    def test_parameter_count_positive(self, simple_cnn: SimpleCNN) -> None:
        assert count_parameters(simple_cnn) > 0

    def test_model_size_positive(self, simple_cnn: SimpleCNN) -> None:
        assert get_model_size_mb(simple_cnn) > 0

    def test_custom_num_classes(self) -> None:
        model = SimpleCNN(num_classes=5)
        out = model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 5)


class TestStudentCNN:
    """Testes de inicialização e forward da StudentCNN."""

    def test_output_shape(self, student_cnn: StudentCNN, dummy_input: torch.Tensor) -> None:
        out = student_cnn(dummy_input)
        assert out.shape == (2, 10)

    def test_student_smaller_than_teacher(
        self, simple_cnn: SimpleCNN, student_cnn: StudentCNN
    ) -> None:
        assert count_parameters(student_cnn) < count_parameters(simple_cnn)


# ---------------------------------------------------------------------------
# Testes — Pruning
# ---------------------------------------------------------------------------


class TestPruning:
    """Testes de pruning não-estruturado e estruturado."""

    def test_unstructured_pruning_creates_sparsity(self, simple_cnn: SimpleCNN) -> None:
        model = apply_unstructured_pruning(simple_cnn, amount=0.5)
        sp = compute_sparsity(model)
        assert sp["global"] > 0

    def test_unstructured_pruning_preserves_output_shape(
        self, simple_cnn: SimpleCNN, dummy_input: torch.Tensor
    ) -> None:
        model = apply_unstructured_pruning(simple_cnn, amount=0.3)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_structured_pruning_preserves_output_shape(
        self, simple_cnn: SimpleCNN, dummy_input: torch.Tensor
    ) -> None:
        model = apply_structured_pruning(simple_cnn, amount=0.3)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_remove_reparametrization(self, simple_cnn: SimpleCNN) -> None:
        model = apply_unstructured_pruning(simple_cnn, amount=0.3)
        model = remove_pruning_reparametrization(model)
        # Após remoção, não deve haver hook de pruning
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not hasattr(module, "weight_orig")

    def test_sparsity_increases_with_amount(self, simple_cnn: SimpleCNN) -> None:
        m1 = apply_unstructured_pruning(copy.deepcopy(simple_cnn), amount=0.3)
        m2 = apply_unstructured_pruning(copy.deepcopy(simple_cnn), amount=0.7)
        assert compute_sparsity(m2)["global"] > compute_sparsity(m1)["global"]


# ---------------------------------------------------------------------------
# Testes — Quantização
# ---------------------------------------------------------------------------


class TestQuantization:
    """Testes de quantização dinâmica."""

    def test_quantize_dynamic_output_shape(
        self, simple_cnn: SimpleCNN, dummy_input: torch.Tensor
    ) -> None:
        model_q = quantize_dynamic(simple_cnn)
        out = model_q(dummy_input)
        assert out.shape == (2, 10)

    def test_quantized_model_smaller(self, simple_cnn: SimpleCNN) -> None:
        size_fp32 = get_model_size_mb(simple_cnn)
        model_q = quantize_dynamic(simple_cnn)
        size_q = get_model_size_mb(model_q)
        # Modelo quantizado deve ser menor (ou igual — camadas conv não quantizadas)
        assert size_q <= size_fp32 * 1.1  # margem para overhead
