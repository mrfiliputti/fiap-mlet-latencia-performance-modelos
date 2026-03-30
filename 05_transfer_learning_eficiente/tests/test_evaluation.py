"""
Testes unitários para evaluation.py.

Valida funções de métricas de classificação, benchmarking e
criação de tabelas comparativas de PEFT.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_metrics,
    print_classification_report,
    create_peft_comparison_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_predictions():
    """Predições perfeitas (100% accuracy)."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    """Predições com erros para testar métricas não triviais."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 4, 4, 4])
    return y_true, y_pred


@pytest.fixture
def label_names():
    """Nomes das 5 classes do dataset."""
    return ["educacao", "financeiro", "juridico", "saude", "tecnologia"]


# ---------------------------------------------------------------------------
# Testes de calculate_metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Testes para calculate_metrics."""

    def test_perfect_accuracy(self, perfect_predictions, label_names):
        """Verifica accuracy = 1.0 para predições perfeitas."""
        y_true, y_pred = perfect_predictions
        metrics = calculate_metrics(y_true, y_pred, label_names=label_names)
        assert metrics["accuracy"] == 1.0

    def test_perfect_f1(self, perfect_predictions, label_names):
        """Verifica F1 = 1.0 para predições perfeitas."""
        y_true, y_pred = perfect_predictions
        metrics = calculate_metrics(y_true, y_pred, label_names=label_names)
        assert metrics["f1_macro"] == 1.0

    def test_imperfect_accuracy(self, imperfect_predictions, label_names):
        """Verifica accuracy < 1.0 para predições com erros."""
        y_true, y_pred = imperfect_predictions
        metrics = calculate_metrics(y_true, y_pred, label_names=label_names)
        assert 0.0 < metrics["accuracy"] < 1.0

    def test_metrics_keys(self, perfect_predictions):
        """Verifica que todas as chaves esperadas estão presentes."""
        y_true, y_pred = perfect_predictions
        metrics = calculate_metrics(y_true, y_pred)
        expected_keys = {"accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"}
        assert expected_keys.issubset(metrics.keys())

    def test_per_class_metrics(self, perfect_predictions, label_names):
        """Verifica métricas por classe quando label_names fornecido."""
        y_true, y_pred = perfect_predictions
        metrics = calculate_metrics(y_true, y_pred, label_names=label_names)
        assert "per_class" in metrics
        assert len(metrics["per_class"]) == 5

    def test_metrics_without_labels(self, imperfect_predictions):
        """Verifica que funciona sem label_names (sem per_class)."""
        y_true, y_pred = imperfect_predictions
        metrics = calculate_metrics(y_true, y_pred, label_names=None)
        assert "per_class" not in metrics
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Testes de print_classification_report
# ---------------------------------------------------------------------------


class TestClassificationReport:
    """Testes para print_classification_report."""

    def test_report_returns_string(self, perfect_predictions, label_names):
        """Verifica que o relatório é uma string."""
        y_true, y_pred = perfect_predictions
        report = print_classification_report(y_true, y_pred, label_names=label_names)
        assert isinstance(report, str)
        assert "precision" in report.lower()


# ---------------------------------------------------------------------------
# Testes de create_peft_comparison_table
# ---------------------------------------------------------------------------


class TestPEFTComparisonTable:
    """Testes para a tabela comparativa de técnicas PEFT."""

    def test_table_has_correct_rows(self):
        """Verifica que a tabela tem 6 técnicas."""
        table = create_peft_comparison_table()
        assert len(table) == 6

    def test_table_has_required_columns(self):
        """Verifica colunas obrigatórias."""
        table = create_peft_comparison_table()
        expected_cols = {"technique", "trainable_params", "gpu_memory_gb"}
        assert expected_cols.issubset(set(table.columns))

    def test_table_includes_lora(self):
        """Verifica que LoRA e QLoRA estão na tabela."""
        table = create_peft_comparison_table()
        techniques = table["technique"].tolist()
        assert any("LoRA" in t for t in techniques)
        assert any("QLoRA" in t for t in techniques)
