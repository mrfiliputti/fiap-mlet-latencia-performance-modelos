"""
Testes unitários para src/evaluation.py.

Valida cálculo de métricas SRE, error budgets e funções de plot.
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # backend sem GUI para testes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor
from src.evaluation import (
    calculate_error_budget,
    calculate_metrics,
    plot_canary_comparison,
    plot_error_budget_gauge,
    plot_latency_distribution,
    plot_slo_dashboard,
)


# ---- Fixtures --------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dp = DataPreprocessor(seed=42)
    df = dp.generate_dataset(n_samples=500)
    df = dp.clean_data(df)
    return dp.prepare_features(df)


# ---- Testes de calculate_metrics --------------------------------------


class TestCalculateMetrics:
    def test_keys_present(self, sample_df: pd.DataFrame) -> None:
        metrics = calculate_metrics(sample_df)
        assert "slis" in metrics
        assert "slos" in metrics
        assert "per_modality" in metrics
        assert "per_version" in metrics
        assert "total_requests" in metrics

    def test_total_requests(self, sample_df: pd.DataFrame) -> None:
        metrics = calculate_metrics(sample_df)
        assert metrics["total_requests"] == len(sample_df)

    def test_slos_have_two_entries(self, sample_df: pd.DataFrame) -> None:
        metrics = calculate_metrics(sample_df)
        assert len(metrics["slos"]) == 2

    def test_per_modality_keys(self, sample_df: pd.DataFrame) -> None:
        metrics = calculate_metrics(sample_df)
        for mod in sample_df["modality"].unique():
            assert mod in metrics["per_modality"]


# ---- Testes de calculate_error_budget ---------------------------------


class TestCalculateErrorBudget:
    def test_budget_basic(self) -> None:
        budget = calculate_error_budget(
            total_requests=10_000,
            error_count=5,
            slo_availability=0.999,
        )
        assert budget["allowed_errors"] == 10.0
        assert budget["observed_errors"] == 5
        assert budget["budget_remaining"] == 5.0
        assert budget["budget_remaining_pct"] == 50.0

    def test_budget_exhausted(self) -> None:
        budget = calculate_error_budget(
            total_requests=10_000,
            error_count=20,
            slo_availability=0.999,
        )
        assert budget["budget_remaining"] == 0.0
        assert budget["budget_remaining_pct"] == 0.0

    def test_downtime_calculated(self) -> None:
        budget = calculate_error_budget(
            total_requests=10_000,
            error_count=0,
            slo_availability=0.999,
            window_days=30,
        )
        assert budget["downtime_budget_minutes"] > 0
        assert budget["downtime_consumed_minutes"] == 0.0


# ---- Testes de plots (apenas verificam que não lançam exceção) --------


class TestPlots:
    def test_plot_latency_distribution(self, sample_df: pd.DataFrame) -> None:
        fig = plot_latency_distribution(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_error_budget_gauge(self) -> None:
        fig = plot_error_budget_gauge(75.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_canary_comparison(self, sample_df: pd.DataFrame) -> None:
        fig = plot_canary_comparison(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_slo_dashboard(self, sample_df: pd.DataFrame) -> None:
        metrics = calculate_metrics(sample_df)
        fig = plot_slo_dashboard(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
