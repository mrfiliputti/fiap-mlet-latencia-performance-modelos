"""
Testes unitários para src/model.py.

Valida SLOMonitor, CanaryAnalyzer e MultimodalOrchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor
from src.model import (
    CanaryAnalyzer,
    CanaryTestResult,
    MultimodalOrchestrator,
    SLOMonitor,
    SLOResult,
)


# ---- Fixtures --------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dp = DataPreprocessor(seed=42)
    df = dp.generate_dataset(n_samples=1000)
    return dp.clean_data(df)


@pytest.fixture
def monitor() -> SLOMonitor:
    return SLOMonitor(slo_latency_p99_ms=200.0, slo_availability=0.999)


# ---- SLOMonitor -------------------------------------------------------


class TestSLOMonitor:
    def test_calculate_slis_keys(self, monitor: SLOMonitor, sample_df: pd.DataFrame) -> None:
        slis = monitor.calculate_slis(sample_df)
        expected_keys = {
            "latency_p50", "latency_p95", "latency_p99",
            "latency_max", "error_rate", "availability", "throughput_rps",
        }
        assert set(slis.keys()) == expected_keys

    def test_availability_range(self, monitor: SLOMonitor, sample_df: pd.DataFrame) -> None:
        slis = monitor.calculate_slis(sample_df)
        assert 0.0 <= slis["availability"] <= 1.0

    def test_evaluate_slos_returns_list(self, monitor: SLOMonitor, sample_df: pd.DataFrame) -> None:
        results = monitor.evaluate_slos(sample_df)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, SLOResult) for r in results)

    def test_erlang_c_stable_system(self) -> None:
        """Para sistema estável (rho < 1), Erlang-C deve ser < 1."""
        ec = SLOMonitor.erlang_c(c=5, rho=0.5)
        assert 0 < ec < 1

    def test_erlang_c_saturated(self) -> None:
        """Sistema saturado (rho >= 1) retorna 1.0."""
        ec = SLOMonitor.erlang_c(c=3, rho=1.0)
        assert ec == 1.0

    def test_mean_queue_wait_decreases_with_replicas(self) -> None:
        """Mais réplicas → menor Wq (retornos decrescentes)."""
        wq_5 = SLOMonitor.mean_queue_wait(lam=80, mu=20, c=5)
        wq_8 = SLOMonitor.mean_queue_wait(lam=80, mu=20, c=8)
        assert wq_8 < wq_5

    def test_mean_queue_wait_saturated(self) -> None:
        """Sistema saturado retorna inf."""
        wq = SLOMonitor.mean_queue_wait(lam=100, mu=20, c=5)
        assert wq == float("inf")


# ---- CanaryAnalyzer ---------------------------------------------------


class TestCanaryAnalyzer:
    def test_latency_same_distribution(self) -> None:
        """Mesma distribuição → não significativo → continue."""
        rng = np.random.default_rng(42)
        stable = rng.normal(100, 10, 500)
        canary = rng.normal(100, 10, 500)
        analyzer = CanaryAnalyzer(alpha=0.05)
        result = analyzer.test_latency(stable, canary)
        assert isinstance(result, CanaryTestResult)
        assert result.decision == "continue"

    def test_latency_degradation_triggers_rollback(self) -> None:
        """Degradação significativa → rollback."""
        rng = np.random.default_rng(42)
        stable = rng.normal(100, 10, 500)
        canary = rng.normal(130, 10, 500)  # 30% mais lento
        analyzer = CanaryAnalyzer(alpha=0.05, latency_degradation_threshold=0.10)
        result = analyzer.test_latency(stable, canary)
        assert result.significant
        assert result.decision == "rollback"

    def test_error_rate_same(self) -> None:
        """Mesma taxa de erro → continue."""
        analyzer = CanaryAnalyzer(alpha=0.05)
        result = analyzer.test_error_rate(
            stable_errors=20, stable_total=1000,
            canary_errors=3, canary_total=100,
        )
        assert result.decision == "continue"

    def test_full_analysis_returns_dict(self, sample_df: pd.DataFrame) -> None:
        analyzer = CanaryAnalyzer()
        results = analyzer.full_analysis(sample_df)
        assert "latency" in results
        assert "error_rate" in results


# ---- MultimodalOrchestrator -------------------------------------------


class TestMultimodalOrchestrator:
    def test_predict_all_modalities(self) -> None:
        orch = MultimodalOrchestrator(seed=42)
        result = orch.predict()
        assert "individual" in result
        assert set(result["individual"].keys()) == {"text", "image", "audio"}
        assert result["total_latency_ms"] > 0

    def test_predict_single_modality(self) -> None:
        orch = MultimodalOrchestrator(seed=42)
        result = orch.predict(modalities=["text"])
        assert "text" in result["individual"]
        assert len(result["individual"]) == 1

    def test_invalid_fusion_strategy(self) -> None:
        with pytest.raises(ValueError):
            MultimodalOrchestrator(fusion_strategy="invalid")

    def test_score_returns_metrics(self) -> None:
        orch = MultimodalOrchestrator(seed=42)
        scores = orch.score(n_requests=50)
        assert "latency_p50" in scores
        assert "latency_p99" in scores
        assert "error_rate" in scores
        assert scores["latency_p50"] <= scores["latency_p99"]
