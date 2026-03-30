"""
Testes unitários para src/evaluation.py.

Verifica cálculo de percentis de latência, conformidade com SLOs
e funções de visualização (sem renderização gráfica).
"""

import numpy as np
import pytest

from src.evaluation import ServingMetricsEvaluator
from src.utils import compute_memory_savings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator(tmp_path):
    """Evaluator com diretório temporário."""
    return ServingMetricsEvaluator(output_dir=str(tmp_path))


@pytest.fixture
def sample_latencies():
    """Array de latências simuladas para testes."""
    rng = np.random.default_rng(42)
    return rng.exponential(scale=50.0, size=1000)


# ---------------------------------------------------------------------------
# Testes de percentis
# ---------------------------------------------------------------------------

class TestLatencyPercentiles:
    """Testes para cálculo de percentis de latência."""

    def test_percentiles_keys(self, sample_latencies):
        """Verifica chaves retornadas."""
        result = ServingMetricsEvaluator.calculate_latency_percentiles(
            sample_latencies
        )
        assert "mean" in result
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result

    def test_percentiles_ordering(self, sample_latencies):
        """P50 < P95 < P99 < max."""
        result = ServingMetricsEvaluator.calculate_latency_percentiles(
            sample_latencies
        )
        assert result["p50"] <= result["p95"]
        assert result["p95"] <= result["p99"]
        assert result["p99"] <= result["max"]

    def test_percentiles_custom(self):
        """Verifica percentis customizados."""
        latencies = np.arange(1, 101, dtype=float)  # 1..100
        result = ServingMetricsEvaluator.calculate_latency_percentiles(
            latencies, percentiles=[25.0, 75.0]
        )
        assert "p25" in result
        assert "p75" in result
        # P25 de 1..100 ≈ 25.75
        assert 24 <= result["p25"] <= 27


# ---------------------------------------------------------------------------
# Testes de SLO compliance
# ---------------------------------------------------------------------------

class TestSLOCompliance:
    """Testes para verificação de SLOs."""

    def test_compliant_scenario(self):
        """Cenário que deve cumprir SLOs."""
        latencies = np.random.default_rng(42).uniform(10, 100, size=100)
        result = ServingMetricsEvaluator.check_slo_compliance(
            latencies, slo_p99_ms=500.0, slo_throughput_min=1.0,
            total_time_s=10.0,
        )
        assert result["p99_compliant"] is True
        assert result["throughput_compliant"] is True
        assert result["overall_compliant"] is True

    def test_non_compliant_latency(self):
        """Cenário que viola SLO de latência."""
        latencies = np.random.default_rng(42).uniform(400, 600, size=100)
        result = ServingMetricsEvaluator.check_slo_compliance(
            latencies, slo_p99_ms=500.0, slo_throughput_min=1.0,
            total_time_s=10.0,
        )
        assert result["p99_compliant"] is False

    def test_non_compliant_throughput(self):
        """Cenário que viola SLO de throughput."""
        latencies = np.array([10.0, 20.0])  # Apenas 2 requisições
        result = ServingMetricsEvaluator.check_slo_compliance(
            latencies, slo_p99_ms=500.0, slo_throughput_min=100.0,
            total_time_s=10.0,
        )
        assert result["throughput_compliant"] is False


# ---------------------------------------------------------------------------
# Testes de memory savings
# ---------------------------------------------------------------------------

class TestMemorySavings:
    """Testes para cálculo de economia de memória."""

    def test_paged_better_than_contiguous(self):
        """PagedAttention deve desperdiçar menos que contígua."""
        seq_lengths = np.array([32, 64, 128, 256, 512])
        result = compute_memory_savings(seq_lengths, block_size=16)
        assert result["paged_waste"] < result["contiguous_waste"]

    def test_paged_utilization_high(self):
        """PagedAttention utilização > 90% para blocos de 16."""
        rng = np.random.default_rng(42)
        seq_lengths = rng.integers(50, 500, size=100)
        result = compute_memory_savings(seq_lengths, block_size=16)
        assert result["paged_utilization"] > 0.90

    def test_savings_nonnegative(self):
        """Economia deve ser não-negativa."""
        seq_lengths = np.array([100, 200, 300])
        result = compute_memory_savings(seq_lengths, block_size=16)
        assert result["savings_vs_contiguous"] >= 0
