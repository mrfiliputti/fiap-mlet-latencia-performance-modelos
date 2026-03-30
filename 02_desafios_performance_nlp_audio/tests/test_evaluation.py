"""
Testes unitários para src/evaluation.py.

Cobre calculate_metrics, attention_memory_theoretical, kv_cache_memory
e PerformanceEvaluator.
"""

import numpy as np
import pytest

from src.evaluation import (
    PerformanceEvaluator,
    attention_memory_theoretical,
    calculate_metrics,
    kv_cache_memory,
)


# ---- Tests: calculate_metrics ---------------------------------------- #

def test_calculate_metrics_perfect():
    """Predição perfeita resulta em MSE=0, R²=1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = calculate_metrics(y, y)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)


def test_calculate_metrics_known_values():
    """Verifica métricas com valores conhecidos."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    metrics = calculate_metrics(y_true, y_pred)
    # MSE = mean((0.5)² + (0.5)² + (0.5)²) = 0.25
    assert metrics["mse"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["mae"] == pytest.approx(0.5, abs=1e-6)


def test_calculate_metrics_r2_range():
    """R² deve estar em (-∞, 1]. Para predição constante → 0."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, 100)
    y_pred = rng.normal(0, 1, 100)
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["r2"] <= 1.0


# ---- Tests: attention_memory_theoretical ----------------------------- #

def test_vanilla_memory_quadratic():
    """Memória vanilla cresce quadraticamente com seq_length."""
    mem_128 = attention_memory_theoretical(128, 128, 4, "vanilla")
    mem_256 = attention_memory_theoretical(256, 128, 4, "vanilla")
    # 256² / 128² = 4
    assert mem_256 / mem_128 == pytest.approx(4.0, rel=1e-6)


def test_flash_memory_linear():
    """Memória flash cresce linearmente com seq_length."""
    mem_128 = attention_memory_theoretical(128, 128, 4, "flash")
    mem_256 = attention_memory_theoretical(256, 128, 4, "flash")
    # 256 / 128 = 2
    assert mem_256 / mem_128 == pytest.approx(2.0, rel=1e-6)


def test_unknown_attention_raises():
    """Tipo de atenção desconhecido lança ValueError."""
    with pytest.raises(ValueError, match="desconhecido"):
        attention_memory_theoretical(128, 128, 4, "inexistente")


# ---- Tests: kv_cache_memory ------------------------------------------ #

def test_kv_cache_memory_formula():
    """Verifica fórmula: 2 × L × H × n × d_k × bytes."""
    # d_model=128, heads=4 → d_k=32
    mem = kv_cache_memory(seq_length=100, d_model=128, num_layers=2,
                          num_heads=4, dtype_bytes=4)
    # 2 × 2 × 4 × 100 × 32 × 4 = 204800
    expected = 2 * 2 * 4 * 100 * 32 * 4
    assert mem == expected


def test_kv_cache_memory_fp16_vs_fp32():
    """FP16 usa metade da memória de FP32."""
    mem_fp32 = kv_cache_memory(512, 256, 4, 4, dtype_bytes=4)
    mem_fp16 = kv_cache_memory(512, 256, 4, 4, dtype_bytes=2)
    assert mem_fp16 * 2 == mem_fp32


# ---- Tests: PerformanceEvaluator ------------------------------------- #

def test_evaluator_results_to_dataframe():
    """results_to_dataframe converte BenchmarkResult para DataFrame."""
    from src.training import BenchmarkResult

    results = [
        BenchmarkResult("vanilla", 64, 128, 4, 1.5, 1024, 100.0),
        BenchmarkResult("linear", 64, 128, 4, 0.5, 512, 300.0),
    ]
    evaluator = PerformanceEvaluator()
    df = evaluator.results_to_dataframe(results)
    assert len(df) == 2
    assert "attention_type" in df.columns
    assert "latency_ms" in df.columns
