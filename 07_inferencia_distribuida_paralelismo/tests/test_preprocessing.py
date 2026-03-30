"""
Testes unitários para src/data_preprocessing.py.

Verifica geração de workload sintético, cálculo de KV-Cache e
particionamento de requisições.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import WorkloadPreprocessor, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessor():
    """Instância do WorkloadPreprocessor com seed fixa."""
    return WorkloadPreprocessor(seed=42)


@pytest.fixture
def sample_workload(preprocessor):
    """Workload sintético com 100 requisições."""
    return preprocessor.generate_workload(n_requests=100)


# ---------------------------------------------------------------------------
# Testes de geração de workload
# ---------------------------------------------------------------------------

class TestWorkloadGeneration:
    """Testes para geração de workload sintético."""

    def test_generate_workload_shape(self, preprocessor):
        """Verifica shape do DataFrame gerado."""
        df = preprocessor.generate_workload(n_requests=50)
        assert df.shape[0] == 50
        assert df.shape[1] == 8

    def test_generate_workload_columns(self, sample_workload):
        """Verifica que todas colunas esperadas existem."""
        expected_cols = [
            "request_id", "arrival_time_ms", "prompt_tokens",
            "max_gen_tokens", "actual_gen_tokens", "model_size_b",
            "num_attention_heads", "hidden_dim",
        ]
        assert list(sample_workload.columns) == expected_cols

    def test_arrival_times_monotonic(self, sample_workload):
        """Verifica que tempos de chegada são monótonos crescentes."""
        times = sample_workload["arrival_time_ms"].values
        assert np.all(np.diff(times) >= 0)

    def test_prompt_tokens_range(self, sample_workload):
        """Verifica que prompt_tokens está no range esperado."""
        assert sample_workload["prompt_tokens"].min() >= 10
        assert sample_workload["prompt_tokens"].max() <= 2048

    def test_actual_gen_tokens_leq_max(self, sample_workload):
        """Verifica que tokens gerados ≤ max_gen_tokens."""
        assert (
            sample_workload["actual_gen_tokens"]
            <= sample_workload["max_gen_tokens"]
        ).all()

    def test_model_sizes_valid(self, sample_workload):
        """Verifica que tamanhos de modelo são válidos."""
        valid_sizes = {7.0, 13.0, 70.0}
        actual_sizes = set(sample_workload["model_size_b"].unique())
        assert actual_sizes.issubset(valid_sizes)

    def test_model_config_consistency(self, sample_workload):
        """Verifica que attention_heads e hidden_dim são consistentes com o modelo."""
        for _, row in sample_workload.iterrows():
            model_key = f"{int(row['model_size_b'])}B"
            expected = MODEL_CONFIGS[model_key]
            assert row["num_attention_heads"] == expected["num_attention_heads"]
            assert row["hidden_dim"] == expected["hidden_dim"]

    def test_reproducibility(self):
        """Verifica que seed fixa gera resultados idênticos."""
        p1 = WorkloadPreprocessor(seed=123)
        p2 = WorkloadPreprocessor(seed=123)
        df1 = p1.generate_workload(n_requests=20)
        df2 = p2.generate_workload(n_requests=20)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Testes de KV-Cache
# ---------------------------------------------------------------------------

class TestKVCacheComputation:
    """Testes para cálculo de memória KV-Cache."""

    def test_compute_kv_cache_adds_columns(self, preprocessor, sample_workload):
        """Verifica que colunas de KV-Cache são adicionadas."""
        result = preprocessor.compute_kv_cache_memory(sample_workload)
        for col in ["total_tokens", "num_blocks", "total_kv_cache_mb"]:
            assert col in result.columns

    def test_num_blocks_formula(self, preprocessor, sample_workload):
        """Verifica fórmula ceil(total_tokens / block_size)."""
        block_size = 16
        result = preprocessor.compute_kv_cache_memory(
            sample_workload, block_size=block_size
        )
        expected_blocks = np.ceil(result["total_tokens"] / block_size).astype(int)
        np.testing.assert_array_equal(
            result["num_blocks"].values, expected_blocks.values
        )

    def test_kv_cache_positive(self, preprocessor, sample_workload):
        """Verifica que KV-Cache é sempre positivo."""
        result = preprocessor.compute_kv_cache_memory(sample_workload)
        assert (result["total_kv_cache_mb"] > 0).all()


# ---------------------------------------------------------------------------
# Testes de particionamento
# ---------------------------------------------------------------------------

class TestPartitioning:
    """Testes para particionamento de workload."""

    def test_split_by_model_keys(self, preprocessor, sample_workload):
        """Verifica que split gera chaves corretas."""
        splits = preprocessor.split_by_model(sample_workload)
        for key in splits:
            assert key in ["7B", "13B", "70B"]

    def test_split_preserves_total(self, preprocessor, sample_workload):
        """Verifica que total de requisições é preservado no split."""
        splits = preprocessor.split_by_model(sample_workload)
        total = sum(len(df) for df in splits.values())
        assert total == len(sample_workload)

    def test_prepare_batching_windows_nonempty(self, preprocessor, sample_workload):
        """Verifica que janelas de batching não são vazias."""
        windows = preprocessor.prepare_batching_windows(
            sample_workload, window_ms=500.0
        )
        assert len(windows) > 0
        for w in windows:
            assert len(w) > 0
