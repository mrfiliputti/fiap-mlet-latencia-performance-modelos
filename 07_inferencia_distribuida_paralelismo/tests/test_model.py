"""
Testes unitários para src/model.py.

Verifica simuladores de tensor parallelism, pipeline parallelism,
PagedAttention KV-Cache, continuous batching, speculative decoding,
e funções de Amdahl/Gustafson.
"""

import numpy as np
import pytest

from src.model import (
    TensorParallelismSimulator,
    PipelineParallelismSimulator,
    PagedKVCacheManager,
    ContinuousBatchingScheduler,
    StaticBatchingScheduler,
    ScheduledRequest,
    SpeculativeDecoder,
    amdahl_speedup,
    gustafson_speedup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_matrices(rng):
    """Matrizes pequenas para testes rápidos."""
    X = rng.standard_normal((8, 64)).astype(np.float32)
    W = rng.standard_normal((64, 64)).astype(np.float32)
    return X, W


# ---------------------------------------------------------------------------
# Tensor Parallelism
# ---------------------------------------------------------------------------

class TestTensorParallelism:
    """Testes do TensorParallelismSimulator (Megatron-LM)."""

    def test_column_partition_shapes(self, small_matrices):
        """Verifica shapes das partições por coluna."""
        _, W = small_matrices
        sim = TensorParallelismSimulator(num_devices=4)
        parts = sim.partition_column_wise(W)
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (64, 16)

    def test_row_partition_shapes(self, small_matrices):
        """Verifica shapes das partições por linha."""
        _, W = small_matrices
        sim = TensorParallelismSimulator(num_devices=4)
        parts = sim.partition_row_wise(W)
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (16, 64)

    def test_column_parallel_correctness(self, small_matrices):
        """Resultado paralelo por coluna é igual ao sequencial."""
        X, W = small_matrices
        sim = TensorParallelismSimulator(num_devices=2)
        output, _ = sim.simulate_forward_column_parallel(X, W)
        expected = X @ W
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_row_parallel_correctness(self, small_matrices):
        """Resultado paralelo por linha é igual ao sequencial."""
        X, W = small_matrices
        sim = TensorParallelismSimulator(num_devices=2)
        output, _ = sim.simulate_forward_row_parallel(X, W)
        expected = X @ W
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_invalid_num_devices(self):
        """Rejeita num_devices < 1."""
        with pytest.raises(ValueError):
            TensorParallelismSimulator(num_devices=0)

    def test_metrics_returned(self, small_matrices):
        """Verifica que métricas de tempo são retornadas."""
        X, W = small_matrices
        sim = TensorParallelismSimulator(num_devices=2)
        _, metrics = sim.simulate_forward_column_parallel(X, W)
        assert "total_time" in metrics
        assert "device_compute_times" in metrics
        assert len(metrics["device_compute_times"]) == 2


# ---------------------------------------------------------------------------
# Pipeline Parallelism
# ---------------------------------------------------------------------------

class TestPipelineParallelism:
    """Testes do PipelineParallelismSimulator (GPipe)."""

    def test_bubble_fraction_formula(self):
        """Verifica fórmula (P-1)/(M+P-1)."""
        sim = PipelineParallelismSimulator(num_stages=4)
        # M=4, P=4 -> (4-1)/(4+4-1) = 3/7
        expected = 3 / 7
        assert abs(sim.compute_bubble_fraction(4) - expected) < 1e-10

    def test_bubble_fraction_single_microbatch(self):
        """Com M=1, bubble é máximo: (P-1)/P."""
        sim = PipelineParallelismSimulator(num_stages=4)
        expected = 3 / 4
        assert abs(sim.compute_bubble_fraction(1) - expected) < 1e-10

    def test_bubble_fraction_many_microbatches(self):
        """Com M >> P, bubble tende a 0."""
        sim = PipelineParallelismSimulator(num_stages=4)
        bf = sim.compute_bubble_fraction(1000)
        assert bf < 0.01

    def test_pipeline_outputs_shape(self, rng):
        """Verifica que outputs têm shape correto."""
        sim = PipelineParallelismSimulator(num_stages=2, hidden_dim=32)
        batch = rng.standard_normal((16, 32)).astype(np.float32)
        outputs, metrics = sim.simulate_pipeline(batch, num_micro_batches=4)
        assert len(outputs) == 4
        assert metrics["num_stages"] == 2
        assert metrics["num_micro_batches"] == 4


# ---------------------------------------------------------------------------
# PagedAttention KV-Cache
# ---------------------------------------------------------------------------

class TestPagedKVCache:
    """Testes do PagedKVCacheManager (Kwon et al., 2023)."""

    def test_initial_state(self):
        """Estado inicial: todos blocos livres."""
        mgr = PagedKVCacheManager(num_blocks=64, block_size=16)
        assert mgr.free_blocks == 64
        assert mgr.utilization == 0.0

    def test_allocate_basic(self):
        """Alocação básica: 32 tokens = 2 blocos com B=16."""
        mgr = PagedKVCacheManager(num_blocks=64, block_size=16)
        assert mgr.allocate(sequence_id=0, num_tokens=32)
        assert len(mgr.block_tables[0]) == 2
        assert mgr.free_blocks == 62

    def test_allocate_ceil(self):
        """Alocação usa ceil: 33 tokens = 3 blocos com B=16."""
        mgr = PagedKVCacheManager(num_blocks=64, block_size=16)
        mgr.allocate(sequence_id=0, num_tokens=33)
        assert len(mgr.block_tables[0]) == 3

    def test_allocate_oom(self):
        """Rejeita alocação quando memória insuficiente."""
        mgr = PagedKVCacheManager(num_blocks=4, block_size=16)
        assert mgr.allocate(sequence_id=0, num_tokens=100) is False

    def test_free_blocks(self):
        """Libera blocos corretamente."""
        mgr = PagedKVCacheManager(num_blocks=64, block_size=16)
        mgr.allocate(sequence_id=0, num_tokens=48)  # 3 blocos
        freed = mgr.free(sequence_id=0)
        assert freed == 3
        assert mgr.free_blocks == 64

    def test_prefix_sharing(self):
        """Prefix sharing incrementa ref_count."""
        mgr = PagedKVCacheManager(num_blocks=64, block_size=16)
        mgr.allocate(sequence_id=0, num_tokens=32)  # 2 blocos
        mgr.share_prefix(source_seq_id=0, target_seq_id=1, num_prefix_blocks=2)
        # Verifica ref_count > 1 nos blocos compartilhados
        shared_ids = mgr.block_tables[0][:2]
        for bid in shared_ids:
            assert mgr.blocks[bid].ref_count == 2

    def test_memory_stats(self):
        """Verifica estatísticas de memória."""
        mgr = PagedKVCacheManager(num_blocks=100, block_size=16)
        mgr.allocate(sequence_id=0, num_tokens=32)
        stats = mgr.get_memory_stats()
        assert stats["total_blocks"] == 100
        assert stats["used_blocks"] == 2
        assert stats["free_blocks"] == 98
        assert abs(stats["utilization"] - 0.02) < 1e-10


# ---------------------------------------------------------------------------
# Continuous Batching
# ---------------------------------------------------------------------------

class TestContinuousBatching:
    """Testes do ContinuousBatchingScheduler (Orca)."""

    def test_add_and_process(self):
        """Processa requisições até completar."""
        scheduler = ContinuousBatchingScheduler(
            max_batch_size=4, max_total_tokens=1000
        )
        for i in range(3):
            scheduler.add_request(ScheduledRequest(
                request_id=i, prompt_tokens=10, max_gen_tokens=5
            ))
        history = scheduler.run_to_completion(max_steps=100)
        assert len(scheduler.completed) == 3

    def test_all_requests_complete(self):
        """Todas requisições devem completar."""
        scheduler = ContinuousBatchingScheduler(
            max_batch_size=8, max_total_tokens=2000
        )
        for i in range(10):
            scheduler.add_request(ScheduledRequest(
                request_id=i, prompt_tokens=20, max_gen_tokens=10
            ))
        scheduler.run_to_completion()
        assert len(scheduler.completed) == 10

    def test_summary_metrics(self):
        """Verifica que summary retorna métricas corretas."""
        scheduler = ContinuousBatchingScheduler(
            max_batch_size=4, max_total_tokens=500
        )
        for i in range(5):
            scheduler.add_request(ScheduledRequest(
                request_id=i, prompt_tokens=10, max_gen_tokens=3
            ))
        scheduler.run_to_completion()
        summary = scheduler.get_summary()
        assert summary["total_completed"] == 5
        assert summary["avg_latency_steps"] > 0


# ---------------------------------------------------------------------------
# Speculative Decoding
# ---------------------------------------------------------------------------

class TestSpeculativeDecoding:
    """Testes do SpeculativeDecoder (Leviathan et al., 2023)."""

    def test_generates_correct_tokens(self):
        """Verifica que gera o número correto de tokens."""
        decoder = SpeculativeDecoder(
            draft_acceptance_rate=0.8, speculation_length=5, seed=42
        )
        result = decoder.simulate_generation(100)
        assert result["total_tokens"] == 100

    def test_speedup_positive(self):
        """Speedup deve ser > 1 com alta acceptance rate."""
        decoder = SpeculativeDecoder(
            draft_acceptance_rate=0.9, speculation_length=5, seed=42
        )
        result = decoder.simulate_generation(200)
        assert result["speedup"] > 1.0

    def test_invalid_acceptance_rate(self):
        """Rejeita acceptance rate inválida."""
        with pytest.raises(ValueError):
            SpeculativeDecoder(draft_acceptance_rate=0.0)


# ---------------------------------------------------------------------------
# Amdahl & Gustafson
# ---------------------------------------------------------------------------

class TestAmdahlGustafson:
    """Testes das leis de Amdahl e Gustafson."""

    def test_amdahl_single_processor(self):
        """Com 1 processador, speedup é 1."""
        assert amdahl_speedup(0.5, 1) == 1.0

    def test_amdahl_limit(self):
        """Speedup converge para 1/s com muitos processadores."""
        s = 0.1
        speedup = amdahl_speedup(s, 10000)
        assert abs(speedup - 1 / s) < 0.1

    def test_gustafson_single_processor(self):
        """Com 1 processador, speedup é 1."""
        assert gustafson_speedup(0.5, 1) == 1.0

    def test_gustafson_linear_scaling(self):
        """Com s=0, Gustafson escala linearmente."""
        assert gustafson_speedup(0.0, 16) == 16.0
