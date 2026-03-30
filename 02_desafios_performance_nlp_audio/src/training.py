"""
Benchmark e treinamento de mecanismos de atenção.

Implementa funções para medir latência, memória e throughput dos
diferentes mecanismos de atenção discutidos na seção 'Tabela
Comparativa: Abordagens de Atenção Eficiente' e na seção
'Decomposição da Latência em Transformers' do material da Aula 02.

Referências:
    Dao, T. et al. (2022). FlashAttention. NeurIPS.
    Tay, Y. et al. (2022). Efficient Transformers: A Survey.
        ACM Computing Surveys, v. 55, n. 6.
    Shoeybi, M. et al. (2019). Megatron-LM. arXiv:1909.08053.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import (
    KVCache,
    LinearAttention,
    LocalAttention,
    SimpleTransformerBlock,
    TiledAttention,
    VanillaAttention,
)


RANDOM_SEED = 42


# ================================================================== #
#  Dataclass para resultados de benchmark                            #
# ================================================================== #

@dataclass
class BenchmarkResult:
    """Resultado de um benchmark individual."""

    attention_type: str
    seq_length: int
    d_model: int
    num_heads: int
    latency_ms: float
    memory_bytes: int
    throughput_tokens_per_sec: float


# ================================================================== #
#  Funções principais de benchmark                                   #
# ================================================================== #

def benchmark_single(
    attention_module: Any,
    X: np.ndarray,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Tuple[float, int]:
    """Executa benchmark de latência para um módulo de atenção.

    Implementa medição de latência conforme a decomposição discutida na
    seção 'Decomposição da Latência em Transformers': mede tempo total
    do forward pass incluindo projeções, softmax e multiplicação por V.

    Args:
        attention_module: Módulo com método ``forward(X)``.
        X: Tensor de entrada (batch, seq_len, d_model).
        n_warmup: Rodadas de warm-up (descartadas).
        n_runs: Rodadas de medição.

    Returns:
        Tupla (latência média em ms, memória estimada em bytes).
    """
    seq_length = X.shape[1]
    batch_size = X.shape[0]

    # Warm-up
    for _ in range(n_warmup):
        attention_module.forward(X)

    # Medição
    times: List[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        attention_module.forward(X)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_latency = float(np.mean(times))
    memory = attention_module.memory_footprint_bytes(seq_length, batch_size)
    return avg_latency, memory


def benchmark_attention_mechanisms(
    seq_lengths: Optional[List[int]] = None,
    d_model: int = 128,
    num_heads: int = 4,
    batch_size: int = 2,
    n_runs: int = 5,
    seed: int = RANDOM_SEED,
) -> List[BenchmarkResult]:
    """Compara múltiplos mecanismos de atenção em diferentes comprimentos.

    Reproduz a análise da 'Tabela Comparativa: Abordagens de Atenção
    Eficiente' (Tay et al., 2022), medindo latência e memória para:
    - Vanilla Attention: O(n²·d) tempo, O(n²) memória
    - Linear Attention:  O(n·d²) tempo, O(n·d) memória
    - Local Attention:   O(n·w·d) tempo, O(n·w) memória
    - Tiled Attention:   O(n²·d) tempo, O(n) memória (Flash-style)

    Args:
        seq_lengths: Lista de comprimentos de sequência para testar.
        d_model: Dimensão do modelo.
        num_heads: Número de cabeças de atenção.
        batch_size: Tamanho do batch.
        n_runs: Repetições para média.
        seed: Seed para reprodutibilidade.

    Returns:
        Lista de BenchmarkResult.
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512]

    rng = np.random.default_rng(seed)
    results: List[BenchmarkResult] = []

    mechanisms = {
        "vanilla": lambda: VanillaAttention(d_model, num_heads),
        "linear": lambda: LinearAttention(d_model, num_heads),
        "local_w32": lambda: LocalAttention(d_model, window_size=32, num_heads=num_heads),
        "tiled_b32": lambda: TiledAttention(d_model, block_size=32),
    }

    for seq_len in seq_lengths:
        X = rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)

        for name, factory in mechanisms.items():
            module = factory()
            latency_ms, memory = benchmark_single(module, X, n_runs=n_runs)
            throughput = (batch_size * seq_len) / (latency_ms / 1000) if latency_ms > 0 else 0

            results.append(
                BenchmarkResult(
                    attention_type=name,
                    seq_length=seq_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    latency_ms=round(latency_ms, 3),
                    memory_bytes=memory,
                    throughput_tokens_per_sec=round(throughput, 1),
                )
            )

    return results


# ================================================================== #
#  Benchmark KV-Cache                                                #
# ================================================================== #

def benchmark_kv_cache(
    total_tokens: int = 128,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[float]]:
    """Mede crescimento de latência e memória do KV-Cache durante geração.

    Conforme discutido na seção 'KV-Cache e Geração Autoregressiva':
    sem cache a complexidade é O(n³); com cache reduz-se para O(n²·d)
    total. O custo de memória é O(n·L·H·d).

    Args:
        total_tokens: Número de tokens a gerar.
        d_model: Dimensão do modelo.
        num_layers: Número de camadas.
        num_heads: Número de cabeças.
        seed: Seed para reprodutibilidade.

    Returns:
        Dicionário com listas: step, latency_ms, memory_bytes,
        cumulative_latency_ms.
    """
    rng = np.random.default_rng(seed)
    cache = KVCache(d_model=d_model, num_layers=num_layers, num_heads=num_heads)
    attn = VanillaAttention(d_model, num_heads)

    steps: List[int] = []
    latencies: List[float] = []
    memory_sizes: List[int] = []
    cumulative: List[float] = []
    cum_lat = 0.0

    for t in range(total_tokens):
        # Simular geração do token t+1
        new_token = rng.normal(0, 1, (1, 1, d_model)).astype(np.float32)

        start = time.perf_counter()
        # Atualizar cache em cada camada
        for layer in range(num_layers):
            new_k = new_token @ attn.W_k
            new_v = new_token @ attn.W_v
            cached_k, cached_v = cache.update(layer, new_k, new_v)
        end = time.perf_counter()

        lat_ms = (end - start) * 1000
        cum_lat += lat_ms

        steps.append(t)
        latencies.append(round(lat_ms, 4))
        memory_sizes.append(cache.memory_bytes())
        cumulative.append(round(cum_lat, 4))

    return {
        "step": steps,
        "latency_ms": latencies,
        "memory_bytes": memory_sizes,
        "cumulative_latency_ms": cumulative,
    }


# ================================================================== #
#  Benchmark Transformer Block                                       #
# ================================================================== #

def benchmark_transformer_block(
    seq_lengths: Optional[List[int]] = None,
    d_model: int = 128,
    num_heads: int = 4,
    batch_size: int = 2,
    n_runs: int = 3,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Perfila componentes do bloco Transformer.

    Decompõe latência em attention vs FFN conforme a seção
    'Decomposição da Latência em Transformers' (Shoeybi et al., 2019):
    FFN responde por ~60 % dos FLOPs em inferência.

    Args:
        seq_lengths: Comprimentos a testar.
        d_model: Dimensão do modelo.
        num_heads: Cabeças de atenção.
        batch_size: Tamanho do batch.
        n_runs: Repetições.
        seed: Seed.

    Returns:
        Lista de dicts com métricas por seq_length.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256]

    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    for seq_len in seq_lengths:
        X = rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)
        block = SimpleTransformerBlock(d_model, num_heads)

        # Medir total
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            block.forward(X)
            times.append((time.perf_counter() - t0) * 1000)

        # Medir apenas atenção
        attn_times = []
        for _ in range(n_runs):
            normed = block._layer_norm(X)
            t0 = time.perf_counter()
            block.attention.forward(normed)
            attn_times.append((time.perf_counter() - t0) * 1000)

        total_ms = float(np.mean(times))
        attn_ms = float(np.mean(attn_times))

        results.append({
            "seq_length": seq_len,
            "total_ms": round(total_ms, 3),
            "attention_ms": round(attn_ms, 3),
            "ffn_ms": round(total_ms - attn_ms, 3),
            "attention_pct": round(100 * attn_ms / max(total_ms, 1e-6), 1),
        })

    return results


# ================================================================== #
#  Treinamento simplificado (regressão de latência)                  #
# ================================================================== #

def train_latency_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Treina regressão linear para prever latência de atenção.

    Modelo simples para demonstrar a relação entre parâmetros de
    configuração (seq_length, d_model, num_heads) e latência
    observada, conectando com a análise de complexidade da seção
    'Decomposição da Latência em Transformers'.

    Args:
        X_train: Features de treino.
        y_train: Latências de treino.
        X_val: Features de validação.
        y_val: Latências de validação.
        learning_rate: Taxa de aprendizado.
        n_epochs: Épocas de treinamento.
        seed: Seed.

    Returns:
        Dicionário com pesos, histórico de loss, e métricas.
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]

    # Normalização
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    # Inicializar pesos
    W = rng.normal(0, 0.01, n_features).astype(np.float64)
    b = 0.0

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(n_epochs):
        # Forward
        y_pred = X_train_norm @ W + b
        residuals = y_pred - y_train
        loss = float(np.mean(residuals ** 2))
        train_losses.append(loss)

        # Validação
        y_val_pred = X_val_norm @ W + b
        val_loss = float(np.mean((y_val_pred - y_val) ** 2))
        val_losses.append(val_loss)

        # Backward (gradiente descente)
        grad_W = 2 * (X_train_norm.T @ residuals) / len(y_train)
        grad_b = 2 * np.mean(residuals)
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

    return {
        "weights": W,
        "bias": b,
        "X_mean": X_mean,
        "X_std": X_std,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_epochs: int = 50,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Validação cruzada k-fold para o preditor de latência.

    Args:
        X: Features completas.
        y: Alvo completo.
        n_folds: Número de folds.
        n_epochs: Épocas por fold.
        seed: Seed.

    Returns:
        Dicionário com métricas por fold e médias.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    fold_size = len(X) // n_folds

    fold_results: List[float] = []

    for fold in range(n_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate(
            [indices[: fold * fold_size], indices[(fold + 1) * fold_size :]]
        )

        result = train_latency_predictor(
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            n_epochs=n_epochs,
            seed=seed,
        )
        fold_results.append(result["final_val_loss"])

    return {
        "fold_losses": fold_results,
        "mean_loss": float(np.mean(fold_results)),
        "std_loss": float(np.std(fold_results)),
        "n_folds": n_folds,
    }
