"""
Métricas de avaliação de performance para mecanismos de atenção e áudio.

Implementa funções de avaliação conectadas à seção 'Tabela Comparativa:
Abordagens de Atenção Eficiente' e 'Decomposição da Latência em
Transformers' do material da Aula 02.

Referências:
    Tay, Y. et al. (2022). Efficient Transformers: A Survey.
        ACM Computing Surveys, v. 55, n. 6.
    Dao, T. et al. (2022). FlashAttention. NeurIPS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


# ================================================================== #
#  Métricas numéricas                                                #
# ================================================================== #

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calcula métricas de regressão para avaliação de preditor de latência.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.

    Returns:
        Dicionário com MSE, RMSE, MAE, MAPE, R².
    """
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y_true), 1e-8)) * 100)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def attention_memory_theoretical(
    seq_length: int,
    d_model: int,
    num_heads: int,
    attention_type: str = "vanilla",
    batch_size: int = 1,
) -> int:
    """Calcula memória teórica para a matriz de atenção.

    Baseado na análise de complexidade da seção 'Fundamentos do
    Mecanismo de Self-Attention':
    - Vanilla: O(n²) → batch × heads × n × n × 4 bytes
    - Linear:  O(n·d) → batch × heads × n × d_k × 4 bytes
    - Flash:   O(n)   → batch × n × d × 4 bytes (sem matriz n×n)

    Conforme tabela comparativa de Tay et al. (2022).

    Args:
        seq_length: Comprimento da sequência.
        d_model: Dimensão do modelo.
        num_heads: Número de cabeças.
        attention_type: Tipo de atenção.
        batch_size: Tamanho do batch.

    Returns:
        Memória estimada em bytes.
    """
    d_k = d_model // num_heads
    bytes_per_elem = 4  # float32

    if attention_type == "vanilla":
        return batch_size * num_heads * seq_length * seq_length * bytes_per_elem
    elif attention_type == "linear":
        return batch_size * num_heads * seq_length * d_k * bytes_per_elem
    elif attention_type in ("flash", "tiled"):
        return batch_size * seq_length * d_model * bytes_per_elem
    elif attention_type == "sparse":
        # Sparse Transformer: O(n√n) (Child et al., 2019)
        sparsity = int(np.sqrt(seq_length))
        return batch_size * num_heads * seq_length * sparsity * bytes_per_elem
    elif attention_type == "local":
        window = min(64, seq_length)
        return batch_size * num_heads * seq_length * window * bytes_per_elem
    else:
        raise ValueError(f"Tipo de atenção desconhecido: {attention_type}")


def kv_cache_memory(
    seq_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dtype_bytes: int = 4,
) -> int:
    """Calcula memória do KV-Cache.

    Conforme seção 'KV-Cache e Geração Autoregressiva':
    Custo = O(n · L · H · d_k) × 2 (K e V) × bytes_per_elem

    Para LLaMA-2 70B com seq 4096 tokens, o cache ocupa ~40 GB em FP16
    (Touvron et al., 2023).

    Args:
        seq_length: Comprimento da sequência em cache.
        d_model: Dimensão do modelo.
        num_layers: Número de camadas.
        num_heads: Número de cabeças.
        dtype_bytes: Bytes por elemento (4=FP32, 2=FP16).

    Returns:
        Memória em bytes.
    """
    d_k = d_model // num_heads
    # 2 tensores (K e V) × layers × heads × seq × d_k
    return 2 * num_layers * num_heads * seq_length * d_k * dtype_bytes


# ================================================================== #
#  Classe agregadora                                                 #
# ================================================================== #

class PerformanceEvaluator:
    """Avalia e visualiza performance de mecanismos de atenção.

    Conecta-se à seção 'Tabela Comparativa: Abordagens de Atenção
    Eficiente' e gera visualizações para as métricas discutidas.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # ---- Tabela de resultados ---------------------------------------- #

    @staticmethod
    def results_to_dataframe(
        results: List[Any],
    ) -> pd.DataFrame:
        """Converte lista de BenchmarkResult em DataFrame.

        Args:
            results: Lista de BenchmarkResult (de training.py).

        Returns:
            DataFrame com colunas de métricas.
        """
        records = []
        for r in results:
            records.append({
                "attention_type": r.attention_type,
                "seq_length": r.seq_length,
                "d_model": r.d_model,
                "num_heads": r.num_heads,
                "latency_ms": r.latency_ms,
                "memory_mb": r.memory_bytes / (1024 ** 2),
                "throughput_tok_s": r.throughput_tokens_per_sec,
            })
        return pd.DataFrame(records)

    # ---- Visualizações ----------------------------------------------- #

    def plot_scaling_comparison(
        self,
        results: List[Any],
        save: bool = True,
    ) -> plt.Figure:
        """Gráfico de latência vs comprimento de sequência por tipo de atenção.

        Reproduz visualmente a Tabela Comparativa da seção Saiba Mais,
        evidenciando o crescimento quadrático da atenção vanilla vs.
        linear/local.

        Args:
            results: Lista de BenchmarkResult.
            save: Se True, salva em outputs/figures/.

        Returns:
            Objeto Figure do matplotlib.
        """
        df = self.results_to_dataframe(results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1) Latência
        sns.lineplot(data=df, x="seq_length", y="latency_ms",
                     hue="attention_type", marker="o", ax=axes[0])
        axes[0].set_title("Latência vs Comprimento de Sequência")
        axes[0].set_xlabel("Comprimento da Sequência (n)")
        axes[0].set_ylabel("Latência (ms)")
        axes[0].set_yscale("log")
        axes[0].legend(title="Tipo")

        # 2) Memória
        sns.lineplot(data=df, x="seq_length", y="memory_mb",
                     hue="attention_type", marker="s", ax=axes[1])
        axes[1].set_title("Memória vs Comprimento de Sequência")
        axes[1].set_xlabel("Comprimento da Sequência (n)")
        axes[1].set_ylabel("Memória (MB)")
        axes[1].set_yscale("log")
        axes[1].legend(title="Tipo")

        # 3) Throughput
        sns.lineplot(data=df, x="seq_length", y="throughput_tok_s",
                     hue="attention_type", marker="^", ax=axes[2])
        axes[2].set_title("Throughput vs Comprimento de Sequência")
        axes[2].set_xlabel("Comprimento da Sequência (n)")
        axes[2].set_ylabel("Tokens / segundo")
        axes[2].legend(title="Tipo")

        fig.suptitle(
            "Comparativo de Mecanismos de Atenção\n"
            "(Tay et al., 2022 — Efficient Transformers: A Survey)",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "scaling_comparison.png",
                        dpi=150, bbox_inches="tight")
        return fig

    def plot_memory_theoretical(
        self,
        seq_lengths: Optional[List[int]] = None,
        d_model: int = 128,
        num_heads: int = 4,
        save: bool = True,
    ) -> plt.Figure:
        """Gráfico de memória teórica para diferentes abordagens.

        Visualiza complexidade de memória da tabela comparativa:
        Vanilla O(n²), Flash O(n), Linear O(n·d), etc.

        Args:
            seq_lengths: Comprimentos a plotar.
            d_model: Dimensão.
            num_heads: Cabeças.
            save: Salvar figura.

        Returns:
            Figura matplotlib.
        """
        if seq_lengths is None:
            seq_lengths = [32, 64, 128, 256, 512, 1024, 2048]

        types = ["vanilla", "linear", "flash", "sparse", "local"]
        data: Dict[str, List[float]] = {t: [] for t in types}

        for n in seq_lengths:
            for t in types:
                mem = attention_memory_theoretical(n, d_model, num_heads, t)
                data[t].append(mem / 1024)  # KB

        fig, ax = plt.subplots(figsize=(10, 6))
        for t in types:
            ax.plot(seq_lengths, data[t], marker="o", label=t.title())

        ax.set_xlabel("Comprimento da Sequência (n)")
        ax.set_ylabel("Memória (KB)")
        ax.set_title(
            "Memória Teórica da Matriz de Atenção por Abordagem\n"
            "(Complexidade: Vanilla O(n²), Flash O(n), Linear O(n·d))"
        )
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "memory_theoretical.png",
                        dpi=150, bbox_inches="tight")
        return fig

    def plot_kv_cache_growth(
        self,
        cache_data: Dict[str, List[float]],
        save: bool = True,
    ) -> plt.Figure:
        """Visualiza crescimento do KV-Cache durante geração autoregressiva.

        Conecta-se à seção 'KV-Cache e Geração Autoregressiva' que
        descreve o custo de memória O(n·L·H·d).

        Args:
            cache_data: Saída de benchmark_kv_cache().
            save: Salvar figura.

        Returns:
            Figura matplotlib.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Memória do cache
        mem_kb = [m / 1024 for m in cache_data["memory_bytes"]]
        axes[0].plot(cache_data["step"], mem_kb, color="steelblue")
        axes[0].set_xlabel("Token Gerado (t)")
        axes[0].set_ylabel("Memória do KV-Cache (KB)")
        axes[0].set_title("Crescimento Linear da Memória do KV-Cache")
        axes[0].grid(True, alpha=0.3)

        # Latência por token
        axes[1].plot(cache_data["step"], cache_data["latency_ms"],
                     color="coral", alpha=0.7)
        axes[1].set_xlabel("Token Gerado (t)")
        axes[1].set_ylabel("Latência por Token (ms)")
        axes[1].set_title("Latência de Atualização do KV-Cache")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            "KV-Cache: Trade-off Memória vs Computação\n"
            "(Shazeer, 2019; Touvron et al., 2023)",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "kv_cache_growth.png",
                        dpi=150, bbox_inches="tight")
        return fig

    def plot_training_loss(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save: bool = True,
    ) -> plt.Figure:
        """Curvas de loss do preditor de latência.

        Args:
            train_losses: Lista de training loss por época.
            val_losses: Lista de validation loss por época.
            save: Salvar figura.

        Returns:
            Figura matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label="Treino", color="steelblue")
        ax.plot(val_losses, label="Validação", color="coral")
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Curvas de Loss — Preditor de Latência de Atenção")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "training_loss.png",
                        dpi=150, bbox_inches="tight")
        return fig

    def plot_attention_heatmap(
        self,
        attn_weights: np.ndarray,
        title: str = "Attention Weights",
        save: bool = True,
    ) -> plt.Figure:
        """Heatmap de pesos de atenção.

        Visualiza a matriz n×n de atenção discutida na seção
        'Fundamentos do Mecanismo de Self-Attention' (Vaswani et al., 2017).

        Args:
            attn_weights: Matriz de atenção (n, n) ou (heads, n, n).
            title: Título do gráfico.
            save: Salvar figura.

        Returns:
            Figura matplotlib.
        """
        if attn_weights.ndim == 3:
            # Média sobre cabeças
            attn_weights = attn_weights.mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attn_weights, cmap="viridis", ax=ax, cbar_kws={"label": "Peso"})
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "attention_heatmap.png",
                        dpi=150, bbox_inches="tight")
        return fig

    def plot_mel_spectrogram(
        self,
        mel_db: np.ndarray,
        sample_rate: int = 16000,
        hop_length: int = 160,
        title: str = "Espectrograma Mel",
        save: bool = True,
    ) -> plt.Figure:
        """Visualiza espectrograma Mel.

        Conecta-se à seção 'Processamento Digital de Sinais para Áudio':
        espectrogramas Mel com escala perceptualmente motivada
        m = 2595 · log10(1 + f/700) (Radford et al., 2023).

        Args:
            mel_db: Espectrograma Mel em dB (n_mels, n_frames).
            sample_rate: Taxa de amostragem.
            hop_length: Hop length.
            title: Título.
            save: Salvar.

        Returns:
            Figura.
        """
        n_mels, n_frames = mel_db.shape
        duration = n_frames * hop_length / sample_rate

        fig, ax = plt.subplots(figsize=(12, 4))
        img = ax.imshow(
            mel_db, aspect="auto", origin="lower",
            extent=[0, duration, 0, n_mels],
            cmap="magma",
        )
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Filtro Mel")
        ax.set_title(f"{title}\n(n_mels={n_mels}, hop={hop_length}, sr={sample_rate})")
        plt.colorbar(img, ax=ax, label="dB")
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "mel_spectrogram.png",
                        dpi=150, bbox_inches="tight")
        return fig
