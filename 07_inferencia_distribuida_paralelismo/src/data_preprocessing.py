"""
Módulo de geração e pré-processamento de workloads de inferência LLM.

Gera datasets sintéticos que simulam requisições de inferência para Large
Language Models, com distribuições realistas de comprimento de prompt e
geração, conforme padrões observados em Kwon et al. (2023) e Yu et al. (2022).

Implementa conceitos da seção 'Continuous Batching e Scheduling Dinâmico'
do material da aula, onde requisições chegam com comprimentos variáveis
e o scheduler deve otimizar utilização de recursos.

Referências:
    Kwon, W. et al. (2023). Efficient Memory Management for Large Language
        Model Serving with PagedAttention. SOSP 2023.
    Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuração de modelos simulados
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "7B": {"num_attention_heads": 32, "hidden_dim": 4096},
    "13B": {"num_attention_heads": 40, "hidden_dim": 5120},
    "70B": {"num_attention_heads": 64, "hidden_dim": 8192},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """Representa uma requisição de inferência para um LLM.

    Modela as propriedades relevantes para scheduling e gerenciamento
    de memória, conforme descrito na seção 'Continuous Batching' do
    material da aula (Yu et al., 2022).
    """

    request_id: int
    arrival_time_ms: float
    prompt_tokens: int
    max_gen_tokens: int
    actual_gen_tokens: int
    model_size_b: float
    num_attention_heads: int
    hidden_dim: int


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class WorkloadPreprocessor:
    """
    Gera e pré-processa workloads sintéticos de inferência LLM.

    Implementa geração de tráfego baseada em processo de Poisson
    com distribuições lognormais para comprimentos de sequência,
    conforme padrões do dataset ShareGPT reportados por Kwon et al. (2023).

    Attributes:
        seed: Semente para reprodutibilidade.
        rng: Gerador de números aleatórios NumPy.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Geração de workload
    # ------------------------------------------------------------------

    def generate_workload(
        self,
        n_requests: int = 5000,
        arrival_rate: float = 50.0,
        model_mix: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Gera workload sintético de requisições de inferência LLM.

        Implementa geração de tráfego com processo de Poisson para
        intervalos entre chegadas e distribuição lognormal para
        comprimentos de prompt, conforme descrito na seção
        'Continuous Batching e Scheduling Dinâmico' do material da aula.

        Args:
            n_requests: Número de requisições a gerar.
            arrival_rate: Taxa média de chegadas (req/s).
            model_mix: Proporção de cada modelo. Default: 7B=50%, 13B=35%, 70B=15%.

        Returns:
            DataFrame com colunas: request_id, arrival_time_ms, prompt_tokens,
            max_gen_tokens, actual_gen_tokens, model_size_b,
            num_attention_heads, hidden_dim.
        """
        if model_mix is None:
            model_mix = {"7B": 0.50, "13B": 0.35, "70B": 0.15}

        # Intervalos entre chegadas — processo de Poisson
        inter_arrivals = self.rng.exponential(
            scale=1000.0 / arrival_rate, size=n_requests
        )
        arrival_times = np.cumsum(inter_arrivals)

        # Comprimentos de prompt — lognormal (Kwon et al., 2023: ShareGPT)
        prompt_tokens = np.clip(
            self.rng.lognormal(mean=5.5, sigma=1.0, size=n_requests).astype(int),
            a_min=10,
            a_max=2048,
        )

        # Tokens máximos de geração
        max_gen_tokens = np.clip(
            self.rng.lognormal(mean=4.5, sigma=1.0, size=n_requests).astype(int),
            a_min=16,
            a_max=1024,
        )

        # Tokens efetivamente gerados (≤ max)
        actual_gen_tokens = np.minimum(
            np.clip(
                self.rng.lognormal(mean=4.0, sigma=1.2, size=n_requests).astype(int),
                a_min=5,
                a_max=512,
            ),
            max_gen_tokens,
        )

        # Seleção aleatória de modelo
        model_names = list(model_mix.keys())
        model_probs = list(model_mix.values())
        selected_models = self.rng.choice(model_names, size=n_requests, p=model_probs)

        model_sizes = np.array(
            [float(m.replace("B", "")) for m in selected_models]
        )
        attention_heads = np.array(
            [MODEL_CONFIGS[m]["num_attention_heads"] for m in selected_models]
        )
        hidden_dims = np.array(
            [MODEL_CONFIGS[m]["hidden_dim"] for m in selected_models]
        )

        df = pd.DataFrame(
            {
                "request_id": np.arange(n_requests),
                "arrival_time_ms": np.round(arrival_times, 2),
                "prompt_tokens": prompt_tokens,
                "max_gen_tokens": max_gen_tokens,
                "actual_gen_tokens": actual_gen_tokens,
                "model_size_b": model_sizes,
                "num_attention_heads": attention_heads,
                "hidden_dim": hidden_dims,
            }
        )
        return df

    # ------------------------------------------------------------------
    # Pré-processamento e particionamento
    # ------------------------------------------------------------------

    def compute_kv_cache_memory(
        self,
        df: pd.DataFrame,
        block_size: int = 16,
        dtype_bytes: int = 2,
        num_layers_map: Optional[dict[float, int]] = None,
    ) -> pd.DataFrame:
        """Calcula uso de memória KV-Cache por requisição.

        Implementa o cálculo de PagedAttention (Kwon et al., 2023)
        conforme descrito na seção 'PagedAttention e Gerenciamento de
        KV-Cache'. Para cada requisição, estima o número de blocos
        necessários: ceil(n / B) onde n = prompt + geração, B = block_size.

        Args:
            df: DataFrame de workload.
            block_size: Tamanho de bloco do PagedAttention (padrão: 16 tokens).
            dtype_bytes: Bytes por elemento (2 = FP16).
            num_layers_map: Mapeamento model_size -> num_layers.

        Returns:
            DataFrame com colunas adicionais: total_tokens, num_blocks,
            kv_cache_bytes_per_layer, total_kv_cache_mb.
        """
        if num_layers_map is None:
            num_layers_map = {7.0: 32, 13.0: 40, 70.0: 80}

        result = df.copy()

        result["total_tokens"] = result["prompt_tokens"] + result["actual_gen_tokens"]

        # Blocos PagedAttention: ceil(total_tokens / block_size)
        result["num_blocks"] = np.ceil(
            result["total_tokens"] / block_size
        ).astype(int)

        # KV-Cache por camada = 2 (K + V) × seq_len × hidden_dim × dtype_bytes
        result["kv_cache_bytes_per_layer"] = (
            2 * result["total_tokens"] * result["hidden_dim"] * dtype_bytes
        )

        # Total KV-Cache = por camada × num_layers
        result["num_layers"] = result["model_size_b"].map(num_layers_map)
        result["total_kv_cache_mb"] = (
            result["kv_cache_bytes_per_layer"] * result["num_layers"] / (1024**2)
        )

        return result

    def split_by_model(
        self, df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Particiona workload por tamanho de modelo.

        Útil para análise de cada classe de modelo separadamente,
        conforme a abordagem de routing discutida na seção
        'Arquitetura de Sistemas de Serving Distribuído'.

        Args:
            df: DataFrame de workload.

        Returns:
            Dicionário {model_name: DataFrame}.
        """
        splits: dict[str, pd.DataFrame] = {}
        for model_size in df["model_size_b"].unique():
            label = f"{int(model_size)}B"
            splits[label] = df[df["model_size_b"] == model_size].reset_index(drop=True)
        return splits

    def prepare_batching_windows(
        self,
        df: pd.DataFrame,
        window_ms: float = 100.0,
    ) -> list[pd.DataFrame]:
        """Segmenta requisições em janelas temporais para análise de batching.

        Simula a chegada de requisições em intervalos fixos, conforme
        utilizado por schedulers de continuous batching
        (Yu et al., 2022).

        Args:
            df: DataFrame de workload ordenado por arrival_time_ms.
            window_ms: Duração de cada janela em milissegundos.

        Returns:
            Lista de DataFrames, um por janela temporal.
        """
        df_sorted = df.sort_values("arrival_time_ms").reset_index(drop=True)
        max_time = df_sorted["arrival_time_ms"].max()

        windows: list[pd.DataFrame] = []
        t = 0.0
        while t < max_time:
            window = df_sorted[
                (df_sorted["arrival_time_ms"] >= t)
                & (df_sorted["arrival_time_ms"] < t + window_ms)
            ]
            if len(window) > 0:
                windows.append(window.reset_index(drop=True))
            t += window_ms

        return windows

    # ------------------------------------------------------------------
    # Geração estática (classmethod)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_workload(
        output_path: str = "data/raw/dataset.csv",
        n_requests: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Gera e salva workload sintético em CSV.

        Função de conveniência que cria o diretório de saída,
        gera o workload e salva em disco.

        Args:
            output_path: Caminho para salvar o CSV.
            n_requests: Número de requisições.
            seed: Semente aleatória.

        Returns:
            DataFrame gerado.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        preprocessor = WorkloadPreprocessor(seed=seed)
        df = preprocessor.generate_workload(n_requests=n_requests)
        df.to_csv(output_path, index=False)
        print(f"Workload sintético salvo em {output_path} ({len(df)} requisições)")
        return df

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    @staticmethod
    def load_data(path: str = "data/raw/dataset.csv") -> pd.DataFrame:
        """Carrega workload de um arquivo CSV.

        Args:
            path: Caminho do CSV.

        Returns:
            DataFrame com o workload.
        """
        return pd.read_csv(path)
