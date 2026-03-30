"""
Pré-processamento e geração de dados de métricas de serving multimodal.

Gera dataset sintético que simula logs de um sistema de inferência multimodal
em produção, conforme a arquitetura de referência apresentada na seção
"Arquiteturas de Serving" do material da aula (Paleyes et al., 2022).

Referências:
    Paleyes, A., Urma, R.-G., Lawrence, N. D. (2022).
        Challenges in Deploying Machine Learning. ACM Computing Surveys.
    Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Gera e pré-processa dados de métricas de serving multimodal.

    O dataset simula um período de 24 h de requisições em um pipeline
    multimodal (texto, imagem, áudio) com duas versões de modelo
    (v1-stable e v2-canary), permitindo exercícios de SLO, error-budget
    e canary analysis conforme Beyer et al. (2016).
    """

    # ---------------- configuração da simulação ----------------
    MODALITIES = ["text", "image", "audio"]
    MODALITY_WEIGHTS = [0.40, 0.35, 0.25]

    MODEL_VERSIONS = ["v1-stable", "v2-canary"]
    VERSION_WEIGHTS = [0.90, 0.10]  # canary com 10 % do tráfego

    INSTANCE_TYPES = ["cpu", "gpu"]

    ERROR_TYPES = ["timeout", "oom", "model_error"]

    # Parâmetros de latência por modalidade (ms) — média e desvio
    LATENCY_PARAMS: dict[str, dict[str, Tuple[float, float]]] = {
        "text":  {"preprocessing": (10, 3), "inference": (35, 10), "postprocessing": (5, 2)},
        "image": {"preprocessing": (15, 5), "inference": (60, 20), "postprocessing": (8, 3)},
        "audio": {"preprocessing": (20, 7), "inference": (80, 25), "postprocessing": (10, 4)},
    }

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    #  Geração de dataset sintético                                       #
    # ------------------------------------------------------------------ #

    def generate_dataset(self, n_samples: int = 10_000) -> pd.DataFrame:
        """Gera dataset sintético de métricas de serving.

        Simula requisições a um sistema multimodal com latências
        diferenciadas por modalidade, conforme discutido na seção
        'Orquestração de Pipelines Multimodais' do material da aula.

        Args:
            n_samples: Número de requisições a gerar.

        Returns:
            DataFrame com métricas de serving.
        """
        rng = self.rng

        # Modalidades e versões
        modalities = rng.choice(
            self.MODALITIES, size=n_samples, p=self.MODALITY_WEIGHTS
        )
        model_versions = rng.choice(
            self.MODEL_VERSIONS, size=n_samples, p=self.VERSION_WEIGHTS
        )

        # Tipo de instância: imagem e áudio preferencialmente em GPU
        instance_types = np.where(
            np.isin(modalities, ["image", "audio"]),
            rng.choice(self.INSTANCE_TYPES, size=n_samples, p=[0.2, 0.8]),
            rng.choice(self.INSTANCE_TYPES, size=n_samples, p=[0.7, 0.3]),
        )

        # Latências por etapa
        preprocessing_ms = np.zeros(n_samples)
        inference_ms = np.zeros(n_samples)
        postprocessing_ms = np.zeros(n_samples)

        for mod in self.MODALITIES:
            mask = modalities == mod
            count = mask.sum()
            params = self.LATENCY_PARAMS[mod]
            preprocessing_ms[mask] = np.abs(
                rng.normal(params["preprocessing"][0], params["preprocessing"][1], count)
            )
            inference_ms[mask] = np.abs(
                rng.normal(params["inference"][0], params["inference"][1], count)
            )
            postprocessing_ms[mask] = np.abs(
                rng.normal(params["postprocessing"][0], params["postprocessing"][1], count)
            )

        # Canary v2 ligeiramente mais lenta (simula regressão para análise)
        canary_mask = model_versions == "v2-canary"
        inference_ms[canary_mask] *= rng.normal(1.08, 0.02, canary_mask.sum())

        # Queue wait — modelado via distribuição exponencial (teoria de filas)
        queue_wait_ms = rng.exponential(scale=5.0, size=n_samples)

        total_latency_ms = preprocessing_ms + inference_ms + postprocessing_ms + queue_wait_ms

        # Confiança da predição
        confidence = np.clip(rng.beta(8, 2, size=n_samples), 0.0, 1.0)

        # Erros (~2 % global, canary um pouco mais)
        base_error_rate = 0.018
        canary_error_rate = 0.035
        error_probs = np.where(canary_mask, canary_error_rate, base_error_rate)
        is_error = rng.binomial(1, error_probs)

        error_type = np.where(
            is_error == 1,
            rng.choice(self.ERROR_TYPES, size=n_samples, p=[0.5, 0.2, 0.3]),
            "none",
        )

        # Utilização de recursos — correlacionada com carga
        cpu_utilization = np.clip(rng.normal(0.55, 0.15, n_samples), 0.05, 0.99)
        memory_utilization = np.clip(rng.normal(0.50, 0.12, n_samples), 0.05, 0.99)

        # Réplicas: varia ao longo do dia (simula auto-scaling)
        hour_of_day = np.linspace(0, 24, n_samples, endpoint=False)
        base_replicas = 3 + 4 * np.sin(np.pi * hour_of_day / 12)  # pico ao meio-dia
        replica_count = np.clip(np.round(base_replicas + rng.normal(0, 0.5, n_samples)), 1, 10).astype(int)

        # Timestamps
        start = pd.Timestamp("2026-03-29 00:00:00")
        timestamps = pd.date_range(start, periods=n_samples, freq="8640ms")  # ~24 h

        # Request IDs
        request_ids = [uuid.uuid4().hex[:32] for _ in range(n_samples)]

        df = pd.DataFrame({
            "request_id": request_ids,
            "timestamp": timestamps[:n_samples],
            "modality": modalities,
            "model_version": model_versions,
            "instance_type": instance_types,
            "preprocessing_ms": np.round(preprocessing_ms, 2),
            "inference_ms": np.round(inference_ms, 2),
            "postprocessing_ms": np.round(postprocessing_ms, 2),
            "total_latency_ms": np.round(total_latency_ms, 2),
            "queue_wait_ms": np.round(queue_wait_ms, 2),
            "confidence": np.round(confidence, 4),
            "is_error": is_error,
            "error_type": error_type,
            "cpu_utilization": np.round(cpu_utilization, 4),
            "memory_utilization": np.round(memory_utilization, 4),
            "replica_count": replica_count,
        })
        return df

    def generate_and_save(
        self,
        path: str = "data/raw/dataset.csv",
        n_samples: int = 10_000,
    ) -> pd.DataFrame:
        """Gera e persiste dataset sintético em CSV.

        Args:
            path: Caminho de saída.
            n_samples: Quantidade de amostras.

        Returns:
            DataFrame gerado.
        """
        df = self.generate_dataset(n_samples)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Dataset salvo em {path} — {len(df)} amostras.")
        return df

    # ------------------------------------------------------------------ #
    #  Carregamento e limpeza                                             #
    # ------------------------------------------------------------------ #

    def load_data(self, path: str = "data/raw/dataset.csv") -> pd.DataFrame:
        """Carrega CSV de métricas de serving.

        Args:
            path: Caminho do arquivo CSV.

        Returns:
            DataFrame carregado.
        """
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados: remove duplicatas e valores nulos.

        Args:
            df: DataFrame bruto.

        Returns:
            DataFrame limpo.
        """
        df = df.drop_duplicates(subset=["request_id"])
        df = df.dropna(subset=["total_latency_ms", "modality", "model_version"])
        # Latências negativas são artefatos — corrigir para 0
        latency_cols = [
            "preprocessing_ms", "inference_ms",
            "postprocessing_ms", "total_latency_ms", "queue_wait_ms",
        ]
        for col in latency_cols:
            df[col] = df[col].clip(lower=0.0)
        return df.reset_index(drop=True)

    def prepare_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Cria features derivadas para análise.

        Adiciona colunas auxiliares usadas nos notebooks de exploração
        e avaliação: hora do dia, flag de SLO violado, etc.

        Args:
            df: DataFrame limpo.

        Returns:
            DataFrame enriquecido.
        """
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["slo_violated"] = (df["total_latency_ms"] > 200).astype(int)
        df["modality_cat"] = df["modality"].astype("category")
        df["version_cat"] = df["model_version"].astype("category")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino e teste por corte temporal.

        Usa split cronológico (não aleatório) para preservar a
        dinâmica temporal do sistema, conforme boas práticas de
        avaliação de sistemas em produção (Burkov, 2020).

        Args:
            df: DataFrame com coluna ``timestamp``.
            test_size: Fração para teste (corte final).
            random_state: Não utilizado (split cronológico).

        Returns:
            Tupla (df_train, df_test).
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
