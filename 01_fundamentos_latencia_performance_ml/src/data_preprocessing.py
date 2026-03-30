"""
Módulo de pré-processamento de dados para benchmarking de latência em ML.

Implementa o pipeline de pré-processamento conforme discutido na seção
'Fundamentação Teórica' da aula, onde a latência de pré-processamento
(L_pre) é um dos componentes da latência total do sistema.

Referência:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning
    Systems. NeurIPS '15.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Pipeline de pré-processamento com medição de latência em cada etapa.

    Implementa o conceito de decomposição da latência total em componentes
    individuais (L_pre, L_inferencia, L_pos), conforme Crankshaw et al. (2017),
    permitindo identificar gargalos no pipeline de ML.

    Referência:
        Crankshaw, D. et al. (2017). Clipper: A Low-Latency Online Prediction
        Serving System. NSDI '17.

    Attributes:
        scaler: StandardScaler para normalização de features.
        feature_names: Nomes das colunas de features.
        latency_log: Registro de latência de cada operação.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Inicializa o pré-processador.

        Args:
            random_state: Seed para reprodutibilidade.
        """
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.latency_log: list[dict[str, float]] = []
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Geração de dataset sintético
    # ------------------------------------------------------------------
    @staticmethod
    def generate_synthetic_dataset(
        output_path: str = "data/raw/dataset.csv",
        n_samples: int = 10_000,
        n_features: int = 15,
        n_informative: int = 10,
        n_redundant: int = 2,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Gera dataset sintético de classificação binária para benchmarking.

        O dataset é projetado para ter features informativas, redundantes e
        ruidosas, permitindo avaliar como a complexidade dos dados impacta a
        latência de diferentes algoritmos. Conforme Reddi et al. (2020),
        benchmarks reproduzíveis são essenciais para comparações justas.

        Referência:
            Reddi, V. J. et al. (2020). MLPerf Inference Benchmark. ISCA '20.

        Args:
            output_path: Caminho para salvar o CSV.
            n_samples: Número de instâncias.
            n_features: Número total de features.
            n_informative: Número de features informativas.
            n_redundant: Número de features redundantes.
            random_state: Seed para reprodutibilidade.

        Returns:
            DataFrame com o dataset gerado.
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=2,
            weights=[0.5, 0.5],
            flip_y=0.03,
            random_state=random_state,
        )

        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y

        # Garantir que o diretório existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return df

    # ------------------------------------------------------------------
    # Pipeline de pré-processamento
    # ------------------------------------------------------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV com medição de latência.

        Mede o tempo de I/O, que é parte da latência de pré-processamento
        (L_pre). Em produção, leituras de disco podem ser gargalos
        significativos conforme Sculley et al. (2015).

        Args:
            filepath: Caminho do arquivo CSV.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        start = time.perf_counter()
        df = pd.read_csv(filepath)
        elapsed = time.perf_counter() - start

        self.feature_names = [c for c in df.columns if c != "target"]
        self._log_latency("load_data", elapsed)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa dados removendo duplicatas e tratando valores ausentes.

        Implementa etapas de limpeza que contribuem para a latência de
        pré-processamento. A medição granular permite identificar se a
        limpeza é um gargalo no pipeline.

        Args:
            df: DataFrame de entrada.

        Returns:
            DataFrame limpo.
        """
        start = time.perf_counter()

        # Remover duplicatas
        df_clean = df.drop_duplicates()

        # Preencher valores ausentes com a mediana (robusto a outliers)
        for col in self.feature_names:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        elapsed = time.perf_counter() - start
        self._log_latency("clean_data", elapsed)
        return df_clean

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara features aplicando normalização (StandardScaler).

        A normalização é parte da latência de pré-processamento. O StandardScaler
        realiza a transformação z = (x - μ) / σ com complexidade O(n·d),
        onde n = amostras e d = features.

        Args:
            df: DataFrame com features e target.
            fit: Se True, ajusta o scaler; se False, apenas transforma.

        Returns:
            Tupla (X, y) com features normalizadas e target.
        """
        start = time.perf_counter()

        X = df[self.feature_names].values
        y = df["target"].values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        elapsed = time.perf_counter() - start
        self._log_latency("prepare_features", elapsed)
        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """
        Divide dados em treino, validação e teste.

        Conforme boas práticas de benchmarking (Reddi et al., 2020), a
        separação de dados deve ser feita antes de qualquer otimização
        para garantir avaliação justa.

        Args:
            X: Features.
            y: Target.
            test_size: Proporção do conjunto de teste.
            val_size: Proporção do conjunto de validação (relativa ao treino).

        Returns:
            Dicionário com splits X_train, X_val, X_test, y_train, y_val, y_test.
        """
        start = time.perf_counter()

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # val_size relativo ao conjunto temporário
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val,
            random_state=self.random_state,
            stratify=y_temp,
        )

        elapsed = time.perf_counter() - start
        self._log_latency("split_data", elapsed)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def get_latency_report(self) -> pd.DataFrame:
        """
        Retorna relatório de latência de todas as operações registradas.

        Implementa o conceito de decomposição de latência onde
        L_total = Σ L_etapa, permitindo identificar gargalos
        conforme Crankshaw et al. (2017).

        Returns:
            DataFrame com operação, latência (ms) e percentual do total.
        """
        if not self.latency_log:
            return pd.DataFrame(columns=["operation", "latency_ms", "pct_total"])

        df = pd.DataFrame(self.latency_log)
        total = df["latency_ms"].sum()
        df["pct_total"] = (df["latency_ms"] / total * 100).round(2)
        return df

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------
    def _log_latency(self, operation: str, elapsed_seconds: float) -> None:
        """Registra latência de uma operação em milissegundos."""
        self.latency_log.append(
            {"operation": operation, "latency_ms": round(elapsed_seconds * 1000, 4)}
        )
