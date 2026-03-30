"""
Testes unitários para src/data_preprocessing.py.

Valida a geração de dataset sintético, limpeza e preparação de
features do DataPreprocessor.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


# ---- Fixtures --------------------------------------------------------


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """DataPreprocessor com seed fixa."""
    return DataPreprocessor(seed=42)


@pytest.fixture
def sample_df(preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Dataset pequeno (500 amostras) para testes rápidos."""
    return preprocessor.generate_dataset(n_samples=500)


# ---- Testes da geração -----------------------------------------------


class TestGenerateDataset:
    def test_shape(self, preprocessor: DataPreprocessor) -> None:
        df = preprocessor.generate_dataset(n_samples=100)
        assert df.shape[0] == 100
        assert df.shape[1] == 16

    def test_columns_present(self, sample_df: pd.DataFrame) -> None:
        expected = {
            "request_id", "timestamp", "modality", "model_version",
            "instance_type", "preprocessing_ms", "inference_ms",
            "postprocessing_ms", "total_latency_ms", "queue_wait_ms",
            "confidence", "is_error", "error_type",
            "cpu_utilization", "memory_utilization", "replica_count",
        }
        assert set(sample_df.columns) == expected

    def test_modality_values(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["modality"].unique()).issubset({"text", "image", "audio"})

    def test_model_version_values(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["model_version"].unique()).issubset({"v1-stable", "v2-canary"})

    def test_latency_positive(self, sample_df: pd.DataFrame) -> None:
        for col in ["preprocessing_ms", "inference_ms", "postprocessing_ms", "total_latency_ms"]:
            assert (sample_df[col] >= 0).all(), f"{col} contém valores negativos"

    def test_reproducibility(self, preprocessor: DataPreprocessor) -> None:
        """Seed fixa deve produzir resultados numéricos idênticos."""
        df1 = DataPreprocessor(seed=99).generate_dataset(100)
        df2 = DataPreprocessor(seed=99).generate_dataset(100)
        # request_id usa uuid4 (não reprodutível) — excluir da comparação
        cols = [c for c in df1.columns if c != "request_id"]
        pd.testing.assert_frame_equal(df1[cols], df2[cols])


# ---- Testes de limpeza -----------------------------------------------


class TestCleanData:
    def test_no_nulls_in_key_columns(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        df_clean = preprocessor.clean_data(sample_df)
        for col in ["total_latency_ms", "modality", "model_version"]:
            assert df_clean[col].notna().all()

    def test_no_negative_latency(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        df_clean = preprocessor.clean_data(sample_df)
        assert (df_clean["total_latency_ms"] >= 0).all()


# ---- Testes de features ----------------------------------------------


class TestPrepareFeatures:
    def test_hour_column(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        df = preprocessor.prepare_features(sample_df)
        assert "hour" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_slo_violated_column(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        df = preprocessor.prepare_features(sample_df)
        assert "slo_violated" in df.columns
        assert set(df["slo_violated"].unique()).issubset({0, 1})


# ---- Testes de split --------------------------------------------------


class TestSplitData:
    def test_split_sizes(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        train, test = preprocessor.split_data(sample_df, test_size=0.2)
        assert len(train) + len(test) == len(sample_df)
        assert abs(len(test) / len(sample_df) - 0.2) < 0.02

    def test_chronological_order(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        train, test = preprocessor.split_data(sample_df, test_size=0.2)
        assert train["timestamp"].max() <= test["timestamp"].min()
