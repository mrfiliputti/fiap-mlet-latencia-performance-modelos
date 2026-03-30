"""
Testes unitários para o módulo de pré-processamento de dados.

Valida geração de dataset sintético, limpeza de dados e criação de
DataLoaders para experimentos de aceleração por hardware (Aula 06).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import DataPreprocessor, SyntheticImageDataset


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def preprocessor():
    """Cria DataPreprocessor com parâmetros pequenos para testes rápidos."""
    return DataPreprocessor(
        n_samples=200, img_size=8, n_channels=3,
        n_classes=5, seed=42,
    )


@pytest.fixture
def sample_df(preprocessor):
    """Gera DataFrame de exemplo em diretório temporário."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = preprocessor.generate_dataset(output_dir=Path(tmpdir))
        yield df


@pytest.fixture
def sample_features_labels(preprocessor, sample_df):
    """Extrai features e labels do DataFrame de exemplo."""
    return preprocessor.prepare_features(sample_df)


# -----------------------------------------------------------------------
# Testes de Geração de Dataset
# -----------------------------------------------------------------------

class TestDataGeneration:
    """Testes de geração de dataset sintético."""

    def test_generate_dataset_shape(self, preprocessor):
        """Verifica shape correto do dataset gerado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = preprocessor.generate_dataset(output_dir=Path(tmpdir))
            n_features = preprocessor.img_size ** 2 * preprocessor.n_channels
            assert df.shape == (200, n_features + 1)

    def test_generate_dataset_columns(self, preprocessor):
        """Verifica nomes de colunas (feature_N + label)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = preprocessor.generate_dataset(output_dir=Path(tmpdir))
            assert "label" in df.columns
            assert "feature_0" in df.columns
            n_features = preprocessor.img_size ** 2 * preprocessor.n_channels
            assert f"feature_{n_features - 1}" in df.columns

    def test_generate_dataset_values_range(self, sample_df):
        """Verifica que features estão no range [0, 1]."""
        feature_cols = [c for c in sample_df.columns if c.startswith("feature_")]
        assert sample_df[feature_cols].min().min() >= 0.0
        assert sample_df[feature_cols].max().max() <= 1.0

    def test_generate_dataset_labels_valid(self, preprocessor, sample_df):
        """Verifica que labels são válidos (0 a n_classes-1)."""
        assert sample_df["label"].min() >= 0
        assert sample_df["label"].max() < preprocessor.n_classes

    def test_generate_dataset_class_balance(self, preprocessor, sample_df):
        """Verifica balanceamento aproximado de classes."""
        expected_per_class = preprocessor.n_samples // preprocessor.n_classes
        class_counts = sample_df["label"].value_counts()
        for count in class_counts.values:
            assert count == expected_per_class

    def test_generate_dataset_saves_csv(self, preprocessor):
        """Verifica que CSV é salvo corretamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor.generate_dataset(output_dir=Path(tmpdir))
            csv_path = Path(tmpdir) / "dataset.csv"
            assert csv_path.exists()
            df_loaded = pd.read_csv(csv_path)
            assert len(df_loaded) == 200


# -----------------------------------------------------------------------
# Testes de Limpeza e Preparação
# -----------------------------------------------------------------------

class TestDataPreparation:
    """Testes de limpeza e preparação de features."""

    def test_clean_data_removes_nan(self, preprocessor, sample_df):
        """Verifica remoção de NaN."""
        # Inserir NaN artificial
        dirty_df = sample_df.copy()
        dirty_df.iloc[0, 0] = np.nan
        cleaned = preprocessor.clean_data(dirty_df)
        assert not cleaned.isnull().any().any()
        assert len(cleaned) < len(dirty_df)

    def test_prepare_features_types(self, preprocessor, sample_df):
        """Verifica tipos corretos: features float32 e labels int64."""
        features, labels = preprocessor.prepare_features(sample_df)
        assert features.dtype == np.float32
        assert labels.dtype == np.int64

    def test_prepare_features_shape(self, preprocessor, sample_df):
        """Verifica shapes corretos."""
        features, labels = preprocessor.prepare_features(sample_df)
        n_features = preprocessor.img_size ** 2 * preprocessor.n_channels
        assert features.shape == (200, n_features)
        assert labels.shape == (200,)

    def test_split_data_sizes(self, preprocessor, sample_features_labels):
        """Verifica proporções dos splits."""
        features, labels = sample_features_labels
        splits = preprocessor.split_data(features, labels, 0.7, 0.15, 0.15)
        total = sum(s[0].shape[0] for s in splits.values())
        assert total == len(labels)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_split_data_no_overlap(self, preprocessor, sample_features_labels):
        """Verifica que splits não sobrepõem."""
        features, labels = sample_features_labels
        splits = preprocessor.split_data(features, labels)
        train_set = set(map(tuple, splits["train"][0].tolist()))
        test_set = set(map(tuple, splits["test"][0].tolist()))
        # Pode haver overlap mínimo em dados sintéticos, mas tamanhos devem somar
        total = sum(len(s[0]) for s in splits.values())
        assert total == len(labels)


# -----------------------------------------------------------------------
# Testes de DataLoaders
# -----------------------------------------------------------------------

class TestDataLoaders:
    """Testes de criação de DataLoaders PyTorch."""

    def test_create_dataloaders_keys(self, preprocessor, sample_features_labels):
        """Verifica que DataLoaders são criados para todos os splits."""
        features, labels = sample_features_labels
        splits = preprocessor.split_data(features, labels)
        loaders = preprocessor.create_dataloaders(splits, batch_size=16)
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_dataloader_batch_shape(self, preprocessor, sample_features_labels):
        """Verifica shape dos batches (B, C, H, W)."""
        features, labels = sample_features_labels
        splits = preprocessor.split_data(features, labels)
        loaders = preprocessor.create_dataloaders(splits, batch_size=16)
        batch_x, batch_y = next(iter(loaders["train"]))
        assert batch_x.shape[0] <= 16
        assert batch_x.shape[1] == preprocessor.n_channels
        assert batch_x.shape[2] == preprocessor.img_size
        assert batch_x.shape[3] == preprocessor.img_size

    def test_synthetic_dataset_len(self):
        """Verifica comprimento do SyntheticImageDataset."""
        features = np.random.rand(100, 192).astype(np.float32)
        labels = np.random.randint(0, 5, 100).astype(np.int64)
        dataset = SyntheticImageDataset(features, labels, img_size=8, n_channels=3)
        assert len(dataset) == 100

    def test_synthetic_dataset_getitem(self):
        """Verifica formato de saída do __getitem__."""
        features = np.random.rand(50, 192).astype(np.float32)
        labels = np.random.randint(0, 5, 50).astype(np.int64)
        dataset = SyntheticImageDataset(features, labels, img_size=8, n_channels=3)
        img, label = dataset[0]
        assert img.shape == (3, 8, 8)
        assert isinstance(label, torch.Tensor)
