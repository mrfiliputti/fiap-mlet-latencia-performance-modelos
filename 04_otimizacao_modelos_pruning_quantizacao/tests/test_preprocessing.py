"""
Testes unitários para o módulo data_preprocessing.

Verifica carregamento do CIFAR-10, transforms e splits de dados.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_preprocessing import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DataPreprocessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """DataPreprocessor com configuração padrão."""
    return DataPreprocessor(data_dir="data/raw", batch_size=32, num_workers=0, seed=42)


@pytest.fixture
def loaded_preprocessor(preprocessor: DataPreprocessor) -> DataPreprocessor:
    """DataPreprocessor já com dados carregados."""
    preprocessor.load_data(download=True)
    return preprocessor


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestDataPreprocessorInit:
    """Testes de inicialização."""

    def test_default_attributes(self) -> None:
        dp = DataPreprocessor()
        assert dp.batch_size == 128
        assert dp.seed == 42

    def test_custom_batch_size(self) -> None:
        dp = DataPreprocessor(batch_size=64)
        assert dp.batch_size == 64


class TestTransforms:
    """Testes de transforms."""

    def test_train_transform_output_shape(self) -> None:
        t = DataPreprocessor.train_transform()
        assert isinstance(t, transforms.Compose)

    def test_eval_transform_output_shape(self) -> None:
        t = DataPreprocessor.eval_transform()
        assert isinstance(t, transforms.Compose)


class TestDataLoading:
    """Testes de carregamento e splits."""

    def test_load_data_creates_datasets(self, loaded_preprocessor: DataPreprocessor) -> None:
        assert loaded_preprocessor._train_dataset is not None
        assert loaded_preprocessor._test_dataset is not None

    def test_split_returns_three_loaders(self, loaded_preprocessor: DataPreprocessor) -> None:
        train_l, val_l, test_l = loaded_preprocessor.split_data(val_fraction=0.1)
        assert isinstance(train_l, DataLoader)
        assert isinstance(val_l, DataLoader)
        assert isinstance(test_l, DataLoader)

    def test_batch_shape(self, loaded_preprocessor: DataPreprocessor) -> None:
        train_l, _, _ = loaded_preprocessor.split_data()
        images, labels = next(iter(train_l))
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == images.shape[0]

    def test_calibration_loader_size(self, loaded_preprocessor: DataPreprocessor) -> None:
        cal = loaded_preprocessor.get_calibration_loader(n_samples=100)
        total = sum(x.size(0) for x, _ in cal)
        assert total == 100


class TestConstants:
    """Testes de constantes."""

    def test_cifar10_classes_count(self) -> None:
        assert len(CIFAR10_CLASSES) == 10

    def test_cifar10_mean_length(self) -> None:
        assert len(CIFAR10_MEAN) == 3

    def test_cifar10_std_length(self) -> None:
        assert len(CIFAR10_STD) == 3
