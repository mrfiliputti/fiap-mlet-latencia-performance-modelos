"""
Testes unitários para o módulo data_preprocessing.

Verifica geração de datasets, carregamento, preparação de features
e filtragem por resolução.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Instância do DataPreprocessor."""
    return DataPreprocessor(random_state=42)


@pytest.fixture
def benchmark_csv(tmp_path) -> str:
    """Gera dataset de benchmark temporário e retorna caminho."""
    path = str(tmp_path / "dataset.csv")
    DataPreprocessor.generate_benchmark_dataset(
        output_path=path,
        n_runs_per_config=3,  # Menos runs para testes rápidos
        random_state=42,
    )
    return path


@pytest.fixture
def loaded_preprocessor(preprocessor: DataPreprocessor, benchmark_csv: str) -> DataPreprocessor:
    """Preprocessor com dados carregados."""
    preprocessor.load_data(benchmark_csv)
    return preprocessor


# ---------------------------------------------------------------------------
# Testes de geração de dataset
# ---------------------------------------------------------------------------


class TestGenerateBenchmarkDataset:
    """Testes de geração do dataset de benchmarks."""

    def test_generates_csv_file(self, tmp_path) -> None:
        """CSV é criado no caminho especificado."""
        path = str(tmp_path / "test.csv")
        DataPreprocessor.generate_benchmark_dataset(path, n_runs_per_config=2)
        assert os.path.exists(path)

    def test_correct_columns(self, benchmark_csv: str) -> None:
        """Dataset possui todas as colunas esperadas."""
        df = pd.read_csv(benchmark_csv)
        expected_cols = {
            "model_name", "model_family", "input_resolution", "hardware",
            "batch_size", "params_millions", "flops_giga", "top1_accuracy",
            "latency_ms", "throughput_imgs_sec", "memory_mb", "run_id",
        }
        assert expected_cols == set(df.columns)

    def test_has_all_families(self, benchmark_csv: str) -> None:
        """Dataset contém modelos CNN, ViT e hybrid."""
        df = pd.read_csv(benchmark_csv)
        families = set(df["model_family"].unique())
        assert {"cnn", "vit", "hybrid"} == families

    def test_positive_values(self, benchmark_csv: str) -> None:
        """Valores numéricos são positivos."""
        df = pd.read_csv(benchmark_csv)
        for col in ["params_millions", "flops_giga", "latency_ms", "throughput_imgs_sec"]:
            assert (df[col] > 0).all(), f"Coluna {col} tem valores não-positivos"

    def test_deterministic_with_seed(self, tmp_path) -> None:
        """Geração é determinística com mesma seed."""
        path1 = str(tmp_path / "d1.csv")
        path2 = str(tmp_path / "d2.csv")
        DataPreprocessor.generate_benchmark_dataset(path1, n_runs_per_config=2, random_state=99)
        DataPreprocessor.generate_benchmark_dataset(path2, n_runs_per_config=2, random_state=99)
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Testes de carregamento
# ---------------------------------------------------------------------------


class TestLoadData:
    """Testes de carregamento de dados."""

    def test_load_existing_file(self, preprocessor: DataPreprocessor, benchmark_csv: str) -> None:
        """Carrega CSV existente com sucesso."""
        df = preprocessor.load_data(benchmark_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_nonexistent_raises(self, preprocessor: DataPreprocessor) -> None:
        """FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("caminho/nao/existe.csv")

    def test_data_stored(self, loaded_preprocessor: DataPreprocessor) -> None:
        """Dados ficam armazenados em self.data."""
        assert loaded_preprocessor.data is not None
        assert len(loaded_preprocessor.data) > 0


# ---------------------------------------------------------------------------
# Testes de preparação de features
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    """Testes de agregação de features."""

    def test_aggregation_reduces_rows(self, loaded_preprocessor: DataPreprocessor) -> None:
        """Agregação reduz número de linhas."""
        agg = loaded_preprocessor.prepare_features()
        assert len(agg) < len(loaded_preprocessor.data)

    def test_aggregation_has_stats(self, loaded_preprocessor: DataPreprocessor) -> None:
        """DataFrame agregado contém colunas de estatísticas."""
        agg = loaded_preprocessor.prepare_features()
        assert "latency_mean_ms" in agg.columns
        assert "latency_std_ms" in agg.columns


# ---------------------------------------------------------------------------
# Testes de imagens sintéticas
# ---------------------------------------------------------------------------


class TestGenerateSyntheticImages:
    """Testes de geração de imagens sintéticas."""

    def test_output_shapes(self) -> None:
        """Shapes corretos para imagens e labels."""
        images, labels = DataPreprocessor.generate_synthetic_images(
            output_dir=None, n_samples=100, img_size=32, n_classes=4,
        )
        assert images.shape == (100, 3, 32, 32)
        assert labels.shape == (100,)

    def test_pixel_range(self) -> None:
        """Pixels no intervalo [0, 1]."""
        images, _ = DataPreprocessor.generate_synthetic_images(
            output_dir=None, n_samples=50, img_size=32,
        )
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_all_classes_present(self) -> None:
        """Todas as classes estão representadas."""
        _, labels = DataPreprocessor.generate_synthetic_images(
            output_dir=None, n_samples=100, n_classes=4,
        )
        assert set(np.unique(labels)) == {0, 1, 2, 3}
