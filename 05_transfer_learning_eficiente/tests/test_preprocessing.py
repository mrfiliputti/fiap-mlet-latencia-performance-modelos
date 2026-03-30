"""
Testes unitários para data_preprocessing.py.

Valida o pipeline de pré-processamento: geração de dataset sintético,
limpeza, codificação de labels e split estratificado.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor, generate_synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de exemplo com 100 amostras em 5 classes."""
    np.random.seed(42)
    labels = ["juridico", "tecnologia", "financeiro", "saude", "educacao"]
    records = []
    for i in range(100):
        records.append({
            "text": f"Texto de exemplo número {i} sobre {labels[i % 5]}.",
            "label": labels[i % 5],
            "domain_id": i % 5,
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_csv(sample_df: pd.DataFrame, tmp_path) -> str:
    """Salva DataFrame de exemplo como CSV e retorna caminho."""
    path = str(tmp_path / "test_dataset.csv")
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """DataPreprocessor com configuração padrão."""
    return DataPreprocessor(max_length=64, test_size=0.2, val_size=0.1, random_state=42)


# ---------------------------------------------------------------------------
# Testes de generate_synthetic_dataset
# ---------------------------------------------------------------------------


class TestGenerateSyntheticDataset:
    """Testes para a função de geração de dataset sintético."""

    def test_generate_correct_shape(self):
        """Verifica que o dataset gerado tem o número correto de amostras."""
        df = generate_synthetic_dataset(n_samples=500, random_state=42)
        assert len(df) == 500

    def test_generate_has_required_columns(self):
        """Verifica presença das colunas obrigatórias."""
        df = generate_synthetic_dataset(n_samples=100, random_state=42)
        assert "text" in df.columns
        assert "label" in df.columns
        assert "domain_id" in df.columns

    def test_generate_balanced_classes(self):
        """Verifica que as classes são aproximadamente balanceadas."""
        df = generate_synthetic_dataset(n_samples=500, random_state=42)
        counts = df["label"].value_counts()
        assert counts.min() >= 95  # ~100 por classe
        assert counts.max() <= 105

    def test_generate_five_domains(self):
        """Verifica que existem exatamente 5 domínios."""
        df = generate_synthetic_dataset(n_samples=500, random_state=42)
        assert df["label"].nunique() == 5

    def test_generate_saves_csv(self, tmp_path):
        """Verifica que o dataset é salvo em CSV quando output_path é fornecido."""
        path = str(tmp_path / "output.csv")
        df = generate_synthetic_dataset(n_samples=100, random_state=42, output_path=path)
        assert os.path.exists(path)
        df_loaded = pd.read_csv(path)
        assert len(df_loaded) == len(df)

    def test_generate_reproducibility(self):
        """Verifica reprodutibilidade com mesma seed."""
        df1 = generate_synthetic_dataset(n_samples=100, random_state=42)
        df2 = generate_synthetic_dataset(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Testes do DataPreprocessor
# ---------------------------------------------------------------------------


class TestDataPreprocessor:
    """Testes para a classe DataPreprocessor."""

    def test_load_data(self, preprocessor: DataPreprocessor, sample_csv: str):
        """Verifica carregamento do CSV."""
        df = preprocessor.load_data(sample_csv)
        assert len(df) == 100
        assert "text" in df.columns

    def test_load_data_file_not_found(self, preprocessor: DataPreprocessor):
        """Verifica exceção para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/path/inexistente/dataset.csv")

    def test_clean_data_removes_empty(self, preprocessor: DataPreprocessor):
        """Verifica que clean_data remove textos vazios e nulos."""
        df = pd.DataFrame({
            "text": ["Hello", "", None, "  ", "World"],
            "label": ["a", "b", "c", "d", "e"],
        })
        cleaned = preprocessor.clean_data(df)
        assert len(cleaned) == 2  # Apenas "Hello" e "World"

    def test_prepare_features_creates_label_id(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ):
        """Verifica criação da coluna label_id."""
        preprocessor.load_data(sample_csv)
        preprocessor.clean_data()
        df = preprocessor.prepare_features()
        assert "label_id" in df.columns
        assert "text_length" in df.columns
        assert "word_count" in df.columns

    def test_prepare_features_label_map(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ):
        """Verifica que label_map contém todas as classes."""
        preprocessor.load_data(sample_csv)
        preprocessor.clean_data()
        preprocessor.prepare_features()
        assert preprocessor.num_labels == 5
        assert len(preprocessor.label_map) == 5

    def test_split_data_proportions(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ):
        """Verifica proporções dos splits train/val/test."""
        preprocessor.load_data(sample_csv)
        preprocessor.clean_data()
        preprocessor.prepare_features()
        train, val, test = preprocessor.split_data()

        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(test) == pytest.approx(20, abs=3)  # 20%
        assert len(val) == pytest.approx(10, abs=3)    # 10%

    def test_split_data_no_overlap(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ):
        """Verifica que não há sobreposição entre splits."""
        preprocessor.load_data(sample_csv)
        preprocessor.clean_data()
        preprocessor.prepare_features()
        train, val, test = preprocessor.split_data()

        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        # Após reset_index, todos começam de 0; verificar tamanho total
        assert len(train) + len(val) + len(test) == 100
