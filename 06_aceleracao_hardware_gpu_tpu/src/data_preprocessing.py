"""
Módulo de pré-processamento de dados para demonstração de aceleração por hardware.

Implementa geração de dataset sintético e preparação de dados para treinamento
de Vision Transformer, conforme caso de estudo discutido na seção 'Hands On'
do material da Aula 06. O dataset é projetado para ser suficientemente grande
para evidenciar diferenças de performance entre precisões numéricas (FP32, FP16, BF16),
conforme discutido na seção sobre Mixed Precision Training (Micikevicius et al., 2018).

Referências:
    Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    DOI: 10.48550/arXiv.1710.03740
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

# Diretórios padrão relativos à raiz da aula
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


class SyntheticImageDataset(Dataset):
    """Dataset PyTorch para imagens sintéticas de classificação.

    Converte dados tabulares (CSV) em tensores de imagem adequados para
    treinamento de Vision Transformer. Suporta configuração de dtype
    para experimentos com Mixed Precision (seção 'Mixed Precision Training'
    do material da aula).

    Args:
        features: Array de features (n_samples, n_features).
        labels: Array de labels (n_samples,).
        img_size: Tamanho da imagem (altura = largura).
        n_channels: Número de canais (3 para RGB).
        dtype: Tipo de dado dos tensores (torch.float32 ou torch.float16).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        img_size: int = 32,
        n_channels: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.features = torch.tensor(features, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.img_size = img_size
        self.n_channels = n_channels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape para formato de imagem (C, H, W)
        img = self.features[idx].reshape(self.n_channels, self.img_size, self.img_size)
        return img, self.labels[idx]


class DataPreprocessor:
    """Pré-processador de dados para experimentos de aceleração por hardware.

    Gera dataset sintético de classificação de imagens e prepara DataLoaders
    para treinamento com diferentes configurações de batch size e precisão
    numérica, conceitos essenciais para explorar Tensor Cores e Mixed Precision
    conforme discutido em Micikevicius et al. (2018).

    Args:
        n_samples: Número total de amostras.
        img_size: Tamanho da imagem (altura = largura).
        n_channels: Número de canais (3 para RGB).
        n_classes: Número de classes.
        seed: Semente para reprodutibilidade.

    Referência:
        Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        img_size: int = 32,
        n_channels: int = 3,
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.seed = seed
        self.n_features = img_size * img_size * n_channels
        self._rng = np.random.RandomState(seed)

    def generate_dataset(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """Gera dataset sintético de classificação de imagens.

        Cria clusters gaussianos no espaço de features para simular padrões
        visuais classificáveis. Dimensões são múltiplos de 64 quando possível,
        para garantir utilização ótima de Tensor Cores conforme discutido na
        seção 'Tensor Cores: Hardware Especializado para Deep Learning'.

        Args:
            output_dir: Diretório para salvar o CSV. Se None, usa data/raw/.

        Returns:
            DataFrame com features e labels.
        """
        if output_dir is None:
            output_dir = DATA_RAW_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Gerando dataset sintético: %d amostras, %d features, %d classes",
                     self.n_samples, self.n_features, self.n_classes)

        samples_per_class = self.n_samples // self.n_classes
        features_list = []
        labels_list = []

        # Gera centróides aleatórios para cada classe
        centroids = self._rng.uniform(0.2, 0.8, size=(self.n_classes, self.n_features))

        for class_id in range(self.n_classes):
            # Cada classe é um cluster gaussiano em torno do centróide
            noise = self._rng.normal(0, 0.15, size=(samples_per_class, self.n_features))
            class_features = np.clip(centroids[class_id] + noise, 0.0, 1.0)
            features_list.append(class_features)
            labels_list.extend([class_id] * samples_per_class)

        features = np.vstack(features_list).astype(np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        # Embaralhar
        shuffle_idx = self._rng.permutation(len(labels))
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        # Criar DataFrame
        col_names = [f"feature_{i}" for i in range(self.n_features)] + ["label"]
        data = np.column_stack([features, labels])
        df = pd.DataFrame(data, columns=col_names)
        df["label"] = df["label"].astype(int)

        # Salvar CSV
        csv_path = output_dir / "dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Dataset salvo em: %s", csv_path)

        return df

    def load_data(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """Carrega dataset do CSV.

        Args:
            csv_path: Caminho do CSV. Se None, usa data/raw/dataset.csv.

        Returns:
            DataFrame com features e labels.
        """
        if csv_path is None:
            csv_path = DATA_RAW_DIR / "dataset.csv"

        if not csv_path.exists():
            logger.warning("Dataset não encontrado em %s. Gerando...", csv_path)
            return self.generate_dataset()

        df = pd.read_csv(csv_path)
        logger.info("Dataset carregado: %d amostras, %d colunas", len(df), len(df.columns))
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados: remove NaN, duplicatas e valores fora do range.

        Args:
            df: DataFrame com features e labels.

        Returns:
            DataFrame limpo.
        """
        initial_len = len(df)
        df = df.dropna()
        df = df.drop_duplicates()

        # Garante features no range [0, 1]
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        df[feature_cols] = df[feature_cols].clip(0.0, 1.0)

        # Garante labels válidas
        df = df[df["label"].isin(range(self.n_classes))]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info("Removidas %d amostras inválidas", removed)

        return df.reset_index(drop=True)

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai features e labels do DataFrame.

        Args:
            df: DataFrame com colunas feature_* e label.

        Returns:
            Tupla (features, labels) como arrays NumPy.
        """
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        features = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)
        return features, labels

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Divide dados em treino, validação e teste.

        Args:
            features: Array de features.
            labels: Array de labels.
            train_ratio: Proporção para treino.
            val_ratio: Proporção para validação.
            test_ratio: Proporção para teste.

        Returns:
            Dicionário com splits {'train': (X, y), 'val': (X, y), 'test': (X, y)}.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Proporções devem somar 1.0"

        n = len(labels)
        indices = self._rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return {
            "train": (features[train_idx], labels[train_idx]),
            "val": (features[val_idx], labels[val_idx]),
            "test": (features[test_idx], labels[test_idx]),
        }

    def create_dataloaders(
        self,
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, DataLoader]:
        """Cria DataLoaders PyTorch para cada split.

        Batch size é configurável para experimentar com diferentes tamanhos —
        batches maiores aproveitam melhor Tensor Cores e paralelismo GPU,
        conforme discutido na seção sobre Tensor Cores (Jia et al., 2018).

        O pin_memory=True permite transferência assíncrona CPU→GPU,
        conceito da seção 'Hierarquia de Memória' sobre overlapping
        compute/transfer.

        Args:
            splits: Dicionário de splits do split_data().
            batch_size: Tamanho do batch (múltiplos de 8 recomendados para Tensor Cores).
            num_workers: Workers para carregamento paralelo.
            pin_memory: Se True, usa pinned memory para transferência GPU eficiente.
            dtype: Tipo de dado dos tensores.

        Returns:
            Dicionário de DataLoaders.
        """
        dataloaders = {}
        for split_name, (features, labels) in splits.items():
            dataset = SyntheticImageDataset(
                features, labels,
                img_size=self.img_size,
                n_channels=self.n_channels,
                dtype=dtype,
            )
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split_name == "train"),
            )
        return dataloaders

    def get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas descritivas do dataset.

        Args:
            df: DataFrame com features e labels.

        Returns:
            Dicionário com estatísticas.
        """
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        return {
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "n_classes": df["label"].nunique(),
            "class_distribution": df["label"].value_counts().sort_index().to_dict(),
            "feature_mean": float(df[feature_cols].values.mean()),
            "feature_std": float(df[feature_cols].values.std()),
            "feature_min": float(df[feature_cols].values.min()),
            "feature_max": float(df[feature_cols].values.max()),
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
