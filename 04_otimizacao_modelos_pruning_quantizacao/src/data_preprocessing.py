"""
Módulo de pré-processamento para CIFAR-10.

Encapsula carregamento, normalização e particionamento do dataset CIFAR-10
utilizado como benchmark para as técnicas de compressão de modelos discutidas
no documento 04 (Aula 4 — Pruning e Quantização).

Referência:
    Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
    Images. Technical Report, University of Toronto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# Estatísticas de normalização CIFAR-10 ──────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class DataPreprocessor:
    """Carrega e prepara o CIFAR-10 para treinamento e avaliação.

    Implementa o pipeline de pré-processamento discutido na seção *Hands On*
    do documento 04: download automático, data augmentation para treino e
    normalização padrão para teste/calibração.

    Attributes:
        data_dir: Diretório raiz dos dados.
        batch_size: Tamanho do batch para DataLoaders.
        num_workers: Workers para carregamento paralelo.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 128,
        num_workers: int = 2,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._train_dataset: Optional[datasets.CIFAR10] = None
        self._test_dataset: Optional[datasets.CIFAR10] = None

    # ── Transforms ──────────────────────────────────────────────────────

    @staticmethod
    def train_transform() -> transforms.Compose:
        """Transform para treino com data augmentation leve."""
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    @staticmethod
    def eval_transform() -> transforms.Compose:
        """Transform para avaliação / calibração (sem augmentation)."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    # ── Carregamento ────────────────────────────────────────────────────

    def load_data(self, download: bool = True) -> None:
        """Baixa (se necessário) e carrega o CIFAR-10.

        Args:
            download: Se ``True``, faz download caso não exista localmente.
        """
        self._train_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=download,
            transform=self.train_transform(),
        )
        self._test_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=download,
            transform=self.eval_transform(),
        )

    # ── Splits ──────────────────────────────────────────────────────────

    def split_data(
        self,
        val_fraction: float = 0.1,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Divide em treino, validação e teste e retorna DataLoaders.

        Args:
            val_fraction: Fração do treino reservada para validação.

        Returns:
            ``(train_loader, val_loader, test_loader)``
        """
        if self._train_dataset is None or self._test_dataset is None:
            self.load_data()

        assert self._train_dataset is not None
        assert self._test_dataset is not None

        n_total = len(self._train_dataset)
        n_val = int(n_total * val_fraction)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(
            self._train_dataset, [n_train, n_val], generator=generator
        )

        # Validação usa transform de eval (sem augmentation)
        val_dataset_eval = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=False,
            transform=self.eval_transform(),
        )
        val_subset_eval = Subset(val_dataset_eval, val_subset.indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset_eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    def get_calibration_loader(self, n_samples: int = 1000) -> DataLoader:
        """Retorna loader com *n_samples* amostras para calibração de quantização.

        Conforme discutido na seção *Saiba Mais* do documento 04,
        a calibração min-max / percentil utiliza um subconjunto
        representativo para definir os parâmetros de escala (scale)
        e ponto zero (zero-point).

        Args:
            n_samples: Número de amostras de calibração.

        Returns:
            DataLoader com subset de calibração.
        """
        if self._train_dataset is None:
            self.load_data()

        assert self._train_dataset is not None

        cal_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=False,
            transform=self.eval_transform(),
        )
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(cal_dataset), generator=generator)[:n_samples]
        cal_subset = Subset(cal_dataset, indices.tolist())

        return DataLoader(
            cal_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
