"""
Funções utilitárias para a Aula 02.

Fornece timers para benchmarking, I/O de modelos e métricas,
e carregamento de configuração. Conecta-se às necessidades de
medição de latência discutidas na seção 'Decomposição da Latência
em Transformers' (Shoeybi et al., 2019).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


# ================================================================== #
#  Timer para benchmarking                                           #
# ================================================================== #

class Timer:
    """Context-manager para medição de latência com alta resolução.

    Utiliza ``time.perf_counter()`` que, em CPython no Windows/Linux,
    oferece resolução sub-microsegundo — adequado para medir seções
    curtas conforme a decomposição de latência de Transformers.

    Exemplo::

        timer = Timer()
        with timer.measure("attention"):
            output = model.forward(X)
        print(timer.get("attention"))  # ms
    """

    def __init__(self) -> None:
        self._records: Dict[str, float] = {}
        self._start: Optional[float] = None

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        """Mede tempo de um bloco e registra com um rótulo.

        Args:
            label: Nome da medição.
        """
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        self._records[label] = elapsed

    def get(self, label: str) -> float:
        """Retorna tempo registrado em ms.

        Args:
            label: Nome da medição.

        Returns:
            Tempo em milissegundos.
        """
        return self._records.get(label, 0.0)

    def all(self) -> Dict[str, float]:
        """Retorna todos os registros."""
        return dict(self._records)

    def reset(self) -> None:
        """Limpa registros."""
        self._records.clear()


# ================================================================== #
#  I/O de modelos                                                    #
# ================================================================== #

def save_model(model: Any, path: Optional[str] = None, name: str = "model") -> str:
    """Serializa modelo (ou qualquer objeto) com pickle.

    Args:
        model: Objeto a salvar.
        path: Diretório de destino. Padrão: outputs/models/.
        name: Nome do arquivo (sem extensão).

    Returns:
        Caminho completo do arquivo salvo.
    """
    if path is None:
        path = str(BASE_DIR / "outputs" / "models")
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def load_model(filepath: str) -> Any:
    """Carrega modelo serializado com pickle.

    Args:
        filepath: Caminho do arquivo .pkl.

    Returns:
        Objeto desserializado.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)  # noqa: S301 — uso pedagógico controlado


# ================================================================== #
#  I/O de métricas                                                   #
# ================================================================== #

def save_metrics(
    metrics: Dict[str, Any],
    path: Optional[str] = None,
    name: str = "metrics",
) -> str:
    """Salva métricas em JSON.

    Args:
        metrics: Dicionário de métricas.
        path: Diretório. Padrão: outputs/logs/.
        name: Nome do arquivo.

    Returns:
        Caminho do arquivo salvo.
    """
    if path is None:
        path = str(BASE_DIR / "outputs" / "logs")
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.json")

    # Converter tipos numpy para Python nativo
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {k: _convert(v) for k, v in metrics.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return filepath


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Carrega métricas de JSON.

    Args:
        filepath: Caminho do arquivo .json.

    Returns:
        Dicionário de métricas.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================== #
#  Configuração                                                      #
# ================================================================== #

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Carrega arquivo de configuração YAML.

    Args:
        path: Caminho do YAML. Se None, usa config padrão.

    Returns:
        Dicionário de configuração.
    """
    if path is None:
        # Configuração padrão embutida
        return {
            "seed": 42,
            "d_model": 128,
            "num_heads": 4,
            "seq_lengths": [32, 64, 128, 256, 512],
            "batch_size": 2,
            "n_benchmark_runs": 5,
            "audio": {
                "sample_rate": 16000,
                "n_mels": 80,
                "n_fft": 400,
                "hop_length": 160,
            },
            "training": {
                "learning_rate": 0.001,
                "n_epochs": 100,
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_bytes(n_bytes: int) -> str:
    """Formata bytes em unidade legível (KB, MB, GB).

    Args:
        n_bytes: Número de bytes.

    Returns:
        String formatada.
    """
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024**2:.1f} MB"
    else:
        return f"{n_bytes / 1024**3:.2f} GB"
