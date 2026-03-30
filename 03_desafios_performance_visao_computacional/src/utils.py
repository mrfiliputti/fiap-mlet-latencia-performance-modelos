"""
Utilitários para benchmarking de visão computacional.

Fornece funções auxiliares para medição de tempo, serialização de modelos,
gerenciamento de configuração e logging, usados ao longo de todos os módulos
e notebooks desta aula.

Referência:
    Reddi, V. J. et al. (2020). MLPerf Inference Benchmark. ISCA '20.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Timer / Profiling
# -------------------------------------------------------------------------


class BenchmarkTimer:
    """
    Context manager para medição precisa de latência.

    Implementa medição de tempo com warm-up e múltiplas repetições
    conforme boas práticas de benchmarking (Reddi et al., 2020).
    Retorna estatísticas de latência incluindo P50, P95 e P99.

    Attributes:
        measurements: Lista de tempos medidos em milissegundos.
    """

    def __init__(self) -> None:
        self.measurements: list[float] = []
        self._start: float = 0.0

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager que registra o tempo decorrido em ms."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.measurements.append(elapsed_ms)

    def run(
        self,
        fn: Any,
        *args: Any,
        n_warmup: int = 5,
        n_runs: int = 50,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Executa função com warm-up e coleta estatísticas de latência.

        Conforme MLPerf Benchmark (Reddi et al., 2020), warm-up é essencial
        para eliminar efeitos de cache cold-start e JIT compilation.

        Args:
            fn: Função a ser benchmarked.
            *args: Argumentos posicionais para fn.
            n_warmup: Número de execuções de warm-up (descartadas).
            n_runs: Número de execuções para medição.
            **kwargs: Argumentos nomeados para fn.

        Returns:
            Dicionário com estatísticas de latência (mean, std, p50, p95, p99).
        """
        # Warm-up — descartar primeiras execuções
        for _ in range(n_warmup):
            fn(*args, **kwargs)

        # Medição
        self.measurements = []
        for _ in range(n_runs):
            with self.measure():
                fn(*args, **kwargs)

        return self.statistics()

    def statistics(self) -> dict[str, float]:
        """Calcula estatísticas descritivas das medições."""
        if not self.measurements:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "n_runs": 0,
            }
        arr = np.array(self.measurements)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "n_runs": len(self.measurements),
        }

    def reset(self) -> None:
        """Limpa histórico de medições."""
        self.measurements = []


# -------------------------------------------------------------------------
# Serialização de modelos
# -------------------------------------------------------------------------


def save_model(model: Any, path: str, metadata: Optional[dict] = None) -> str:
    """
    Salva modelo PyTorch com metadados opcionais.

    Args:
        model: Modelo PyTorch (nn.Module) ou state_dict.
        path: Caminho de destino (.pt ou .pth).
        metadata: Dicionário com metadados adicionais (hyperparams, etc.).

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    import torch

    dirpath = Path(path).parent
    dirpath.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": (
            model.state_dict() if hasattr(model, "state_dict") else model
        ),
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, path)
    logger.info("Modelo salvo em %s", path)
    return str(Path(path).resolve())


def load_model(path: str) -> dict[str, Any]:
    """
    Carrega checkpoint de modelo PyTorch.

    Args:
        path: Caminho do arquivo .pt/.pth.

    Returns:
        Dicionário com 'model_state_dict', 'metadata' e 'timestamp'.

    Raises:
        FileNotFoundError: Se o arquivo não existe.
    """
    import torch

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Modelo carregado de %s", path)
    return checkpoint


# -------------------------------------------------------------------------
# Métricas / Logging
# -------------------------------------------------------------------------


def save_metrics(metrics: dict[str, Any], path: str) -> str:
    """
    Salva métricas em arquivo JSON.

    Args:
        metrics: Dicionário de métricas.
        path: Caminho de destino (.json).

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    dirpath = Path(path).parent
    dirpath.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Métricas salvas em %s", path)
    return str(Path(path).resolve())


def load_metrics(path: str) -> dict[str, Any]:
    """
    Carrega métricas de arquivo JSON.

    Args:
        path: Caminho do arquivo .json.

    Returns:
        Dicionário de métricas.

    Raises:
        FileNotFoundError: Se o arquivo não existe.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Configuração
# -------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "benchmark": {
        "n_warmup": 5,
        "n_runs": 50,
        "resolutions": [128, 224, 320, 384, 512],
    },
    "models": {
        "cnn": ["mobilenetv2_100", "efficientnet_b0", "resnet18", "resnet50"],
        "vit": ["deit_tiny_patch16_224", "deit_small_patch16_224"],
    },
    "paths": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "models": "outputs/models",
        "figures": "outputs/figures",
        "logs": "outputs/logs",
    },
}


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """
    Carrega configuração de arquivo JSON ou retorna configuração padrão.

    Args:
        path: Caminho para arquivo de configuração JSON. Se None, usa default.

    Returns:
        Dicionário de configuração.
    """
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        # Merge com defaults
        config = {**DEFAULT_CONFIG, **user_config}
        return config
    return DEFAULT_CONFIG.copy()


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configura logging para a aula.

    Args:
        level: Nível de logging (default: INFO).
        log_file: Caminho opcional para arquivo de log.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def ensure_dirs(base_path: str = ".") -> None:
    """Cria diretórios de saída se não existirem."""
    for subdir in ["outputs/models", "outputs/figures", "outputs/logs",
                    "data/raw", "data/processed"]:
        Path(base_path, subdir).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """
    Detecta dispositivo disponível (CUDA > MPS > CPU).

    Returns:
        String do dispositivo: 'cuda', 'mps' ou 'cpu'.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """
    Fixa seeds para reprodutibilidade.

    Args:
        seed: Valor da seed.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
