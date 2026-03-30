"""
Utilitários para serialização, configuração e logging estruturado.

Funções auxiliares para persistência de resultados, configuração
de experimentos e logging em formato JSON, seguindo boas práticas
de observabilidade conforme Sridharan (2018).

Referências:
    Sridharan, C. (2018). Distributed Systems Observability. O'Reilly.
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems. NeurIPS.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ================================================================== #
#  Serialização                                                        #
# ================================================================== #


def save_model(model: Any, path: str) -> None:
    """Salva modelo (ou objeto) em formato pickle.

    Args:
        model: Objeto a serializar.
        path: Caminho de destino.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    """Carrega modelo serializado em pickle.

    Args:
        path: Caminho do arquivo.

    Returns:
        Objeto deserializado.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Persiste métricas em JSON.

    Usa ``NumpyEncoder`` para serializar tipos NumPy.

    Args:
        metrics: Dicionário de métricas.
        path: Caminho de destino.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def load_metrics(path: str) -> Dict[str, Any]:
    """Carrega métricas de arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Dicionário de métricas.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder que suporta tipos NumPy."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


# ================================================================== #
#  Configuração                                                        #
# ================================================================== #


DEFAULT_CONFIG: Dict[str, Any] = {
    "slo_latency_p99_ms": 200.0,
    "slo_availability": 0.999,
    "window_days": 30,
    "autoscaler": {
        "min_replicas": 2,
        "max_replicas": 10,
        "cpu_upper": 0.70,
        "cpu_lower": 0.30,
        "cooldown_steps": 5,
    },
    "canary": {
        "alpha": 0.05,
        "stages": [0.10, 0.25, 0.50, 1.0],
        "latency_degradation_threshold": 0.10,
        "error_increase_threshold": 0.50,
    },
    "dataset": {
        "n_samples": 10_000,
        "seed": 42,
    },
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Carrega configuração de arquivo JSON ou retorna defaults.

    Args:
        path: Caminho do arquivo de configuração. Se ``None``,
            retorna ``DEFAULT_CONFIG``.

    Returns:
        Dicionário de configuração.
    """
    if path is None or not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Merge com defaults
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    return merged


def save_config(config: Dict[str, Any], path: str) -> None:
    """Salva configuração em JSON.

    Args:
        config: Dicionário de configuração.
        path: Caminho de destino.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ================================================================== #
#  Logging estruturado                                                 #
# ================================================================== #


def setup_logger(
    name: str = "ml_serving",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura logger com formato JSON estruturado.

    Segue recomendações de Sridharan (2018) para observabilidade:
    logs em JSON com campos padronizados (timestamp, level, message).

    Args:
        name: Nome do logger.
        log_file: Caminho para arquivo de log. Se ``None``, usa stdout.
        level: Nível de logging.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar handlers duplicados
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_inference_event(
    logger: logging.Logger,
    request_id: str,
    model_version: str,
    modality: str,
    latency_ms: float,
    is_error: bool,
    confidence: float,
) -> None:
    """Registra evento de inferência em log estruturado.

    Campos recomendados na seção 'Monitoramento de Latência e
    Observabilidade em Produção' do material da aula.

    Args:
        logger: Logger configurado.
        request_id: ID da requisição.
        model_version: Versão do modelo.
        modality: Modalidade processada.
        latency_ms: Latência em ms.
        is_error: Se houve erro.
        confidence: Confiança da predição.
    """
    msg = (
        f"request_id={request_id} model_version={model_version} "
        f"modality={modality} latency_ms={latency_ms:.2f} "
        f"is_error={is_error} confidence={confidence:.4f}"
    )
    if is_error:
        logger.error(msg)
    else:
        logger.info(msg)
