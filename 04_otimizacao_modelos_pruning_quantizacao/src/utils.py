"""
Módulo de utilitários: serialização, logging e helpers.

Funções auxiliares para salvar/carregar modelos, registrar métricas e
gerenciar configurações do pipeline de compressão.

Referência:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning
    Systems. NeurIPS.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_model(model: nn.Module, filepath: str) -> None:
    """Salva state_dict do modelo em disco.

    Args:
        model: Modelo treinado.
        filepath: Caminho .pt para salvar.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, filepath: str, device: str = "cpu") -> nn.Module:
    """Carrega state_dict em modelo existente.

    Args:
        model: Instância do modelo (mesma arquitetura).
        filepath: Caminho do state_dict .pt.
        device: Dispositivo para mapear pesos.

    Returns:
        Modelo com pesos carregados.
    """
    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def save_metrics(metrics: dict[str, Any], filepath: str) -> None:
    """Salva métricas em arquivo JSON.

    Args:
        metrics: Dicionário de métricas.
        filepath: Caminho .json para salvar.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Converte tipos numpy para serialização JSON
    sanitized: dict[str, Any] = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            sanitized[k] = v.item()
        elif isinstance(v, dict):
            sanitized[k] = {
                sk: sv.item() if hasattr(sv, "item") else sv for sk, sv in v.items()
            }
        else:
            sanitized[k] = v

    sanitized["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)


def load_metrics(filepath: str) -> dict[str, Any]:
    """Carrega métricas de arquivo JSON.

    Args:
        filepath: Caminho do JSON.

    Returns:
        Dicionário de métricas.
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Retorna melhor dispositivo disponível (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Define seeds para reprodutibilidade.

    Args:
        seed: Semente aleatória.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_size_on_disk_mb(model: nn.Module) -> float:
    """Calcula tamanho do modelo serializado em disco (MB).

    Mais preciso que medir pela memória pois inclui overhead de
    serialização. Útil para comparar modelos FP32 vs INT8 conforme
    a Tabela 2 do documento 04.

    Args:
        model: Modelo PyTorch.

    Returns:
        Tamanho em MB.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        size = os.path.getsize(tmp.name) / (1024 * 1024)
    os.unlink(tmp.name)
    return size
