"""
Módulo de utilitários para a Aula 07.

Funções auxiliares para serialização de métricas, configurações
e resultados de benchmarks de inferência distribuída.

Referência:
    Conceitos gerais de instrumentação e métricas conforme discutido
    na seção 'Estratégias de Auto-scaling e SLOs' do material da aula.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# =========================================================================
# Serialização de métricas
# =========================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder que suporta tipos NumPy."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_metrics(
    metrics: dict,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Salva métricas de benchmark em arquivo JSON.

    Args:
        metrics: Dicionário de métricas.
        path: Caminho do arquivo de saída.
        metadata: Metadados opcionais (timestamp, descrição, etc.).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    if metadata:
        output["metadata"] = metadata

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def load_metrics(path: str) -> dict:
    """Carrega métricas de arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Dicionário com métricas carregadas.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Configurações
# =========================================================================

def save_config(config: dict, path: str) -> None:
    """Salva configuração de experimento.

    Args:
        config: Dicionário de configuração.
        path: Caminho do arquivo de saída.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def load_config(path: str) -> dict:
    """Carrega configuração de experimento.

    Args:
        path: Caminho do arquivo.

    Returns:
        Dicionário com configuração.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Helpers de caminhos
# =========================================================================

def get_project_root() -> Path:
    """Retorna o diretório raiz da aula 07.

    Returns:
        Path para 07_inferencia_distribuida_paralelismo/.
    """
    return Path(__file__).resolve().parent.parent


def ensure_output_dirs() -> dict[str, Path]:
    """Cria e retorna diretórios de saída.

    Returns:
        Dicionário com paths de outputs/{models,figures,logs}.
    """
    root = get_project_root()
    dirs = {
        "models": root / "outputs" / "models",
        "figures": root / "outputs" / "figures",
        "logs": root / "outputs" / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =========================================================================
# Formatação de resultados
# =========================================================================

def format_benchmark_table(results: pd.DataFrame, title: str = "") -> str:
    """Formata DataFrame de benchmark como tabela legível.

    Args:
        results: DataFrame com resultados.
        title: Título opcional.

    Returns:
        String formatada.
    """
    lines = []
    if title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {title}")
        lines.append(f"{'='*60}")
    lines.append(results.to_string(index=False))
    return "\n".join(lines)


def compute_memory_savings(
    sequence_lengths: np.ndarray,
    block_size: int = 16,
) -> dict:
    """Calcula economia de memória do PagedAttention vs. alocação contígua.

    Implementa análise quantitativa da seção 'PagedAttention e Gerenciamento
    de KV-Cache' (Kwon et al., 2023): alocação contígua desperdiça ~50% vs.
    PagedAttention com >95% de utilização.

    Args:
        sequence_lengths: Comprimentos de sequências observados.
        block_size: Tamanho de bloco do PagedAttention.

    Returns:
        Dicionário com métricas de economia.
    """
    max_seq = int(np.max(sequence_lengths))

    # Alocação contígua: aloca max_seq para todas
    contig_total = len(sequence_lengths) * max_seq
    actual_total = int(np.sum(sequence_lengths))

    # PagedAttention: aloca ceil(n/B) * B para cada
    paged_total = int(
        np.sum(np.ceil(sequence_lengths / block_size).astype(int) * block_size)
    )

    return {
        "actual_tokens": actual_total,
        "contiguous_allocated": contig_total,
        "paged_allocated": paged_total,
        "contiguous_waste": contig_total - actual_total,
        "paged_waste": paged_total - actual_total,
        "contiguous_utilization": actual_total / max(contig_total, 1),
        "paged_utilization": actual_total / max(paged_total, 1),
        "savings_vs_contiguous": 1 - paged_total / max(contig_total, 1),
    }
