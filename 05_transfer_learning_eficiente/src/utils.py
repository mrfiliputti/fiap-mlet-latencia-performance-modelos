"""
Utilitários para Transfer Learning Eficiente.

Funções auxiliares para serialização, carregamento, configuração e
instrumentação de tempo/memória. Utilizadas ao longo de todo o pipeline
de fine-tuning (LoRA/QLoRA) e exportação ONNX.

Referências:
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
        Language Models. ICLR.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialização de modelos
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str, *, save_peft_only: bool = False) -> None:
    """
    Salva modelo PyTorch ou adaptador PEFT em disco.

    Para modelos PEFT, salvar apenas os adaptadores é mais eficiente
    (somente ~0.06% dos parâmetros para LoRA r=16), conforme discutido
    na seção 'Low-Rank Adaptation (LoRA)' do material da Aula 5.

    Args:
        model: Modelo PyTorch ou PeftModel.
        path: Diretório destino.
        save_peft_only: Se True, salva apenas os adaptadores PEFT.
    """
    os.makedirs(path, exist_ok=True)
    if save_peft_only and hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        logger.info("Adaptadores PEFT salvos em %s", path)
    else:
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        logger.info("State dict salvo em %s", path)


def load_model(model: Any, path: str) -> Any:
    """
    Carrega pesos em modelo PyTorch existente.

    Args:
        model: Modelo com arquitetura compatível.
        path: Caminho para o arquivo .pt ou diretório PEFT.

    Returns:
        Modelo com pesos carregados.
    """
    pt_file = os.path.join(path, "model.pt") if os.path.isdir(path) else path
    if os.path.exists(pt_file):
        state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pesos carregados de %s", pt_file)
    else:
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {pt_file}")
    return model


# ---------------------------------------------------------------------------
# Métricas / logs
# ---------------------------------------------------------------------------


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """
    Salva dicionário de métricas como JSON.

    Args:
        metrics: Dicionário de métricas (ex: accuracy, f1, latency).
        path: Caminho do arquivo JSON.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Métricas salvas em %s", path)


def load_metrics(path: str) -> Dict[str, Any]:
    """
    Carrega métricas de arquivo JSON.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Dicionário de métricas.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 5,
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "output_dir": "outputs",
    "random_state": 42,
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega configuração de arquivo JSON ou retorna defaults.

    Configurações padrão usam hiperparâmetros recomendados pela
    literatura PEFT: r=16, alpha=32, lr=2e-4 (Hu et al., 2022).

    Args:
        path: Caminho para JSON de configuração. Se None, usa defaults.

    Returns:
        Dicionário de configuração.
    """
    if path is not None and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        merged = {**DEFAULT_CONFIG, **config}
        logger.info("Configuração carregada de %s", path)
        return merged
    return DEFAULT_CONFIG.copy()


# ---------------------------------------------------------------------------
# Instrumentação de tempo e memória
# ---------------------------------------------------------------------------


@contextmanager
def timer(description: str = "Operação") -> Generator[Dict[str, float], None, None]:
    """
    Context manager para medir tempo de execução.

    Útil para benchmarking de latência conforme discutido na seção
    'Estratégias de Warm-up e Caching' do material da Aula 5.

    Args:
        description: Descrição da operação.

    Yields:
        Dicionário com 'elapsed' preenchido ao sair do contexto.

    Exemplo:
        >>> with timer("Inferência") as t:
        ...     outputs = model(inputs)
        >>> print(f"Latência: {t['elapsed']:.4f}s")
    """
    result: Dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        logger.info("%s: %.4fs", description, result["elapsed"])


def get_gpu_memory_mb() -> float:
    """
    Retorna uso atual de memória GPU em MB.

    Relevante para comparar consumo de memória entre fine-tuning
    completo, LoRA e QLoRA (Tabela 2 do material da Aula 5).

    Returns:
        Memória alocada em MB, ou 0.0 se CUDA não disponível.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calcula tamanho do modelo em MB baseado nos parâmetros.

    Args:
        model: Módulo PyTorch.

    Returns:
        Tamanho estimado em MB.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 2)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Conta parâmetros totais e treináveis de um modelo.

    Central para demonstrar a eficiência de LoRA, que treina apenas
    ~0.06% dos parâmetros para r=16 (Hu et al., 2022, Tabela 2).

    Args:
        model: Módulo PyTorch (base ou PEFT).

    Returns:
        {'total': int, 'trainable': int, 'frozen': int, 'trainable_pct': float}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(pct, 4),
    }


def set_seed(seed: int = 42) -> None:
    """
    Define seed global para reprodutibilidade.

    Args:
        seed: Valor inteiro da seed.
    """
    import random
    random.seed(seed)
    np._set_seed_helper(seed) if hasattr(np, "_set_seed_helper") else None
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Cria diretório se não existir e retorna o caminho."""
    os.makedirs(path, exist_ok=True)
    return path
