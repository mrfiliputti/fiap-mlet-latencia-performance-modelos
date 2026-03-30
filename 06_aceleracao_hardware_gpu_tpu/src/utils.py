"""
Módulo de utilitários para aceleração por hardware.

Funções auxiliares para gerenciamento de dispositivos (CPU/GPU),
serialização de modelos, métricas e configuração. Inclui detecção
automática de capacidades de hardware (CUDA, Tensor Cores) relevantes
para as técnicas de aceleração discutidas no material da Aula 06.

Referências:
    NVIDIA Corporation (2022). NVIDIA H100 Tensor Core GPU Architecture.
    Kirk, D.B., Hwu, W. (2022). Programming Massively Parallel Processors.
"""

import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ---------------------------------------------------------------------------
# Gerenciamento de Dispositivos
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Detecta e retorna melhor dispositivo disponível.

    Verifica disponibilidade de CUDA (GPU NVIDIA) conforme hierarquia
    de aceleradores discutida na seção 'Fundamentos da Arquitetura
    de GPUs para Machine Learning'.

    Args:
        prefer_gpu: Se True, prefere GPU quando disponível.

    Returns:
        torch.device para CPU ou CUDA.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Usando GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Usando CPU")
    return device


def get_hardware_info() -> Dict[str, Any]:
    """Coleta informações detalhadas do hardware disponível.

    Inspeciona capacidades do dispositivo relevantes para as técnicas
    da aula: Tensor Cores, suporte a BF16/FP16, memória HBM.

    Conceitos da seção 'Fundamentos da Arquitetura de GPUs':
    - Streaming Multiprocessors (SMs)
    - Tensor Cores
    - HBM (High Bandwidth Memory)

    Returns:
        Dicionário com informações de hardware.

    Referência:
        NVIDIA Corporation (2022). H100 Tensor Core GPU Architecture Whitepaper.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": gpu_props.name,
            "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            # Total de memória HBM — conceito da seção 'Hierarquia de Memória'
            "gpu_total_memory_gb": gpu_props.total_mem / (1024 ** 3),
            # Contagem de SMs — organização fundamental discutida na aula
            "gpu_sm_count": gpu_props.multi_processor_count,
            "cuda_version": torch.version.cuda or "N/A",
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
            # Suporte a BF16 — formato discutido em Mixed Precision Training
            "bf16_supported": torch.cuda.is_bf16_supported(),
            # Tensor Cores disponíveis em compute capability >= 7.0
            "tensor_cores_available": gpu_props.major >= 7,
            "n_gpus": torch.cuda.device_count(),
        })

        # Verificar capability para TF32 (Ampere+, compute >= 8.0)
        # TF32 é discutido na seção sobre Tensor Cores da aula
        info["tf32_supported"] = gpu_props.major >= 8
    else:
        info.update({
            "gpu_name": "N/A (CPU only)",
            "tensor_cores_available": False,
            "bf16_supported": False,
            "tf32_supported": False,
        })

    return info


def set_seed(seed: int = 42) -> None:
    """Define seed para reprodutibilidade.

    Args:
        seed: Semente.
    """
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinístico (pode reduzir performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Serialização de Modelos
# ---------------------------------------------------------------------------

def save_model(
    model: nn.Module,
    path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Salva modelo treinado com metadados.

    Args:
        model: Modelo PyTorch.
        path: Caminho para salvar. Se None, usa outputs/models/.
        metadata: Metadados adicionais.

    Returns:
        Caminho do arquivo salvo.
    """
    if path is None:
        path = OUTPUT_DIR / "models" / "model.pt"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, path)
    logger.info("Modelo salvo: %s (%.2f MB)",
                path, path.stat().st_size / (1024 * 1024))
    return path


def load_model(
    model: nn.Module,
    path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Carrega pesos do modelo a partir de checkpoint.

    Args:
        model: Instância do modelo (arquitetura já definida).
        path: Caminho do checkpoint.
        device: Dispositivo para carregar.

    Returns:
        Modelo com pesos carregados.
    """
    if path is None:
        path = OUTPUT_DIR / "models" / "model.pt"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("Modelo carregado de: %s", path)
    return model


# ---------------------------------------------------------------------------
# Serialização de Métricas
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: Dict[str, Any],
    path: Optional[Path] = None,
    filename: str = "metrics.json",
) -> Path:
    """Salva métricas em formato JSON.

    Args:
        metrics: Dicionário de métricas.
        path: Diretório. Se None, usa outputs/logs/.
        filename: Nome do arquivo.

    Returns:
        Caminho do arquivo salvo.
    """
    if path is None:
        path = OUTPUT_DIR / "logs"

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename

    # Converter arrays numpy para listas
    serializable = _make_serializable(metrics)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info("Métricas salvas: %s", filepath)
    return filepath


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Carrega configuração de arquivo JSON.

    Args:
        path: Caminho do arquivo de configuração.

    Returns:
        Dicionário de configuração.
    """
    if path is None:
        path = BASE_DIR / "config.json"

    path = Path(path)
    if not path.exists():
        logger.warning("Arquivo de configuração não encontrado: %s. Usando padrões.", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_serializable(obj: Any) -> Any:
    """Converte objeto para formato serializável JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# ---------------------------------------------------------------------------
# Utilitários de Performance
# ---------------------------------------------------------------------------

def estimate_flops(
    model: nn.Module,
    input_shape: tuple,
) -> Dict[str, int]:
    """Estima FLOPs do modelo para análise de roofline.

    Útil para calcular intensidade aritmética e classificar o
    modelo como compute-bound ou memory-bound usando o Modelo
    Roofline (Williams et al., 2009).

    Args:
        model: Modelo PyTorch.
        input_shape: Shape de entrada (B, C, H, W).

    Returns:
        Dicionário com estimativas de FLOPs.
    """
    total_flops = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features * batch_size
            flops = 2 * module.in_features * module.out_features * input_shape[0]
            total_flops += flops
        elif isinstance(module, nn.Conv2d):
            # Estimativa simplificada
            out_h = input_shape[2] // module.stride[0]
            out_w = input_shape[3] // module.stride[1]
            flops = (2 * module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                     * module.out_channels * out_h * out_w * input_shape[0])
            total_flops += flops

        if hasattr(module, "weight") and module.weight is not None:
            total_params += module.weight.numel()

    # Bytes = parâmetros * 4 (FP32) + ativações estimadas
    param_bytes = total_params * 4
    # Estimativa de intensidade aritmética
    ai = total_flops / max(param_bytes, 1)

    return {
        "estimated_flops": total_flops,
        "estimated_gflops": total_flops / 1e9,
        "total_parameters": total_params,
        "param_bytes_fp32": param_bytes,
        "arithmetic_intensity": ai,
    }


def format_size(n_bytes: int) -> str:
    """Formata tamanho em bytes para string legível.

    Args:
        n_bytes: Tamanho em bytes.

    Returns:
        String formatada (ex: '1.5 GB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"
