"""
Módulo de definição de modelos e compressão.

Implementa uma CNN leve para CIFAR-10 e funções de pruning/quantização
conforme as técnicas discutidas no documento 04 (Aula 4).

Referências:
    Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning both
        Weights and Connections for Efficient Neural Networks. NeurIPS.
    Frankle, J. & Carlin, M. (2019). The Lottery Ticket Hypothesis. ICLR.
    Jacob, B. et al. (2018). Quantization and Training of Neural Networks
        for Efficient Integer-Arithmetic-Only Inference. CVPR.
    Molchanov, P. et al. (2017). Pruning CNNs for Resource Efficient
        Inference. ICLR.
    Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge
        in a Neural Network. NIPS Workshop.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant


# ═══════════════════════════════════════════════════════════════════════════
# 1. CNN base para CIFAR-10
# ═══════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """CNN compacta para CIFAR-10 (≈ 1.1 M parâmetros).

    Arquitetura projetada para ciclos de treinamento rápidos em CPU,
    permitindo iterar sobre técnicas de compressão de forma pedagógica.

    Architecture:
        conv1 (3→32) → BN → ReLU → conv2 (32→64) → BN → ReLU → MaxPool →
        conv3 (64→128) → BN → ReLU → MaxPool →
        fc1 (128*8*8 → 256) → ReLU → Dropout → fc2 (256 → 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class StudentCNN(nn.Module):
    """Modelo *student* menor para Knowledge Distillation.

    Conforme Hinton et al. (2015), o student deve ter capacidade
    suficiente para capturar o *dark knowledge* do teacher, mas
    ser substancialmente menor (≈ 4× menos parâmetros).

    Architecture:
        conv1 (3→16) → BN → ReLU → MaxPool →
        conv2 (16→32) → BN → ReLU → MaxPool →
        fc1 (32*8*8 → 64) → ReLU → fc2 (64 → 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 2. Funções de pruning
# ═══════════════════════════════════════════════════════════════════════════

def apply_unstructured_pruning(
    model: nn.Module,
    amount: float = 0.3,
    prune_type: str = "l1",
) -> nn.Module:
    """Aplica pruning não-estruturado baseado em magnitude.

    Implementa o paradigma *magnitude-based pruning* de Han et al. (2015),
    removendo pesos individuais com menor magnitude absoluta. Conforme a
    Tabela 1 do documento 04, esta estratégia atinge taxas de compressão
    de 80-95 % com alta preservação de acurácia.

    Args:
        model: Modelo PyTorch.
        amount: Fração de pesos a remover (0, 1).
        prune_type: ``"l1"`` (magnitude) ou ``"random"``.

    Returns:
        Modelo com máscaras de pruning aplicadas.
    """
    prune_fn = prune.l1_unstructured if prune_type == "l1" else prune.random_unstructured

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune_fn(module, name="weight", amount=amount)
    return model


def apply_structured_pruning(
    model: nn.Module,
    amount: float = 0.3,
    n: int = 2,
    dim: int = 0,
) -> nn.Module:
    """Aplica pruning estruturado (filtros/neurônios inteiros).

    Conforme discutido na seção *Saiba Mais* do documento 04, o pruning
    estruturado remove estruturas completas (filtros em CNNs, neurônios em
    camadas lineares), resultando em arquiteturas menores que podem ser
    executadas eficientemente em hardware padrão — sem suporte especializado
    para esparsidade.

    Args:
        model: Modelo PyTorch.
        amount: Fração de filtros/neurônios a remover.
        n: Norma Ln usada para ordenar importância.
        dim: Dimensão ao longo da qual podar (0 = filtros de saída).

    Returns:
        Modelo com pruning estruturado aplicado.
    """
    for _name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
    return model


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    """Remove reparametrizações de pruning, consolidando as máscaras.

    Após fine-tuning, é necessário "permanentizar" a esparsidade para que
    o modelo possa ser exportado / quantizado sem as hooks de pruning.

    Args:
        model: Modelo com pruning aplicado.

    Returns:
        Modelo com pesos permanentes (sem hooks).
    """
    for _name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # Camada sem pruning — ignora
    return model


def compute_sparsity(model: nn.Module) -> dict[str, float]:
    """Calcula a esparsidade (% de zeros) por camada e global.

    Útil para verificar a Hipótese da Lottery Ticket (Frankle & Carlin,
    2019): sub-redes com 10-20 % dos parâmetros originais podem igualar
    a acurácia da rede densa.

    Args:
        model: Modelo (com ou sem pruning).

    Returns:
        Dicionário ``{"layer_name": sparsity_pct, ..., "global": pct}``.
    """
    total_zeros = 0
    total_params = 0
    sparsity: dict[str, float] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            n_zeros = int(torch.sum(weight == 0).item())
            n_elem = weight.nelement()
            total_zeros += n_zeros
            total_params += n_elem
            sparsity[name] = 100.0 * n_zeros / n_elem

    sparsity["global"] = 100.0 * total_zeros / total_params if total_params > 0 else 0.0
    return sparsity


# ═══════════════════════════════════════════════════════════════════════════
# 3. Funções de quantização
# ═══════════════════════════════════════════════════════════════════════════

def quantize_dynamic(
    model: nn.Module,
    layers: Optional[set[type]] = None,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Quantização dinâmica INT8 (pesos estáticos, ativações dinâmicas).

    Implementa o paradigma de quantização dinâmica discutido na seção *Saiba
    Mais* do documento 04, convertendo camadas lineares para inteiros de 8
    bits em tempo de inferência. Conforme Jacob et al. (2018), a quantização
    INT8 reduz o tamanho do modelo em ~4× com degradação < 1 % para
    arquiteturas bem comportadas.

    Args:
        model: Modelo FP32 treinado.
        layers: Tipos de camadas a quantizar. Padrão: ``{nn.Linear}``.
        dtype: Tipo de dado alvo (``torch.qint8``).

    Returns:
        Modelo quantizado.
    """
    if layers is None:
        layers = {nn.Linear}
    model_q = quant.quantize_dynamic(model, layers, dtype=dtype)
    return model_q


def prepare_static_quantization(model: nn.Module) -> nn.Module:
    """Prepara modelo para quantização estática (insere observers).

    A quantização estática necessita de uma etapa de *calibração*, onde
    os observers coletam estatísticas (min/max) das ativações a partir
    de um subconjunto representativo dos dados, conforme a seção *Saiba
    Mais* do documento 04.

    Args:
        model: Modelo FP32.

    Returns:
        Modelo com ``QuantStub`` / ``DeQuantStub`` e observers configurados.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy.qconfig = quant.get_default_qconfig("x86")
    quant.prepare(model_copy, inplace=True)
    return model_copy


def calibrate_and_convert(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Calibra observers e converte modelo para INT8.

    Executa dados de calibração pelo modelo preparado para coletar
    estatísticas de ativação, e em seguida converte para INT8.

    Args:
        model: Modelo preparado com observers (``prepare_static_quantization``).
        calibration_loader: DataLoader de calibração.
        device: Dispositivo para calibração.

    Returns:
        Modelo INT8 convertido.
    """
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            model(images)
    quant.convert(model, inplace=True)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. Helpers
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Retorna tamanho aproximado do modelo em MB (pesos em memória).

    Conforme a Tabela 2 do documento 04, modelos INT8 ocupam ~4× menos
    memória que FP32.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)
