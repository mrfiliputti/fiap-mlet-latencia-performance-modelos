"""
Módulo de modelos de visão computacional com cálculo de FLOPs.

Implementa o cálculo teórico de FLOPs para convoluções e atenção conforme
as fórmulas da seção 'Saiba Mais', além de modelos CNN e ViT pedagógicos
para demonstração de conceitos como depthwise separable convolutions
(Howard et al., 2017) e patch embedding (Dosovitskiy et al., 2021).

Referência:
    He, K. & Sun, J. (2015). CNNs at Constrained Time Cost. CVPR 2015.
    Howard, A. G. et al. (2017). MobileNets: Efficient CNNs. arXiv:1704.04861.
    Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words. ICLR 2021.
    Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals. CVPR 2018.
    Tan, M. & Le, Q. V. (2019). EfficientNet. ICML 2019.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================================
# Cálculos teóricos de FLOPs — seções 1 e 3 do material
# =========================================================================


def calculate_conv_flops(
    kernel_size: int,
    c_in: int,
    c_out: int,
    h_out: int,
    w_out: int,
) -> int:
    """
    Calcula FLOPs de uma camada convolucional padrão.

    Implementa a fórmula da seção 1 ('Fundamentos de Complexidade
    Computacional em Visão Computacional'):

        FLOPs_conv = 2 × K² × C_in × C_out × H_out × W_out

    O fator 2 contabiliza multiplicações e adições conforme
    He & Sun (2015).

    Args:
        kernel_size: Tamanho do kernel (K).
        c_in: Número de canais de entrada (C_in).
        c_out: Número de canais de saída (C_out).
        h_out: Altura do feature map de saída (H_out).
        w_out: Largura do feature map de saída (W_out).

    Returns:
        Número total de FLOPs (inteiro).
    """
    return 2 * (kernel_size ** 2) * c_in * c_out * h_out * w_out


def calculate_attention_flops(n_patches: int, d_model: int) -> int:
    """
    Calcula FLOPs do mecanismo de self-attention em Vision Transformers.

    Implementa a fórmula da seção 1:

        FLOPs_attention = 4 × n × d² + 2 × n² × d

    onde n = número de patches e d = dimensionalidade do embedding.
    O termo n² evidencia a complexidade quadrática que limita a
    escalabilidade de ViTs para resoluções elevadas
    (Dosovitskiy et al., 2021).

    Args:
        n_patches: Número de patches da imagem (n).
        d_model: Dimensionalidade do embedding (d).

    Returns:
        Número total de FLOPs (inteiro).
    """
    return 4 * n_patches * (d_model ** 2) + 2 * (n_patches ** 2) * d_model


def calculate_depthwise_separable_reduction(c_out: int, kernel_size: int) -> float:
    """
    Calcula fator de redução de FLOPs com depthwise separable convolutions.

    Implementa a fórmula de redução da seção 2 ('Arquiteturas CNN Eficientes'):

        Redução = 1/C_out + 1/K²

    Para K=3, redução ≈ 1/C_out + 1/9, resultando em speedup de ~8-9x
    conforme Howard et al. (2017).

    Args:
        c_out: Número de canais de saída.
        kernel_size: Tamanho do kernel.

    Returns:
        Fator de redução (entre 0 e 1; menor = mais eficiente).
    """
    return (1.0 / c_out) + (1.0 / (kernel_size ** 2))


def calculate_n_patches(image_size: int, patch_size: int = 16) -> int:
    """
    Calcula número de patches para Vision Transformer.

    Uma imagem de tamanho (H, W) com patches de tamanho (P, P) produz
    n = (H/P) × (W/P) patches (Dosovitskiy et al., 2021).

    Args:
        image_size: Resolução da imagem (assumindo quadrada).
        patch_size: Tamanho do patch (default: 16, padrão ViT).

    Returns:
        Número de patches.
    """
    return (image_size // patch_size) ** 2


def calculate_receptive_field(
    kernel_sizes: list[int],
    strides: list[int],
) -> int:
    """
    Calcula campo receptivo efetivo para sequência de camadas convolucionais.

    Implementa a fórmula da seção 9 ('Campos Receptivos'):

        RF_l = RF_{l-1} + (k_l - 1) × Π_{i=1}^{l-1} s_i

    O campo receptivo cresce linearmente com profundidade para stride=1,
    mas strides > 1 amplificam o crescimento (Lin et al., 2017).

    Args:
        kernel_sizes: Lista de tamanhos de kernel por camada.
        strides: Lista de strides por camada.

    Returns:
        Campo receptivo efetivo (em pixels).

    Raises:
        ValueError: Se listas têm tamanhos diferentes.
    """
    if len(kernel_sizes) != len(strides):
        raise ValueError("kernel_sizes e strides devem ter mesmo tamanho.")

    rf = 1
    for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
        stride_product = 1
        for j in range(i):
            stride_product *= strides[j]
        rf += (k - 1) * stride_product

    return rf


def calculate_efficientnet_scaling(
    phi: float,
    alpha: float = 1.2,
    beta: float = 1.1,
    gamma: float = 1.15,
) -> dict[str, float]:
    """
    Calcula fatores de compound scaling do EfficientNet.

    Implementa as fórmulas da seção 2:

        depth  = α^φ
        width  = β^φ
        resolution = γ^φ

    Com restrição α × β² × γ² ≈ 2, onde φ controla os recursos
    disponíveis (Tan & Le, 2019).

    Args:
        phi: Coeficiente de scaling definido pelo usuário.
        alpha: Fator de profundidade (default: 1.2).
        beta: Fator de largura (default: 1.1).
        gamma: Fator de resolução (default: 1.15).

    Returns:
        Dicionário com fatores 'depth', 'width', 'resolution' e 'constraint'.
    """
    depth = alpha ** phi
    width = beta ** phi
    resolution = gamma ** phi
    constraint = alpha * (beta ** 2) * (gamma ** 2)

    return {
        "depth": round(depth, 4),
        "width": round(width, 4),
        "resolution": round(resolution, 4),
        "constraint_value": round(constraint, 4),
        "phi": phi,
    }


# =========================================================================
# Modelos pedagógicos — CNN simples
# =========================================================================


class SimpleCNN(nn.Module):
    """
    CNN simples com convoluções padrão para uso como baseline.

    Implementa uma CNN convencional com convoluções 3×3 padrão para
    comparação com arquiteturas eficientes. Conforme He & Sun (2015),
    a complexidade de cada camada é O(K² × C_in × C_out × H × W).

    Attributes:
        n_classes: Número de classes de saída.
    """

    def __init__(self, n_classes: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: 3×3 conv padrão
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================================================================
# Modelo com Depthwise Separable Convolutions — seção 2
# =========================================================================


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution conforme Howard et al. (2017).

    Decompõe convolução padrão em depthwise (filtro por canal) +
    pointwise (1×1), reduzindo FLOPs por fator de ~8-9x para K=3.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DepthwiseSeparableCNN(nn.Module):
    """
    CNN eficiente estilo MobileNet com depthwise separable convolutions.

    Implementa conceitos da seção 2 ('Arquiteturas CNN Eficientes'):
    substitui convoluções padrão por depthwise separable convolutions
    conforme Howard et al. (2017), reduzindo FLOPs em ~8x.

    Attributes:
        n_classes: Número de classes de saída.
    """

    def __init__(self, n_classes: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        # Primeira camada: convolução padrão (mesma prática do MobileNet)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Blocos depthwise separable
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================================================================
# Wrapper para benchmarking com timm — seção 10 (fronteira de Pareto)
# =========================================================================


class VisionBenchmarkModel:
    """
    Wrapper para benchmarking de modelos de visão computacional.

    Encapsula modelos do timm ou modelos customizados para medição
    padronizada de latência, FLOPs e throughput, conforme metodologia
    de benchmark estabelecida por Reddi et al. (2020) e aplicada na
    análise de fronteira de Pareto (seção 10 do material).

    Attributes:
        model_name: Nome do modelo.
        model: Instância do modelo PyTorch.
        device: Dispositivo de execução.
        is_timm: Se o modelo vem da biblioteca timm.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = False,
        n_classes: int = 4,
        device: str = "cpu",
    ) -> None:
        """
        Inicializa modelo para benchmarking.

        Args:
            model_name: Nome do modelo. Pode ser um nome do timm
                (e.g., 'efficientnet_b0', 'deit_small_patch16_224')
                ou 'simple_cnn' / 'mobilenet_style' para modelos pedagógicos.
            pretrained: Se True, carrega pesos pré-treinados (somente timm).
            n_classes: Número de classes de saída.
            device: Dispositivo ('cpu', 'cuda', 'mps').
        """
        self.model_name = model_name
        self.device = device
        self.is_timm = False
        self.latency_history: list[dict[str, Any]] = []

        if model_name == "simple_cnn":
            self.model = SimpleCNN(n_classes=n_classes)
        elif model_name == "mobilenet_style":
            self.model = DepthwiseSeparableCNN(n_classes=n_classes)
        else:
            try:
                import timm as timm_lib
                self.model = timm_lib.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=n_classes,
                )
                self.is_timm = True
            except (ImportError, RuntimeError) as e:
                raise ValueError(
                    f"Modelo '{model_name}' não encontrado. "
                    f"Instale timm ou use 'simple_cnn'/'mobilenet_style'. "
                    f"Erro: {e}"
                )

        self.model = self.model.to(device).eval()

    def count_parameters(self) -> dict[str, int]:
        """
        Conta parâmetros do modelo.

        Returns:
            Dicionário com 'total', 'trainable' e 'non_trainable'.
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "total_millions": round(total / 1e6, 2),
        }

    def estimate_flops(self, input_size: int = 224) -> dict[str, float]:
        """
        Estima FLOPs por forward pass (aproximação por camadas).

        Percorre as camadas do modelo e aplica as fórmulas de FLOPs
        da seção 1 do material:
        - Conv2d: 2 × K² × C_in × C_out × H × W
        - Linear: 2 × in_features × out_features

        Args:
            input_size: Resolução de entrada (lado do quadrado).

        Returns:
            Dicionário com 'total_flops', 'total_gflops' e detalhamento.
        """
        total_flops = 0
        layer_details: list[dict[str, Any]] = []
        # Rastrear dimensão espacial simulada
        h, w = input_size, input_size
        c = 3  # canais de entrada

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                k = module.kernel_size[0]
                s = module.stride[0]
                p = module.padding[0]
                c_in = module.in_channels
                c_out = module.out_channels
                groups = module.groups

                h_out = (h + 2 * p - k) // s + 1
                w_out = (w + 2 * p - k) // s + 1

                if groups == 1:
                    flops = calculate_conv_flops(k, c_in, c_out, h_out, w_out)
                else:
                    # Depthwise: FLOPs por grupo
                    flops = 2 * (k ** 2) * (c_in // groups) * c_out * h_out * w_out

                total_flops += flops
                layer_details.append({
                    "name": name,
                    "type": "Conv2d",
                    "flops": flops,
                    "output_size": f"{c_out}×{h_out}×{w_out}",
                })
                h, w, c = h_out, w_out, c_out

            elif isinstance(module, nn.Linear):
                flops = 2 * module.in_features * module.out_features
                total_flops += flops
                layer_details.append({
                    "name": name,
                    "type": "Linear",
                    "flops": flops,
                })

            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                s = module.stride if isinstance(module.stride, int) else module.stride[0]
                h = h // s
                w = w // s

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                out = module.output_size
                if isinstance(out, int):
                    h, w = out, out
                elif out is not None:
                    h = out[0] if out[0] is not None else h
                    w = out[1] if out[1] is not None else w

        return {
            "total_flops": total_flops,
            "total_gflops": round(total_flops / 1e9, 4),
            "total_mflops": round(total_flops / 1e6, 2),
            "n_layers_counted": len(layer_details),
            "layer_details": layer_details,
        }

    @torch.no_grad()
    def benchmark_latency(
        self,
        input_size: int = 224,
        batch_size: int = 1,
        n_warmup: int = 5,
        n_runs: int = 50,
    ) -> dict[str, Any]:
        """
        Mede latência de inferência com warm-up.

        Implementa protocolo de benchmarking conforme MLPerf
        (Reddi et al., 2020): descarta warm-up runs e coleta
        estatísticas com percentis.

        Args:
            input_size: Resolução de entrada.
            batch_size: Tamanho do batch.
            n_warmup: Execuções de warm-up.
            n_runs: Execuções de medição.

        Returns:
            Dicionário com estatísticas de latência e throughput.
        """
        from src.utils import BenchmarkTimer

        x = torch.randn(batch_size, 3, input_size, input_size, device=self.device)

        timer = BenchmarkTimer()
        stats = timer.run(
            lambda: self.model(x),
            n_warmup=n_warmup,
            n_runs=n_runs,
        )

        stats["input_size"] = input_size
        stats["batch_size"] = batch_size
        stats["model_name"] = self.model_name
        stats["device"] = self.device
        stats["throughput_imgs_sec"] = round(
            batch_size / (stats["mean_ms"] / 1000.0), 2
        )

        self.latency_history.append(stats)
        return stats

    @torch.no_grad()
    def resolution_sweep(
        self,
        resolutions: Optional[list[int]] = None,
        n_runs: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Executa sweep de resoluções para análise de impacto.

        Conforme seção 4 do material, dobrar a resolução aumenta FLOPs
        por fator ~4 em CNNs. Este método permite verificar
        empiricamente esta relação.

        Args:
            resolutions: Lista de resoluções a testar.
            n_runs: Execuções por resolução.

        Returns:
            Lista de resultados de benchmark por resolução.
        """
        if resolutions is None:
            resolutions = [128, 224, 320, 384, 512]

        results = []
        for res in resolutions:
            try:
                stats = self.benchmark_latency(
                    input_size=res,
                    n_warmup=3,
                    n_runs=n_runs,
                )
                flops_info = self.estimate_flops(input_size=res)
                stats["flops_giga"] = flops_info["total_gflops"]
                results.append(stats)
                logger.info(
                    "  %s @ %d: %.2f ms (%.2f GFLOPs)",
                    self.model_name, res, stats["mean_ms"], stats["flops_giga"],
                )
            except RuntimeError as e:
                logger.warning("Resolução %d falhou para %s: %s", res, self.model_name, e)

        return results
