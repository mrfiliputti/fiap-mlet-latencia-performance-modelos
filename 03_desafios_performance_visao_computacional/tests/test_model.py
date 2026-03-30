"""
Testes unitários para o módulo model.

Verifica cálculos de FLOPs, contagem de parâmetros, modelos customizados
e wrapper de benchmark com timm.
"""

import numpy as np
import pytest
import torch

from src.model import (
    SimpleCNN,
    DepthwiseSeparableCNN,
    VisionBenchmarkModel,
    calculate_attention_flops,
    calculate_conv_flops,
    calculate_depthwise_separable_reduction,
    calculate_efficientnet_scaling,
    calculate_n_patches,
    calculate_receptive_field,
)


# ---------------------------------------------------------------------------
# Testes de cálculo de FLOPs
# ---------------------------------------------------------------------------


class TestCalculateConvFlops:
    """Testes para FLOPs_conv = 2 × K² × C_in × C_out × H_out × W_out."""

    def test_known_value_3x3(self) -> None:
        """Conv 3×3 com 64 → 128 canais, saída 56×56."""
        expected = 2 * 9 * 64 * 128 * 56 * 56
        result = calculate_conv_flops(k=3, c_in=64, c_out=128, h_out=56, w_out=56)
        assert result == expected

    def test_known_value_1x1(self) -> None:
        """Conv 1×1 pointwise."""
        expected = 2 * 1 * 256 * 512 * 28 * 28
        result = calculate_conv_flops(k=1, c_in=256, c_out=512, h_out=28, w_out=28)
        assert result == expected

    def test_symmetry(self) -> None:
        """FLOPs crescem linearmente com C_out."""
        f1 = calculate_conv_flops(3, 64, 64, 56, 56)
        f2 = calculate_conv_flops(3, 64, 128, 56, 56)
        assert abs(f2 / f1 - 2.0) < 1e-6

    def test_quadratic_with_kernel(self) -> None:
        """FLOPs crescem quadraticamente com K."""
        f3 = calculate_conv_flops(3, 64, 64, 56, 56)
        f7 = calculate_conv_flops(7, 64, 64, 56, 56)
        ratio = f7 / f3
        expected_ratio = 49 / 9
        assert abs(ratio - expected_ratio) < 1e-6


class TestCalculateAttentionFlops:
    """Testes para FLOPs_attn = 4nd² + 2n²d."""

    def test_known_value(self) -> None:
        """ViT-Base: 196 patches, d=768."""
        n, d = 196, 768
        expected = 4 * n * d**2 + 2 * n**2 * d
        result = calculate_attention_flops(n, d)
        assert result == expected

    def test_quadratic_with_patches(self) -> None:
        """Dobrar patches mais que dobra FLOPs (componente n²d)."""
        f1 = calculate_attention_flops(196, 768)
        f2 = calculate_attention_flops(392, 768)
        assert f2 > 2 * f1


class TestCalculateDepthwiseSeparableReduction:
    """Testes para redução 1/C_out + 1/K² (Howard et al., 2017)."""

    def test_mobilenet_values(self) -> None:
        """Caso clássico MobileNet: K=3, C_out=128."""
        reduction = calculate_depthwise_separable_reduction(c_out=128, k=3)
        expected = 1.0 / 128 + 1.0 / 9
        assert abs(reduction - expected) < 1e-6

    def test_large_c_out_approximation(self) -> None:
        """Para C_out grande, redução ≈ 1/K²."""
        reduction = calculate_depthwise_separable_reduction(c_out=10000, k=3)
        assert abs(reduction - 1.0 / 9) < 0.01

    def test_k1_means_no_spatial_saving(self) -> None:
        """K=1 sem economia espacial, redução ≈ 1/C_out + 1."""
        reduction = calculate_depthwise_separable_reduction(c_out=64, k=1)
        expected = 1.0 / 64 + 1.0
        assert abs(reduction - expected) < 1e-6


class TestCalculateNPatches:
    """Testes para n = (H × W) / P²."""

    def test_vit_base_224(self) -> None:
        """ViT-Base com imagem 224×224 e patch 16 → 196 patches."""
        assert calculate_n_patches(224, 16) == 196

    def test_vit_384(self) -> None:
        """Imagem 384×384 com patch 16 → 576 patches."""
        assert calculate_n_patches(384, 16) == 576


class TestCalculateReceptiveField:
    """Testes para campo receptivo."""

    def test_single_layer(self) -> None:
        """Uma conv 3×3 stride 1: RF = 3."""
        kernels = [3]
        strides = [1]
        assert calculate_receptive_field(kernels, strides) == 3

    def test_two_layers(self) -> None:
        """Duas conv 3×3 stride 1: RF = 5."""
        kernels = [3, 3]
        strides = [1, 1]
        assert calculate_receptive_field(kernels, strides) == 5

    def test_with_stride(self) -> None:
        """Conv 3×3 stride 2 + conv 3×3 stride 1: RF = 7."""
        kernels = [3, 3]
        strides = [2, 1]
        assert calculate_receptive_field(kernels, strides) == 7


class TestCalculateEfficientNetScaling:
    """Testes para compound scaling (Tan & Le, 2019)."""

    def test_b0_is_baseline(self) -> None:
        """φ=0 retorna (1.0, 1.0, 1.0)."""
        d, w, r = calculate_efficientnet_scaling(phi=0)
        assert abs(d - 1.0) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(r - 1.0) < 1e-6

    def test_phi1_scales_up(self) -> None:
        """φ=1 gera escalas > 1."""
        d, w, r = calculate_efficientnet_scaling(phi=1)
        assert d > 1.0
        assert w > 1.0
        assert r > 1.0


# ---------------------------------------------------------------------------
# Testes de modelos
# ---------------------------------------------------------------------------


class TestSimpleCNN:
    """Testes para o modelo SimpleCNN."""

    def test_forward_shape(self) -> None:
        """Saída tem forma (batch, n_classes)."""
        model = SimpleCNN(n_classes=10, in_channels=3)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_parametrize_classes(self) -> None:
        """n_classes configurável."""
        model = SimpleCNN(n_classes=5)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 5)


class TestDepthwiseSeparableCNN:
    """Testes para o modelo MobileNet-style."""

    def test_forward_shape(self) -> None:
        """Saída tem forma (batch, n_classes)."""
        model = DepthwiseSeparableCNN(n_classes=10, in_channels=3)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_fewer_params_than_simple(self) -> None:
        """Depthwise separable tem menos parâmetros que CNN simples."""
        simple = SimpleCNN(n_classes=10)
        dw = DepthwiseSeparableCNN(n_classes=10)
        p_simple = sum(p.numel() for p in simple.parameters())
        p_dw = sum(p.numel() for p in dw.parameters())
        assert p_dw < p_simple


class TestVisionBenchmarkModel:
    """Testes para o wrapper de benchmark."""

    def test_simple_cnn_creation(self) -> None:
        """Cria modelo simple_cnn."""
        vb = VisionBenchmarkModel("simple_cnn", pretrained=False)
        assert vb.model is not None

    def test_mobilenet_style_creation(self) -> None:
        """Cria modelo mobilenet_style."""
        vb = VisionBenchmarkModel("mobilenet_style", pretrained=False)
        assert vb.model is not None

    def test_count_parameters(self) -> None:
        """count_parameters retorna inteiro positivo."""
        vb = VisionBenchmarkModel("simple_cnn", pretrained=False)
        params = vb.count_parameters()
        assert isinstance(params, int)
        assert params > 0

    def test_estimate_flops(self) -> None:
        """estimate_flops retorna float positivo."""
        vb = VisionBenchmarkModel("simple_cnn", pretrained=False)
        flops = vb.estimate_flops(input_size=(3, 64, 64))
        assert flops > 0

    def test_benchmark_latency(self) -> None:
        """benchmark_latency retorna dicionário com estatísticas."""
        vb = VisionBenchmarkModel("simple_cnn", pretrained=False)
        stats = vb.benchmark_latency(input_size=(3, 32, 32), n_runs=5, warmup=1)
        assert "mean_ms" in stats
        assert "p50_ms" in stats
        assert stats["mean_ms"] > 0
