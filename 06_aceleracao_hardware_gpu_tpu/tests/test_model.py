"""
Testes unitários para o módulo de modelo (SimpleViT).

Valida inicialização, forward pass, gradient checkpointing, e
métricas de memória do Vision Transformer usado como caso de estudo
na Aula 06 de Aceleração com Hardware.

Referências testadas:
    Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import SimpleViT, PatchEmbedding, MultiHeadSelfAttention, TransformerBlock, create_model


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def small_model():
    """Cria modelo ViT pequeno para testes rápidos."""
    return create_model(
        img_size=8, patch_size=2, in_channels=3,
        n_classes=5, embed_dim=32, depth=2,
        n_heads=2, use_checkpointing=False,
    )


@pytest.fixture
def sample_input():
    """Tensor de entrada de exemplo: (B=4, C=3, H=8, W=8)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 8, 8)


@pytest.fixture
def sample_labels():
    """Labels de exemplo."""
    return torch.randint(0, 5, (4,))


# -----------------------------------------------------------------------
# Testes de PatchEmbedding
# -----------------------------------------------------------------------

class TestPatchEmbedding:
    """Testes do módulo PatchEmbedding."""

    def test_patch_embedding_output_shape(self):
        """Verifica shape de saída: (B, n_patches, embed_dim)."""
        pe = PatchEmbedding(img_size=8, patch_size=2, in_channels=3, embed_dim=32)
        x = torch.randn(4, 3, 8, 8)
        out = pe(x)
        # 8/2 = 4 patches por dimensão, 4*4 = 16 patches total
        assert out.shape == (4, 16, 32)

    def test_patch_embedding_n_patches(self):
        """Verifica cálculo correto de número de patches."""
        pe = PatchEmbedding(img_size=32, patch_size=4)
        assert pe.n_patches == 64  # (32/4)^2


# -----------------------------------------------------------------------
# Testes de MultiHeadSelfAttention
# -----------------------------------------------------------------------

class TestAttention:
    """Testes do módulo de atenção multi-cabeça."""

    def test_attention_output_shape(self):
        """Verifica shape de saída igual à entrada."""
        attn = MultiHeadSelfAttention(embed_dim=32, n_heads=4)
        x = torch.randn(4, 16, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_invalid_heads(self):
        """Verifica erro para embed_dim não divisível por n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=32, n_heads=3)


# -----------------------------------------------------------------------
# Testes de SimpleViT
# -----------------------------------------------------------------------

class TestSimpleViT:
    """Testes do Vision Transformer completo."""

    def test_forward_output_shape(self, small_model, sample_input):
        """Verifica shape de saída: (B, n_classes)."""
        out = small_model(sample_input)
        assert out.shape == (4, 5)

    def test_forward_dtype(self, small_model, sample_input):
        """Verifica dtype de saída float32."""
        out = small_model(sample_input)
        assert out.dtype == torch.float32

    def test_count_parameters(self, small_model):
        """Verifica que contagem de parâmetros é positiva."""
        n_params = small_model.count_parameters()
        assert n_params > 0
        # Modelo pequeno deve ter < 1M parâmetros
        assert n_params < 1_000_000

    def test_get_memory_footprint(self, small_model):
        """Verifica footprint de memória."""
        mem = small_model.get_memory_footprint()
        assert mem["params_count"] > 0
        assert mem["fp32_mb"] > 0
        assert mem["fp16_mb"] == pytest.approx(mem["fp32_mb"] / 2, rel=0.01)
        assert mem["int8_mb"] == pytest.approx(mem["fp32_mb"] / 4, rel=0.01)

    def test_predict(self, small_model, sample_input):
        """Verifica predict retorna classes (inteiros 0..n_classes-1)."""
        preds = small_model.predict(sample_input)
        assert preds.shape == (4,)
        assert preds.min() >= 0
        assert preds.max() < 5

    def test_gradient_checkpointing_same_output(self, sample_input):
        """Verifica que checkpointing não altera saída do forward pass."""
        torch.manual_seed(42)
        model_no_ckpt = create_model(
            img_size=8, patch_size=2, embed_dim=32, depth=2,
            n_heads=2, n_classes=5, use_checkpointing=False,
        )
        torch.manual_seed(42)
        model_ckpt = create_model(
            img_size=8, patch_size=2, embed_dim=32, depth=2,
            n_heads=2, n_classes=5, use_checkpointing=True,
        )
        # Copiar pesos
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())

        model_no_ckpt.eval()
        model_ckpt.eval()

        with torch.no_grad():
            out1 = model_no_ckpt(sample_input)
            out2 = model_ckpt(sample_input)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_gradient_checkpointing_training(self, sample_input, sample_labels):
        """Verifica que gradient checkpointing funciona em modo treino."""
        model = create_model(
            img_size=8, patch_size=2, embed_dim=32, depth=2,
            n_heads=2, n_classes=5, use_checkpointing=True,
        )
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        out = model(sample_input)
        loss = criterion(out, sample_labels)
        loss.backward()
        optimizer.step()

        # Verificar que gradientes foram computados
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_create_model_factory(self):
        """Verifica factory function create_model."""
        model = create_model(img_size=16, patch_size=4, n_classes=3)
        assert isinstance(model, SimpleViT)
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 3)

    def test_fit_method(self, small_model, sample_input, sample_labels):
        """Verifica método fit simplificado."""
        from src.data_preprocessing import SyntheticImageDataset
        from torch.utils.data import DataLoader

        features = sample_input.numpy().reshape(4, -1)
        dataset = SyntheticImageDataset(features, sample_labels.numpy(), img_size=8, n_channels=3)
        loader = DataLoader(dataset, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        losses = small_model.fit(loader, optimizer, criterion, torch.device("cpu"), n_epochs=2)
        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)

    def test_score_method(self, small_model):
        """Verifica método score retorna valor no range [0, 1]."""
        from src.data_preprocessing import SyntheticImageDataset
        from torch.utils.data import DataLoader

        features = np.random.rand(20, 192).astype(np.float32)
        labels = np.random.randint(0, 5, 20).astype(np.int64)
        dataset = SyntheticImageDataset(features, labels, img_size=8, n_channels=3)
        loader = DataLoader(dataset, batch_size=10)

        acc = small_model.score(loader, torch.device("cpu"))
        assert 0.0 <= acc <= 1.0
