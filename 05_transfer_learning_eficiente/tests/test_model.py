"""
Testes unitários para model.py.

Valida o PEFTModelWrapper: configuração LoRA, contagem de parâmetros
e consistência da API.

Nota: testes projetados para executar em CPU sem download de modelos
grandes. Usam mocks para AutoModel quando necessário.
"""

import pytest
import torch
import torch.nn as nn

from src.model import LoRAConfig, QLoRAConfig, PEFTModelWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lora_config() -> LoRAConfig:
    """Configuração LoRA padrão."""
    return LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        task_type="SEQ_CLS",
    )


@pytest.fixture
def qlora_config() -> QLoRAConfig:
    """Configuração QLoRA padrão."""
    return QLoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )


@pytest.fixture
def wrapper() -> PEFTModelWrapper:
    """PEFTModelWrapper com modelo DistilBERT."""
    return PEFTModelWrapper(
        model_name="distilbert-base-uncased",
        num_labels=5,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Testes de configuração
# ---------------------------------------------------------------------------


class TestLoRAConfig:
    """Testes das dataclasses de configuração."""

    def test_lora_config_defaults(self, lora_config: LoRAConfig):
        """Verifica valores padrão da configuração LoRA."""
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        assert lora_config.task_type == "SEQ_CLS"

    def test_qlora_extends_lora(self, qlora_config: QLoRAConfig):
        """Verifica que QLoRAConfig herda de LoRAConfig e adiciona campos."""
        assert isinstance(qlora_config, LoRAConfig)
        assert qlora_config.load_in_4bit is True
        assert qlora_config.bnb_4bit_quant_type == "nf4"
        assert qlora_config.bnb_4bit_compute_dtype == "bfloat16"
        assert qlora_config.bnb_4bit_use_double_quant is True

    def test_qlora_has_lora_fields(self, qlora_config: QLoRAConfig):
        """Verifica que QLoRA preserva campos LoRA."""
        assert qlora_config.r == 16
        assert qlora_config.lora_alpha == 32


# ---------------------------------------------------------------------------
# Testes do PEFTModelWrapper
# ---------------------------------------------------------------------------


class TestPEFTModelWrapper:
    """Testes do wrapper de modelo PEFT."""

    def test_init(self, wrapper: PEFTModelWrapper):
        """Verifica inicialização do wrapper."""
        assert wrapper.model_name == "distilbert-base-uncased"
        assert wrapper.num_labels == 5
        assert wrapper.device == "cpu"
        assert wrapper.base_model is None

    def test_repr(self, wrapper: PEFTModelWrapper):
        """Verifica representação string."""
        r = repr(wrapper)
        assert "distilbert-base-uncased" in r
        assert "sem PEFT" in r

    def test_resolve_device_cpu(self):
        """Verifica resolução de device para 'cpu'."""
        assert PEFTModelWrapper._resolve_device("cpu") == "cpu"

    def test_model_property_raises_without_load(self, wrapper: PEFTModelWrapper):
        """Verifica que acessar model sem carregar gera erro."""
        with pytest.raises(RuntimeError, match="Nenhum modelo carregado"):
            _ = wrapper.model

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA não disponível — testa apenas com GPU",
    )
    def test_resolve_device_auto_cuda(self):
        """Verifica que 'auto' resolve para CUDA quando disponível."""
        assert PEFTModelWrapper._resolve_device("auto") == "cuda"

    def test_load_base_model(self, wrapper: PEFTModelWrapper):
        """Verifica carregamento do modelo base."""
        model = wrapper.load_base_model(quantize_4bit=False)
        assert model is not None
        assert wrapper.tokenizer is not None
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_apply_lora_reduces_trainable(self, wrapper: PEFTModelWrapper):
        """Verifica que LoRA reduz parâmetros treináveis."""
        wrapper.load_base_model(quantize_4bit=False)
        base_trainable = sum(
            p.numel() for p in wrapper.base_model.parameters() if p.requires_grad
        )

        peft_model = wrapper.apply_lora(r=8, lora_alpha=16)
        peft_trainable = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )

        # LoRA deve ter MENOS parâmetros treináveis que o modelo completo
        assert peft_trainable < base_trainable

    def test_apply_lora_without_base_raises(self, wrapper: PEFTModelWrapper):
        """Verifica erro ao aplicar LoRA sem modelo base carregado."""
        with pytest.raises(RuntimeError, match="Modelo base não carregado"):
            wrapper.apply_lora()

    def test_count_parameters_after_lora(self, wrapper: PEFTModelWrapper):
        """Verifica contagem de parâmetros após aplicar LoRA."""
        wrapper.load_base_model(quantize_4bit=False)
        wrapper.apply_lora(r=8, lora_alpha=16)
        stats = wrapper.count_parameters()
        assert stats["trainable"] > 0
        assert stats["frozen"] > 0
        assert stats["trainable_pct"] < 10.0  # Menos de 10% treináveis

    def test_model_property_returns_peft(self, wrapper: PEFTModelWrapper):
        """Verifica que .model retorna PEFT model quando aplicado."""
        wrapper.load_base_model(quantize_4bit=False)
        wrapper.apply_lora(r=4)
        assert wrapper.model is wrapper.peft_model

    def test_default_target_modules(self):
        """Verifica módulos alvo padrão para diferentes arquiteturas."""
        w1 = PEFTModelWrapper(model_name="distilbert-base-uncased", num_labels=2)
        assert "q_lin" in w1._get_default_target_modules()

        w2 = PEFTModelWrapper(model_name="bert-base-uncased", num_labels=2)
        assert "query" in w2._get_default_target_modules()
