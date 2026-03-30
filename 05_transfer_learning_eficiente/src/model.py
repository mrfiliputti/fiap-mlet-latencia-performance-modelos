"""
Definição de modelo com suporte a PEFT (LoRA, QLoRA, Adapters).

Implementa wrapper sobre modelos Hugging Face para fine-tuning eficiente
utilizando Low-Rank Adaptation (LoRA) e QLoRA, conforme formalizado por
Hu et al. (2022) e Dettmers et al. (2023).

A classe ``PEFTModelWrapper`` encapsula:
- Carregamento de modelo base pré-treinado (com ou sem quantização 4-bit)
- Aplicação de adaptadores LoRA com decomposição W = W₀ + BA
- Fusão de pesos para inferência sem overhead de latência
- Exportação ONNX para serving otimizado

Referências:
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
        Language Models. ICLR.
    Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of
        Quantized LLMs. NeurIPS.
    Houlsby, N. et al. (2019). Parameter-Efficient Transfer Learning
        for NLP. ICML.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuração de LoRA
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """
    Configuração para adaptadores Low-Rank Adaptation.

    Implementa os hiperparâmetros definidos por Hu et al. (2022):
    - ``r``:  Rank dos adaptadores (dimensão intrínseca do subespaço).
              Controla trade-off capacidade expressiva vs eficiência.
              Valores típicos: 4–64 (Tabela 2, material da Aula 5).
    - ``lora_alpha``: Fator de escalonamento α. A atualização final é
                      escalada por α/r, modulando a magnitude (Hu et al., 2022).
    - ``target_modules``: Módulos lineares onde injetar adaptadores.
              Para transformers, tipicamente q_proj, v_proj, k_proj, o_proj.
    - ``lora_dropout``: Dropout para regularização dos adaptadores.
    """

    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_lin", "v_lin"]
    )
    lora_dropout: float = 0.05
    task_type: str = "SEQ_CLS"


@dataclass
class QLoRAConfig(LoRAConfig):
    """
    Configuração estendida para QLoRA (LoRA + Quantização 4-bit).

    QLoRA (Dettmers et al., 2023) combina LoRA com quantização NF4
    para reduzir drasticamente requisitos de memória. Os pesos do
    modelo base são quantizados para 4 bits usando o formato
    NormalFloat4, projetado para dados com distribuição normal.

    Parâmetros adicionais:
    - ``load_in_4bit``:  Ativa quantização 4-bit do modelo base.
    - ``bnb_4bit_quant_type``: Tipo de quantização (nf4 ou fp4).
    - ``bnb_4bit_compute_dtype``: Dtype para forward pass (bfloat16
      oferece melhor range dinâmico que float16).
    - ``bnb_4bit_use_double_quant``: Quantização dupla — os parâmetros
      de quantização (escalas) são eles próprios quantizados.
    """

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# ---------------------------------------------------------------------------
# PEFTModelWrapper
# ---------------------------------------------------------------------------


class PEFTModelWrapper:
    """
    Wrapper para fine-tuning eficiente de modelos com LoRA/QLoRA.

    Encapsula o fluxo completo ilustrado na Figura 1 do material: carregar
    modelo base → aplicar adaptadores LoRA → treinar → fundir pesos →
    exportar ONNX.

    O design permite comparar eficiência de parâmetros entre fine-tuning
    completo e PEFT, reproduzindo a Tabela 2 do material da Aula 5.

    Args:
        model_name: Nome do modelo Hugging Face (ex: 'distilbert-base-uncased').
        num_labels: Número de classes de saída para classificação.
        device: Dispositivo ('cpu', 'cuda', 'auto').

    Referências:
        Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation. ICLR.
        Dettmers, T. et al. (2023). QLoRA. NeurIPS.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = self._resolve_device(device)
        self.base_model: Optional[nn.Module] = None
        self.peft_model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self._is_peft_applied: bool = False
        self._lora_config: Optional[LoRAConfig] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ----- Carregamento do modelo base --------------------------------------

    def load_base_model(self, quantize_4bit: bool = False) -> nn.Module:
        """
        Carrega modelo pré-treinado do Hugging Face Hub.

        Quando ``quantize_4bit=True``, aplica quantização NF4 via
        BitsAndBytesConfig conforme Snippet 1 do material (QLoRA,
        Dettmers et al., 2023).

        Args:
            quantize_4bit: Se True, carrega modelo em 4-bit (requer
                bitsandbytes e GPU).

        Returns:
            Modelo carregado (AutoModelForSequenceClassification).
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Carregando modelo base: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs: Dict[str, Any] = {
            "num_labels": self.num_labels,
        }

        # Configuração QLoRA (Snippet 1 da Aula 5)
        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Quantização NF4 4-bit ativada (QLoRA)")
            except ImportError:
                logger.warning(
                    "bitsandbytes não disponível; carregando sem quantização."
                )

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )

        if not quantize_4bit:
            self.base_model = self.base_model.to(self.device)

        logger.info(
            "Modelo carregado no dispositivo: %s (%d parâmetros)",
            self.device,
            sum(p.numel() for p in self.base_model.parameters()),
        )
        return self.base_model

    # ----- Aplicação de LoRA ------------------------------------------------

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
        task_type: str = "SEQ_CLS",
    ) -> nn.Module:
        """
        Aplica adaptadores LoRA ao modelo base.

        Implementa a decomposição W = W₀ + BA (Hu et al., 2022),
        onde B ∈ ℝ^(d×r) e A ∈ ℝ^(r×k) são matrizes treináveis
        de rank r ≪ min(d, k), conforme Snippet 2 do material.

        A inicialização segue a convenção: A ~ N(0, σ²) e B = 0,
        garantindo ΔW = BA = 0 no início do treinamento.

        Args:
            r: Rank dos adaptadores (dimensão intrínseca).
            lora_alpha: Fator de escalonamento α.
            target_modules: Módulos alvo para LoRA. Se None, usa
                defaults para o modelo (q_lin, v_lin para DistilBERT).
            lora_dropout: Dropout dos adaptadores.
            task_type: Tipo de tarefa PEFT ('SEQ_CLS', 'CAUSAL_LM').

        Returns:
            PeftModel com adaptadores aplicados.

        Referência:
            Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
            Language Models. ICLR. Seção 4.1.
        """
        if self.base_model is None:
            raise RuntimeError("Modelo base não carregado. Execute load_base_model().")

        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

        # Mapear task_type string para Enum
        task_type_enum = getattr(TaskType, task_type, TaskType.SEQ_CLS)

        if target_modules is None:
            target_modules = self._get_default_target_modules()

        # Configuração LoRA (Snippet 2 do material da Aula 5)
        peft_config = PeftLoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type=task_type_enum,
        )

        self.peft_model = get_peft_model(self.base_model, peft_config)
        self._is_peft_applied = True
        self._lora_config = LoRAConfig(
            r=r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout, task_type=task_type,
        )

        # Log contagem de parâmetros (Tabela 2 do material)
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(
            "LoRA aplicado (r=%d): %d treináveis / %d total (%.4f%%)",
            r, trainable, total, 100.0 * trainable / total,
        )
        return self.peft_model

    def _get_default_target_modules(self) -> List[str]:
        """Retorna módulos alvo padrão baseado na arquitetura do modelo."""
        name_lower = self.model_name.lower()
        if "distilbert" in name_lower:
            return ["q_lin", "v_lin"]
        if "bert" in name_lower:
            return ["query", "value"]
        if "gpt2" in name_lower or "gpt" in name_lower:
            return ["c_attn"]
        if "llama" in name_lower:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        # Fallback genérico
        return ["query", "value"]

    # ----- Modelo ativo -----------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """Retorna modelo PEFT se aplicado, caso contrário modelo base."""
        if self._is_peft_applied and self.peft_model is not None:
            return self.peft_model
        if self.base_model is not None:
            return self.base_model
        raise RuntimeError("Nenhum modelo carregado.")

    # ----- Contagem de parâmetros -------------------------------------------

    def count_parameters(self) -> Dict[str, Any]:
        """
        Conta parâmetros do modelo atual (base ou PEFT).

        Reproduz análise da Tabela 2 do material: compara parâmetros
        treináveis entre fine-tuning completo e LoRA.

        Returns:
            Dicionário com total, trainable, frozen, trainable_pct.
        """
        from src.utils import count_parameters
        return count_parameters(self.model)

    # ----- Fusão de pesos (merge) -------------------------------------------

    def merge_lora_weights(self) -> nn.Module:
        """
        Funde adaptadores LoRA com pesos base: W = W₀ + BA.

        Conforme discutido na seção 'Low-Rank Adaptation (LoRA)' do material,
        após treinamento as matrizes BA podem ser mescladas com W₀ através
        de adição matricial simples. O modelo resultante possui arquitetura
        idêntica ao original, sem latência adicional de inferência.

        Returns:
            Modelo com pesos fundidos.
        """
        if not self._is_peft_applied or self.peft_model is None:
            raise RuntimeError("LoRA não aplicado. Execute apply_lora() primeiro.")

        merged = self.peft_model.merge_and_unload()
        logger.info("Adaptadores LoRA fundidos com pesos base (W = W₀ + BA)")
        return merged

    # ----- Exportação ONNX ---------------------------------------------------

    def export_onnx(
        self,
        output_path: str = "outputs/models/model.onnx",
        opset_version: int = 17,
        max_length: int = 128,
    ) -> str:
        """
        Exporta modelo para formato ONNX com otimizações de grafo.

        Implementa o pipeline de exportação descrito no Snippet 3 do
        material da Aula 5, aplicando constant folding e eixos dinâmicos
        para batch e sequence length variáveis.

        A escolha de opset_version ≥ 17 suporta operadores nativos para
        attention patterns otimizados (seção 'ONNX: Interoperabilidade' do material).

        Args:
            output_path: Caminho destino para o arquivo .onnx.
            opset_version: Versão do opset ONNX (≥17 recomendado).
            max_length: Comprimento máximo de sequência para input de exemplo.

        Returns:
            Caminho do arquivo ONNX salvo.

        Referência:
            Material da Aula 5, seção 'ONNX: Interoperabilidade e
            Otimização Cross-Platform'.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        model = self.model
        model.eval()

        # Input de exemplo para tracing
        dummy_input_ids = torch.randint(
            0, 1000, (1, max_length), device=self.device
        )
        dummy_attention_mask = torch.ones(
            (1, max_length), dtype=torch.long, device=self.device
        )

        # Exportação ONNX (Snippet 3 do material)
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )
        logger.info("Modelo exportado para ONNX: %s", output_path)
        return output_path

    # ----- Representação ----------------------------------------------------

    def __repr__(self) -> str:
        peft_str = "LoRA aplicado" if self._is_peft_applied else "sem PEFT"
        return (
            f"PEFTModelWrapper(model='{self.model_name}', "
            f"labels={self.num_labels}, {peft_str})"
        )
