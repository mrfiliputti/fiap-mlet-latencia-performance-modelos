"""
Lógica de treinamento para fine-tuning eficiente com PEFT.

Implementa funções de treinamento utilizando Hugging Face Trainer com
suporte a LoRA e QLoRA, conforme descrito nos Snippets 1-2 do material
da Aula 5. Inclui comparação de eficiência entre fine-tuning completo
e Parameter-Efficient Fine-Tuning.

Referências:
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
        Language Models. ICLR.
    Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of
        Quantized LLMs. NeurIPS.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset para Hugging Face Trainer
# ---------------------------------------------------------------------------


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch para classificação textual com Transformer.

    Tokeniza textos on-the-fly e retorna dicionários compatíveis com
    AutoModelForSequenceClassification.

    Args:
        texts: Lista de textos.
        labels: Lista de labels (inteiros).
        tokenizer: Tokenizer Hugging Face.
        max_length: Comprimento máximo de sequência.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = 128,
    text_col: str = "text",
    label_col: str = "label_id",
) -> Tuple[TextClassificationDataset, TextClassificationDataset, TextClassificationDataset]:
    """
    Cria datasets PyTorch a partir de DataFrames.

    Args:
        train_df: DataFrame de treino.
        val_df: DataFrame de validação.
        test_df: DataFrame de teste.
        tokenizer: Tokenizer Hugging Face.
        max_length: Comprimento máximo de tokenização.
        text_col: Nome da coluna de texto.
        label_col: Nome da coluna de label (inteiro).

    Returns:
        Tupla (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = TextClassificationDataset(
        texts=train_df[text_col].tolist(),
        labels=train_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = TextClassificationDataset(
        texts=val_df[text_col].tolist(),
        labels=val_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = TextClassificationDataset(
        texts=test_df[text_col].tolist(),
        labels=test_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return train_dataset, val_dataset, test_dataset


def compute_metrics_fn(eval_pred: Any) -> Dict[str, float]:
    """
    Função de métricas para Hugging Face Trainer.

    Calcula accuracy e macro F1 durante validação.

    Args:
        eval_pred: Objeto EvalPrediction do Trainer.

    Returns:
        Dicionário com accuracy e f1_macro.
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ---------------------------------------------------------------------------
# Treinamento com PEFT
# ---------------------------------------------------------------------------


def train_with_peft(
    model: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str = "outputs/models/lora_adapter",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    fp16: bool = False,
) -> Dict[str, Any]:
    """
    Treina modelo com Hugging Face Trainer (compatível com PEFT).

    Implementa o pipeline de treinamento eficiente conforme discutido
    na seção 'Hands On' do material da Aula 5. Usa learning rate 2e-4
    (padrão recomendado para LoRA) e warmup progressivo para
    estabilidade numérica com quantização (Dettmers et al., 2023).

    Args:
        model: Modelo PyTorch (base ou PEFT).
        train_dataset: Dataset de treino.
        val_dataset: Dataset de validação.
        output_dir: Diretório para checkpoints.
        num_epochs: Número de épocas de treinamento.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado (2e-4 recomendado para LoRA).
        weight_decay: Regularização L2.
        warmup_ratio: Proporção de passos para warmup linear.
        logging_steps: Frequência de logging.
        save_strategy: Estratégia de salvamento ('epoch', 'steps').
        fp16: Usar mixed precision (requer GPU).

    Returns:
        Dicionário com métricas de treinamento e tempo.

    Referência:
        Hu, E.J. et al. (2022). LoRA, Seção 5 (Experimental Setup).
    """
    from transformers import TrainingArguments, Trainer

    os.makedirs(output_dir, exist_ok=True)

    # Configuração de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    logger.info("Iniciando treinamento (%d épocas, lr=%.1e)", num_epochs, learning_rate)
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # Métricas finais
    eval_metrics = trainer.evaluate()

    results = {
        "train_loss": train_result.training_loss,
        "train_runtime_s": elapsed,
        "train_samples_per_second": len(train_dataset) * num_epochs / elapsed,
        "eval_accuracy": eval_metrics.get("eval_accuracy", 0.0),
        "eval_f1_macro": eval_metrics.get("eval_f1_macro", 0.0),
        "eval_loss": eval_metrics.get("eval_loss", 0.0),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }

    logger.info(
        "Treinamento concluído: loss=%.4f, acc=%.4f, f1=%.4f (%.1fs)",
        results["train_loss"],
        results["eval_accuracy"],
        results["eval_f1_macro"],
        elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Comparação de eficiência
# ---------------------------------------------------------------------------


def compare_training_efficiency(
    base_model: Any,
    peft_model: Any,
) -> pd.DataFrame:
    """
    Compara parâmetros treináveis entre fine-tuning completo e PEFT.

    Reproduz análise da Tabela 2 do material da Aula 5 em escala
    reduzida, mostrando a redução de parâmetros treináveis com LoRA.

    Args:
        base_model: Modelo com fine-tuning completo (todos params treináveis).
        peft_model: Modelo com adaptadores PEFT aplicados.

    Returns:
        DataFrame comparativo com colunas:
        [method, total_params, trainable_params, trainable_pct, estimated_memory_mb].

    Referência:
        Hu, E.J. et al. (2022). Tabela 2 — Comparação de técnicas PEFT.
    """
    from src.utils import count_parameters, get_model_size_mb

    base_stats = count_parameters(base_model)
    peft_stats = count_parameters(peft_model)

    rows = [
        {
            "method": "Full Fine-Tuning",
            "total_params": base_stats["total"],
            "trainable_params": base_stats["total"],  # todos treináveis
            "trainable_pct": 100.0,
            "estimated_memory_mb": get_model_size_mb(base_model),
        },
        {
            "method": "LoRA (PEFT)",
            "total_params": peft_stats["total"],
            "trainable_params": peft_stats["trainable"],
            "trainable_pct": peft_stats["trainable_pct"],
            "estimated_memory_mb": get_model_size_mb(peft_model),
        },
    ]
    return pd.DataFrame(rows)


def benchmark_training_memory(
    model: Any,
    dataset: Dataset,
    batch_size: int = 16,
    num_steps: int = 5,
) -> Dict[str, float]:
    """
    Mede consumo de memória GPU durante treinamento.

    Relevante para comparar LoRA vs QLoRA conforme Tabela 2 do
    material (memória GPU de 28GB para full FT vs 6GB para QLoRA
    em LLaMA-7B).

    Args:
        model: Modelo PyTorch.
        dataset: Dataset de treino.
        batch_size: Tamanho do batch.
        num_steps: Número de passos de forward+backward.

    Returns:
        Dicionário com peak_memory_mb e avg_step_time_ms.
    """
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível; benchmark de memória retorna 0.")
        return {"peak_memory_mb": 0.0, "avg_step_time_ms": 0.0}

    model.train()
    torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4
    )

    step_times: List[float] = []
    for i, batch in enumerate(loader):
        if i >= num_steps:
            break
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        start = time.perf_counter()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_times.append((time.perf_counter() - start) * 1000)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_ms = float(np.mean(step_times)) if step_times else 0.0

    return {"peak_memory_mb": peak_mb, "avg_step_time_ms": avg_ms}
