"""
Módulo de treinamento, fine-tuning e knowledge distillation.

Implementa loops de treinamento para o modelo base, fine-tuning pós-pruning
e knowledge distillation conforme o framework teacher-student de Hinton et
al. (2015) discutido na seção *Saiba Mais* do documento 04.

Referências:
    Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge
        in a Neural Network. NIPS Workshop.
    Han, S. et al. (2015). Deep Compression. ICLR.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# 1. Distillation Loss
# ═══════════════════════════════════════════════════════════════════════════

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Função de perda para Knowledge Distillation.

    Combina *soft loss* (KL divergence com temperatura T) e *hard loss*
    (cross-entropy com labels reais), conforme proposto por Hinton et al.
    (2015). O parâmetro de temperatura T controla a "suavidade" das
    distribuições: valores maiores revelam mais *dark knowledge*.

    A fórmula implementada, discutida na seção *Saiba Mais* do doc 04:

        L = α · T² · D_KL(p_student^T || p_teacher^T)
            + (1 − α) · L_CE(p_student, y)

    O fator T² compensa a redução nos gradientes causada pela
    temperatura elevada, mantendo escala adequada para o treinamento.

    Args:
        student_logits: Logits do modelo student, shape ``(B, C)``.
        teacher_logits: Logits do modelo teacher, shape ``(B, C)``.
        labels: Rótulos inteiros, shape ``(B,)``.
        temperature: Temperatura para soft labels (T > 1 suaviza).
        alpha: Peso da soft loss vs. hard loss (0 = só hard, 1 = só soft).

    Returns:
        Perda escalar combinada.

    Reference:
        Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the
        Knowledge in a Neural Network. NIPS Deep Learning Workshop.
    """
    # Soft loss: KL divergence com temperatura T
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
        temperature * temperature
    )

    # Hard loss: cross-entropy padrão
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training loop genérico
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Treina o modelo por uma época.

    Args:
        model: Modelo a treinar.
        loader: DataLoader de treino.
        criterion: Função de perda.
        optimizer: Otimizador.
        device: Dispositivo (cpu / cuda).

    Returns:
        ``(loss_media, acuracia)`` da época.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Avalia modelo em um DataLoader.

    Args:
        model: Modelo a avaliar.
        loader: DataLoader de validação/teste.
        criterion: Função de perda.
        device: Dispositivo.

    Returns:
        ``(loss_media, acuracia)``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
    """Treina modelo completo com SGD + CosineAnnealing.

    Pipeline de treinamento conforme descrito na seção *Saiba Mais* do
    documento 04: primeiro treina-se o modelo denso até convergência para
    estabelecer baseline de acurácia e métricas de referência.

    Args:
        model: Modelo a treinar.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        epochs: Número de épocas.
        lr: Learning rate inicial.
        device: Dispositivo.

    Returns:
        Histórico com chaves ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc``.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
            f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}"
        )

    return history


# ═══════════════════════════════════════════════════════════════════════════
# 3. Fine-tuning pós-pruning
# ═══════════════════════════════════════════════════════════════════════════

def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 0.001,
    device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
    """Fine-tuna modelo após pruning para recuperar acurácia.

    Conforme o pipeline de Han et al. (2015) discutido no documento 04,
    após cada etapa de pruning realiza-se fine-tuning com learning rate
    reduzida para que os pesos remanescentes compensem as conexões
    removidas.

    Args:
        model: Modelo com pruning aplicado.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        epochs: Épocas de fine-tuning (tipicamente 3-5).
        lr: Learning rate (menor que treinamento original).
        device: Dispositivo.

    Returns:
        Histórico de fine-tuning.
    """
    return train_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Knowledge Distillation training
# ═══════════════════════════════════════════════════════════════════════════

def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
    """Treina student via Knowledge Distillation com teacher fixo.

    Implementa o framework teacher-student de Hinton et al. (2015),
    utilizando soft labels geradas pelo teacher com temperatura T.
    Conforme seção *Saiba Mais* do documento 04, o *dark knowledge*
    contido nas probabilidades suavizadas revela relações entre classes
    que auxiliam o student a aprender representações mais ricas.

    Args:
        student: Modelo student (menor).
        teacher: Modelo teacher (maior, já treinado).
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        epochs: Número de épocas.
        lr: Learning rate do student.
        temperature: Temperatura para soft labels.
        alpha: Peso de soft loss vs. hard loss.
        device: Dispositivo.

    Returns:
        Histórico de treinamento com distillation.
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = Adam(student.parameters(), lr=lr)
    criterion_eval = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Distill {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(
                student_logits, teacher_logits, labels, temperature, alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        t_loss = running_loss / total
        t_acc = correct / total
        v_loss, v_acc = evaluate(student, val_loader, criterion_eval, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Distill Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
            f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}"
        )

    return history
