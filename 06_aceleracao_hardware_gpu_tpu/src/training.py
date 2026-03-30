"""
Módulo de treinamento com suporte a Mixed Precision e profiling.

Implementa loop de treinamento com Automatic Mixed Precision (AMP) conforme
proposto por Micikevicius et al. (2018) e demonstrado no Snippet 1 do material
da Aula 06. O GradScaler realiza loss scaling dinâmico para evitar underflow
em gradientes FP16.

Também implementa gradient checkpointing (Chen et al., 2016, Snippet 3) e
integração com PyTorch Profiler (Snippet 2) para análise de gargalos conforme
discutido na seção 'Profiling Avançado: Identificando Gargalos'.

Referências:
    Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    Dao, T. (2023). FlashAttention-2. NeurIPS.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class TrainingConfig:
    """Configuração de treinamento.

    Parâmetros alinhados com as recomendações do material da aula:
    - batch_size múltiplo de 8 para Tensor Cores
    - use_amp para Mixed Precision Training (Micikevicius et al., 2018)
    - use_checkpointing para Gradient Checkpointing (Chen et al., 2016)

    Args:
        n_epochs: Número de épocas.
        learning_rate: Taxa de aprendizado.
        batch_size: Tamanho do batch (múltiplo de 8 recomendado).
        weight_decay: Regularização L2.
        use_amp: Ativar Automatic Mixed Precision.
        amp_dtype: Tipo de dado para AMP (float16 ou bfloat16).
        use_checkpointing: Ativar gradient checkpointing.
        gradient_clip_max_norm: Norma máxima para gradient clipping.
        log_interval: Intervalo de logging em batches.
        seed: Semente para reprodutibilidade.
    """
    n_epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    use_checkpointing: bool = False
    gradient_clip_max_norm: float = 1.0
    log_interval: int = 10
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Métricas coletadas durante treinamento.

    Registra tempos de execução para análise de performance conforme
    discutido na seção 'Profiling Avançado: Identificando Gargalos'.
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    peak_memory_mb: List[float] = field(default_factory=list)


class Trainer:
    """Treinador com suporte a Mixed Precision e Gradient Checkpointing.

    Implementa o loop de treinamento demonstrado nos Snippets 1 e 3 do
    material da Aula 06, com os seguintes componentes:

    1. **AMP (Automatic Mixed Precision):** Usa autocast para forward pass
       em FP16/BF16 e GradScaler para loss scaling dinâmico, conforme
       Micikevicius et al. (2018).

    2. **Gradient Checkpointing:** Configurável via modelo, reduz memória
       de O(n) para O(√n) conforme Chen et al. (2016).

    3. **Gradient Clipping:** Evita explosão de gradientes, especialmente
       importante em mixed precision onde ranges numéricos são limitados.

    Args:
        model: Modelo PyTorch.
        config: Configuração de treinamento.
        device: Dispositivo (CPU/GPU).

    Referências:
        Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
        Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        # Move modelo para dispositivo
        self.model.to(self.device)

        # Otimizador AdamW com weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Critério de perda
        self.criterion = nn.CrossEntropyLoss()

        # Configuração de Mixed Precision — Snippet 1 do material da aula
        # "O GradScaler ajusta automaticamente o fator de escala do loss
        #  para evitar underflow em gradientes FP16"
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Métricas
        self.metrics = TrainingMetrics()

        logger.info(
            "Trainer configurado: device=%s, amp=%s, dtype=%s, checkpointing=%s",
            self.device, self.use_amp, config.amp_dtype, config.use_checkpointing,
        )

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Treina uma época com Mixed Precision.

        Implementa o loop de treinamento do Snippet 1 da aula:

            scaler = GradScaler()
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        Args:
            train_loader: DataLoader de treino.

        Returns:
            Tupla (loss médio, acurácia).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision forward pass — conforme Snippet 1
            # autocast converte operações elegíveis para FP16/BF16
            # mantendo operações sensíveis em FP32 (ex: softmax, LayerNorm)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.config.amp_dtype,
            ):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Loss scaling e backward — evita underflow de gradientes FP16
            # Loss_scaled = S × Loss_original (Micikevicius et al., 2018)
            self.scaler.scale(loss).backward()

            # Gradient clipping — importante em mixed precision
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_max_norm,
            )

            # Optimizer step com unscaling automático
            # ∇W_final = ∇W_scaled / S
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    "  Batch %d/%d — Loss: %.4f",
                    batch_idx + 1, len(train_loader), loss.item(),
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Valida modelo no conjunto de validação.

        Args:
            val_loader: DataLoader de validação.

        Returns:
            Tupla (loss médio, acurácia).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.config.amp_dtype,
            ):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingMetrics:
        """Loop de treinamento completo com medição de performance.

        Mede throughput (amostras/segundo) e consumo de memória GPU
        por época, métricas fundamentais para análise de performance
        conforme discutido na seção 'Profiling Avançado'.

        Args:
            train_loader: DataLoader de treino.
            val_loader: DataLoader de validação (opcional).

        Returns:
            Métricas coletadas durante treinamento.
        """
        logger.info("Iniciando treinamento: %d épocas", self.config.n_epochs)
        best_val_loss = float("inf")

        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()

            # Treinamento
            train_loss, train_acc = self.train_epoch(train_loader)
            self.metrics.train_losses.append(train_loss)
            self.metrics.train_accuracies.append(train_acc)

            # Validação
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.metrics.val_losses.append(val_loss)
                self.metrics.val_accuracies.append(val_acc)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            # Tempo e throughput
            epoch_time = time.time() - epoch_start
            self.metrics.epoch_times.append(epoch_time)

            n_samples = len(train_loader.dataset)
            throughput = n_samples / epoch_time
            self.metrics.throughputs.append(throughput)

            # Memória GPU (se disponível)
            if self.device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                self.metrics.peak_memory_mb.append(peak_mem)
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                self.metrics.peak_memory_mb.append(0.0)

            logger.info(
                "Época %d/%d — Train Loss: %.4f, Train Acc: %.4f | "
                "Val Loss: %.4f, Val Acc: %.4f | "
                "Tempo: %.2fs, Throughput: %.0f amostras/s, Mem: %.1f MB",
                epoch + 1, self.config.n_epochs,
                train_loss, train_acc, val_loss, val_acc,
                epoch_time, throughput, self.metrics.peak_memory_mb[-1],
            )

            # Salva melhor modelo
            if val_loader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

        return self.metrics

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Salva checkpoint do modelo."""
        ckpt_dir = OUTPUT_DIR / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, path)
        logger.info("Checkpoint salvo: %s (val_loss=%.4f)", path, val_loss)


def benchmark_precision(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    n_iterations: int = 50,
    warmup: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Benchmarka modelo em diferentes precisões numéricas.

    Compara performance de FP32 vs FP16 vs BF16, demonstrando o impacto
    de Mixed Precision Training discutido na seção 'Mixed Precision Training:
    Acelerando sem Comprometer' (Micikevicius et al., 2018).

    Formatos testados conforme material da aula:
    - FP32: Precisão completa (baseline)
    - FP16: Half precision com Tensor Cores (1 bit sinal, 5 expoente, 10 mantissa)
    - BF16: Brain Float 16 (1 bit sinal, 8 expoente, 7 mantissa)

    Args:
        model: Modelo PyTorch.
        input_shape: Shape do tensor de entrada (B, C, H, W).
        device: Dispositivo.
        n_iterations: Número de iterações para benchmark.
        warmup: Iterações de warmup (descartadas).

    Returns:
        Dicionário com tempos médios e throughput por precisão.
    """
    results = {}
    model = model.to(device)
    model.eval()

    precisions = {"fp32": torch.float32}
    if device.type == "cuda":
        precisions["fp16"] = torch.float16
        if torch.cuda.is_bf16_supported():
            precisions["bf16"] = torch.bfloat16

    for name, dtype in precisions.items():
        times = []
        x = torch.randn(input_shape, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(dtype != torch.float32),
                dtype=dtype,
            ):
                with torch.no_grad():
                    _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        for _ in range(n_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(dtype != torch.float32),
                dtype=dtype,
            ):
                with torch.no_grad():
                    _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = input_shape[0] / avg_time

        results[name] = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5 * 1000,
            "throughput_samples_per_sec": throughput,
        }

        logger.info(
            "Benchmark %s: %.2f ms/batch, %.0f amostras/s",
            name.upper(), results[name]["avg_time_ms"], throughput,
        )

    return results


def cross_validate(
    model_fn,
    features: "np.ndarray",
    labels: "np.ndarray",
    n_folds: int = 3,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Cross-validation simplificada para avaliação do modelo.

    Args:
        model_fn: Callable que retorna nova instância do modelo.
        features: Array de features.
        labels: Array de labels.
        n_folds: Número de folds.
        config: Configuração de treinamento.
        device: Dispositivo.

    Returns:
        Dicionário com métricas por fold.
    """
    import numpy as np
    from src.data_preprocessing import SyntheticImageDataset

    if config is None:
        config = TrainingConfig(n_epochs=3)
    if device is None:
        device = torch.device("cpu")

    fold_size = len(labels) // n_folds
    indices = np.arange(len(labels))
    np.random.RandomState(config.seed).shuffle(indices)

    fold_accuracies = []
    fold_losses = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_dataset = SyntheticImageDataset(features[train_idx], labels[train_idx])
        val_dataset = SyntheticImageDataset(features[val_idx], labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        model = model_fn()
        trainer = Trainer(model, config, device)
        trainer.train(train_loader, val_loader)

        val_loss, val_acc = trainer.validate(val_loader)
        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)

        logger.info("Fold %d/%d — Val Acc: %.4f, Val Loss: %.4f",
                     fold + 1, n_folds, val_acc, val_loss)

    return {"accuracies": fold_accuracies, "losses": fold_losses}


def train_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, TrainingMetrics]:
    """Função de conveniência para treinamento completo.

    Args:
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        config: Configuração. Se None, usa padrão.
        device: Dispositivo. Se None, auto-detecta.
        model: Modelo. Se None, cria SimpleViT padrão.

    Returns:
        Tupla (modelo treinado, métricas).
    """
    from src.model import create_model
    from src.utils import get_device

    if config is None:
        config = TrainingConfig()
    if device is None:
        device = get_device()
    if model is None:
        model = create_model(use_checkpointing=config.use_checkpointing)

    trainer = Trainer(model, config, device)
    metrics = trainer.train(train_loader, val_loader)

    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_preprocessing import DataPreprocessor
    from src.utils import get_device

    device = get_device()
    logger.info("Dispositivo: %s", device)

    # Preparar dados
    preprocessor = DataPreprocessor(n_samples=2000)
    df = preprocessor.load_data()
    df = preprocessor.clean_data(df)
    features, labels = preprocessor.prepare_features(df)
    splits = preprocessor.split_data(features, labels)
    loaders = preprocessor.create_dataloaders(splits, batch_size=64)

    # Treinar
    config = TrainingConfig(n_epochs=5, use_amp=(device.type == "cuda"))
    model, metrics = train_model(
        loaders["train"], loaders["val"], config=config, device=device,
    )

    logger.info("Treinamento concluído. Loss final: %.4f", metrics.train_losses[-1])
