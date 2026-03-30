"""
Módulo de avaliação, profiling e análise de performance.

Implementa métricas de avaliação do modelo e ferramentas de profiling para
análise de gargalos em GPU, conforme discutido na seção 'Profiling Avançado:
Identificando Gargalos' do material da Aula 06. Inclui implementação do
Modelo Roofline (Williams, Waterman & Patterson, 2009) para classificação
de operações como compute-bound ou memory-bound.

Integra com PyTorch Profiler conforme demonstrado no Snippet 2 da aula:

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        model(input_tensor)

Referências:
    Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful
        Visual Performance Model. Communications of the ACM, 52(4), 65-76.
    Dao, T. (2023). FlashAttention-2. NeurIPS.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ---------------------------------------------------------------------------
# Modelo Roofline — Williams, Waterman & Patterson (2009)
# ---------------------------------------------------------------------------

class RooflineModel:
    """Implementação do Modelo Roofline para análise de workloads GPU.

    O Modelo Roofline (Williams, Waterman & Patterson, 2009) caracteriza a
    performance atingível de um workload como:

        Performance_atingível = min(Peak_compute, BW_mem × AI)

    onde AI (Arithmetic Intensity) = FLOPs / Bytes Transferidos.

    Workloads com baixa AI são memory-bound (limitados pela largura de banda);
    workloads com alta AI são compute-bound (limitados pela capacidade de compute).

    Conforme Tabela 1 do material da aula:
    - Matmul: ~100 FLOP/byte → compute-bound
    - BatchNorm: ~2 FLOP/byte → memory-bound
    - Softmax: ~1 FLOP/byte → memory-bound

    Args:
        peak_compute_tflops: Pico de performance computacional em TFLOPS.
        memory_bandwidth_gb_s: Largura de banda de memória em GB/s.

    Referência:
        Williams, S. et al. (2009). Roofline: An Insightful Visual
        Performance Model for Multicore Architectures. CACM, 52(4), 65-76.
    """

    # Perfis de hardware conhecidos (TFLOPS FP16, GB/s HBM)
    HARDWARE_PROFILES = {
        "A100_80GB": {"peak_tflops": 312.0, "bw_gb_s": 2039.0},
        "H100_SXM": {"peak_tflops": 989.0, "bw_gb_s": 3352.0},
        "V100_32GB": {"peak_tflops": 125.0, "bw_gb_s": 900.0},
        "RTX_4090":  {"peak_tflops": 165.0, "bw_gb_s": 1008.0},
        "T4":        {"peak_tflops": 65.0,  "bw_gb_s": 300.0},
        "CPU_generic": {"peak_tflops": 0.5, "bw_gb_s": 50.0},
    }

    # Intensidades aritméticas típicas de operações DL (FLOP/byte)
    # Conforme Tabela 1 do material da aula (Dao et al., 2022)
    TYPICAL_AI = {
        "matmul_large": 100.0,
        "conv_3x3": 30.0,
        "layernorm": 3.0,
        "batchnorm": 2.0,
        "softmax": 1.0,
        "gelu": 1.5,
        "elementwise_add": 0.25,
    }

    def __init__(
        self,
        peak_compute_tflops: float = 312.0,
        memory_bandwidth_gb_s: float = 2039.0,
    ) -> None:
        self.peak_compute_tflops = peak_compute_tflops
        self.memory_bandwidth_gb_s = memory_bandwidth_gb_s
        # Ponto de equilíbrio (ridge point) em FLOP/byte
        self.ridge_point = (peak_compute_tflops * 1e12) / (memory_bandwidth_gb_s * 1e9)

    @classmethod
    def from_hardware(cls, hardware_name: str) -> "RooflineModel":
        """Cria modelo roofline a partir de perfil de hardware.

        Args:
            hardware_name: Nome do hardware (ex: 'A100_80GB', 'H100_SXM').

        Returns:
            Instância de RooflineModel.
        """
        if hardware_name not in cls.HARDWARE_PROFILES:
            available = ", ".join(cls.HARDWARE_PROFILES.keys())
            raise ValueError(f"Hardware '{hardware_name}' não encontrado. Disponíveis: {available}")
        profile = cls.HARDWARE_PROFILES[hardware_name]
        return cls(profile["peak_tflops"], profile["bw_gb_s"])

    def attainable_performance(self, arithmetic_intensity: float) -> float:
        """Calcula performance atingível para dada intensidade aritmética.

        Performance = min(Peak_compute, BW_mem × AI)

        Conforme equação do modelo roofline (Williams et al., 2009).

        Args:
            arithmetic_intensity: Intensidade aritmética em FLOP/byte.

        Returns:
            Performance atingível em TFLOPS.
        """
        memory_bound_perf = (self.memory_bandwidth_gb_s * 1e9 * arithmetic_intensity) / 1e12
        return min(self.peak_compute_tflops, memory_bound_perf)

    def classify_operation(self, arithmetic_intensity: float) -> str:
        """Classifica operação como compute-bound ou memory-bound.

        Args:
            arithmetic_intensity: Intensidade aritmética em FLOP/byte.

        Returns:
            'compute-bound', 'memory-bound' ou 'balanced'.
        """
        ratio = arithmetic_intensity / self.ridge_point
        if ratio > 1.2:
            return "compute-bound"
        elif ratio < 0.8:
            return "memory-bound"
        else:
            return "balanced"

    def analyze_operations(self) -> List[Dict[str, Any]]:
        """Analisa todas as operações típicas de deep learning.

        Classifica operações conforme Tabela 1 do material da aula
        usando o modelo roofline.

        Returns:
            Lista de dicionários com análise por operação.
        """
        results = []
        for op_name, ai in self.TYPICAL_AI.items():
            perf = self.attainable_performance(ai)
            classification = self.classify_operation(ai)
            results.append({
                "operation": op_name,
                "arithmetic_intensity": ai,
                "attainable_tflops": perf,
                "classification": classification,
                "utilization_pct": (perf / self.peak_compute_tflops) * 100,
            })
        return results

    def get_roofline_data(
        self,
        ai_range: Tuple[float, float] = (0.1, 500.0),
        n_points: int = 200,
    ) -> Dict[str, np.ndarray]:
        """Gera dados para plotar o gráfico de roofline.

        Args:
            ai_range: Range de intensidades aritméticas (min, max).
            n_points: Número de pontos.

        Returns:
            Dicionário com arrays para plotting.
        """
        ai_values = np.logspace(
            np.log10(ai_range[0]), np.log10(ai_range[1]), n_points
        )
        perf_values = np.array([self.attainable_performance(ai) for ai in ai_values])

        return {
            "arithmetic_intensity": ai_values,
            "attainable_performance": perf_values,
            "ridge_point": self.ridge_point,
            "peak_compute": self.peak_compute_tflops,
        }


# ---------------------------------------------------------------------------
# Profiling com PyTorch Profiler — Snippet 2 do material da aula
# ---------------------------------------------------------------------------

def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    n_iterations: int = 5,
    warmup: int = 2,
) -> Dict[str, Any]:
    """Realiza profiling do modelo usando PyTorch Profiler.

    Implementa o Snippet 2 do material da Aula 06:

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True) as prof:
            model(input_tensor)

    Gera métricas detalhadas de tempo e memória por operação, conforme
    discutido na seção 'Profiling Avançado: Identificando Gargalos':
    "O PyTorch Profiler integra-se nativamente ao framework, capturando
    traces de execução com overhead mínimo."

    Args:
        model: Modelo PyTorch.
        input_shape: Shape do tensor de entrada.
        device: Dispositivo.
        n_iterations: Iterações para profiling.
        warmup: Iterações de warmup.

    Returns:
        Dicionário com resultados do profiling.
    """
    from torch.profiler import profile, ProfilerActivity

    model = model.to(device)
    model.eval()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    x = torch.randn(input_shape, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Profiling — Snippet 2 da aula
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_iterations):
            with torch.no_grad():
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Extrair métricas — "print(prof.key_averages().table(sort_by='cuda_time_total'))"
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=20)

    # Processar resultados
    events = []
    for evt in prof.key_averages():
        event_info = {
            "name": evt.key,
            "cpu_time_ms": evt.cpu_time_total / 1000,
            "count": evt.count,
        }
        if device.type == "cuda":
            event_info["cuda_time_ms"] = evt.cuda_time_total / 1000
        events.append(event_info)

    # Ordenar por tempo
    time_key = "cuda_time_ms" if device.type == "cuda" else "cpu_time_ms"
    events.sort(key=lambda e: e.get(time_key, 0), reverse=True)

    return {
        "table": table,
        "events": events[:20],
        "total_events": len(events),
        "device": str(device),
    }


def benchmark_memory_with_checkpointing(
    model_fn,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compara uso de memória com e sem gradient checkpointing.

    Demonstra o conceito de Chen et al. (2016) discutido na seção
    'Otimização de Memória GPU: Gradient Checkpointing':
    "checkpointing pode reduzir uso de memória em até 60% em modelos grandes"

    Args:
        model_fn: Callable(use_checkpointing) que retorna modelo.
        input_shape: Shape do tensor de entrada.
        device: Dispositivo.

    Returns:
        Dicionário com métricas de memória para cada configuração.
    """
    results = {}

    for use_ckpt, label in [(False, "sem_checkpointing"), (True, "com_checkpointing")]:
        model = model_fn(use_checkpointing=use_ckpt).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(input_shape, device=device)
        y = torch.randint(0, 10, (input_shape[0],), device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        # Forward + backward
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        else:
            # CPU: estimativa baseada em parâmetros
            peak_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
            current_mem = peak_mem

        results[label] = {
            "peak_memory_mb": peak_mem,
            "current_memory_mb": current_mem,
            "n_parameters": sum(p.numel() for p in model.parameters()),
        }

        del model, optimizer, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Calcular redução
    if "sem_checkpointing" in results and "com_checkpointing" in results:
        base = results["sem_checkpointing"]["peak_memory_mb"]
        ckpt = results["com_checkpointing"]["peak_memory_mb"]
        if base > 0:
            results["reducao_pct"] = ((base - ckpt) / base) * 100
        else:
            results["reducao_pct"] = 0.0

    return results


# ---------------------------------------------------------------------------
# Métricas de Classificação
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 10,
) -> Dict[str, float]:
    """Calcula métricas de classificação.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Labels preditos.
        n_classes: Número de classes.

    Returns:
        Dicionário com métricas.
    """
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / max(total, 1)

    # Per-class accuracy
    per_class_acc = {}
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[f"class_{c}_accuracy"] = float((y_pred[mask] == c).sum() / mask.sum())

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            confusion[t, p] += 1

    return {
        "accuracy": float(accuracy),
        "n_correct": int(correct),
        "n_total": int(total),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_classes: int = 10,
) -> Dict[str, Any]:
    """Avalia modelo completo com métricas e tempo de inferência.

    Mede latência de inferência, métrica essencial para comparação
    entre aceleradores conforme Tabela 2 do material da aula.

    Args:
        model: Modelo PyTorch.
        data_loader: DataLoader de teste.
        device: Dispositivo.
        n_classes: Número de classes.

    Returns:
        Dicionário com métricas e timing.
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            preds = model(inputs).argmax(dim=-1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = calculate_metrics(y_true, y_pred, n_classes)
    metrics["avg_inference_time_ms"] = np.mean(inference_times) * 1000
    metrics["std_inference_time_ms"] = np.std(inference_times) * 1000
    metrics["total_inference_time_s"] = sum(inference_times)
    metrics["throughput_samples_per_sec"] = len(y_true) / sum(inference_times)

    return metrics


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Matriz de Confusão",
) -> None:
    """Plota matriz de confusão.

    Args:
        confusion_matrix: Matriz (n_classes, n_classes).
        class_names: Nomes das classes.
        save_path: Caminho para salvar figura.
        title: Título do gráfico.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_classes = len(confusion_matrix)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(
        confusion_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figura salva: %s", save_path)

    plt.close(fig)
    return fig


def plot_roofline(
    roofline: RooflineModel,
    operations: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Modelo Roofline — Análise de Performance GPU",
) -> None:
    """Plota gráfico do Modelo Roofline.

    Visualiza o modelo roofline (Williams et al., 2009) com operações
    de deep learning classificadas como compute-bound ou memory-bound,
    conforme Figura 1 do material da aula.

    Args:
        roofline: Instância de RooflineModel.
        operations: Dicionário {nome: intensidade_aritmética}.
        save_path: Caminho para salvar.
        title: Título do gráfico.

    Referência:
        Williams, S. et al. (2009). Roofline: An Insightful Visual
        Performance Model. CACM, 52(4).
    """
    import matplotlib.pyplot as plt

    data = roofline.get_roofline_data()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Roofline
    ax.loglog(
        data["arithmetic_intensity"], data["attainable_performance"],
        "b-", linewidth=2, label="Roofline",
    )

    # Ridge point
    ax.axvline(data["ridge_point"], color="gray", linestyle="--", alpha=0.5,
               label=f"Ridge Point ({data['ridge_point']:.1f} FLOP/byte)")

    # Operações
    if operations is None:
        operations = RooflineModel.TYPICAL_AI

    colors = {"compute-bound": "green", "memory-bound": "red", "balanced": "orange"}
    for op_name, ai in operations.items():
        perf = roofline.attainable_performance(ai)
        classification = roofline.classify_operation(ai)
        ax.plot(ai, perf, "o", color=colors[classification], markersize=8)
        ax.annotate(
            op_name.replace("_", " "),
            (ai, perf), textcoords="offset points",
            xytext=(5, 5), fontsize=8,
        )

    ax.set_xlabel("Intensidade Aritmética (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance Atingível (TFLOPS)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figura salva: %s", save_path)

    plt.close(fig)
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Curvas de Treinamento",
) -> None:
    """Plota curvas de loss e acurácia do treinamento.

    Args:
        metrics: Dicionário com listas de métricas por época.
        save_path: Caminho para salvar.
        title: Título.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(metrics.get("train_losses", [])) + 1)

    # Loss
    if "train_losses" in metrics:
        axes[0].plot(epochs, metrics["train_losses"], "b-o", label="Treino", markersize=4)
    if "val_losses" in metrics:
        axes[0].plot(epochs, metrics["val_losses"], "r-o", label="Validação", markersize=4)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Acurácia
    if "train_accuracies" in metrics:
        axes[1].plot(epochs, metrics["train_accuracies"], "b-o", label="Treino", markersize=4)
    if "val_accuracies" in metrics:
        axes[1].plot(epochs, metrics["val_accuracies"], "r-o", label="Validação", markersize=4)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_title(f"{title} — Acurácia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_precision_comparison(
    benchmark_results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    title: str = "Comparação de Precisões Numéricas",
) -> None:
    """Plota comparação de performance entre precisões (FP32, FP16, BF16).

    Visualiza os resultados de benchmark de diferentes formatos
    numéricos conforme discutido na seção 'Mixed Precision Training'
    (Micikevicius et al., 2018).

    Args:
        benchmark_results: Resultados do benchmark_precision().
        save_path: Caminho para salvar.
        title: Título.
    """
    import matplotlib.pyplot as plt

    names = list(benchmark_results.keys())
    times = [benchmark_results[n]["avg_time_ms"] for n in names]
    throughputs = [benchmark_results[n]["throughput_samples_per_sec"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"fp32": "#2196F3", "fp16": "#4CAF50", "bf16": "#FF9800"}
    bar_colors = [colors.get(n, "#9E9E9E") for n in names]

    # Tempo por batch
    axes[0].bar(names, times, color=bar_colors)
    axes[0].set_ylabel("Tempo por Batch (ms)")
    axes[0].set_title(f"{title} — Latência")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(times):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)

    # Throughput
    axes[1].bar(names, throughputs, color=bar_colors)
    axes[1].set_ylabel("Throughput (amostras/s)")
    axes[1].set_title(f"{title} — Throughput")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(throughputs):
        axes[1].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
