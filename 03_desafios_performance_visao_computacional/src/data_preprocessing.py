"""
Módulo de pré-processamento e geração de dados para benchmarking de visão.

Implementa geração de datasets sintéticos de benchmark de arquiteturas e
datasets de imagens para treinamento, conforme os conceitos de complexidade
computacional e trade-offs de resolução discutidos na seção 'Saiba Mais'.

Referência:
    Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks. ICML 2019.
    He, K. & Sun, J. (2015). Convolutional Neural Networks at Constrained
    Time Cost. CVPR 2015.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Dados-âncora de arquiteturas (extraídos da seção Saiba Mais)
# -------------------------------------------------------------------------

_ARCHITECTURE_ANCHORS: list[dict[str, Any]] = [
    # CNNs — Tabela 1 e seção 2 do material
    {"model_name": "mobilenetv2",        "family": "cnn",    "params": 3.4,   "flops_224": 0.30,  "top1": 72.0, "lat_gpu": 1.2,  "lat_cpu": 18.0},
    {"model_name": "mobilenetv3_large",  "family": "cnn",    "params": 5.4,   "flops_224": 0.22,  "top1": 75.2, "lat_gpu": 1.8,  "lat_cpu": 15.0},
    {"model_name": "efficientnet_b0",    "family": "cnn",    "params": 5.3,   "flops_224": 0.39,  "top1": 77.1, "lat_gpu": 2.1,  "lat_cpu": 22.0},
    {"model_name": "efficientnet_b4",    "family": "cnn",    "params": 19.0,  "flops_224": 4.20,  "top1": 82.9, "lat_gpu": 8.3,  "lat_cpu": 95.0},
    {"model_name": "efficientnetv2_s",   "family": "cnn",    "params": 21.5,  "flops_224": 8.40,  "top1": 83.9, "lat_gpu": 5.2,  "lat_cpu": 78.0},
    {"model_name": "resnet18",           "family": "cnn",    "params": 11.7,  "flops_224": 1.82,  "top1": 69.8, "lat_gpu": 1.5,  "lat_cpu": 12.0},
    {"model_name": "resnet50",           "family": "cnn",    "params": 25.6,  "flops_224": 4.12,  "top1": 76.1, "lat_gpu": 3.2,  "lat_cpu": 35.0},
    {"model_name": "vgg16",              "family": "cnn",    "params": 138.4, "flops_224": 15.50, "top1": 71.6, "lat_gpu": 4.8,  "lat_cpu": 120.0},
    {"model_name": "convnext_tiny",      "family": "cnn",    "params": 28.6,  "flops_224": 4.47,  "top1": 82.1, "lat_gpu": 3.8,  "lat_cpu": 42.0},
    {"model_name": "convnext_small",     "family": "cnn",    "params": 50.2,  "flops_224": 8.69,  "top1": 83.1, "lat_gpu": 5.9,  "lat_cpu": 68.0},
    # ViTs — Seção 3 do material
    {"model_name": "deit_tiny",          "family": "vit",    "params": 5.7,   "flops_224": 1.26,  "top1": 72.2, "lat_gpu": 2.8,  "lat_cpu": 30.0},
    {"model_name": "deit_small",         "family": "vit",    "params": 22.1,  "flops_224": 4.61,  "top1": 79.8, "lat_gpu": 4.5,  "lat_cpu": 55.0},
    {"model_name": "deit_base",          "family": "vit",    "params": 86.6,  "flops_224": 17.58, "top1": 81.8, "lat_gpu": 8.9,  "lat_cpu": 180.0},
    {"model_name": "vit_base_patch16",   "family": "vit",    "params": 86.6,  "flops_224": 17.58, "top1": 77.9, "lat_gpu": 9.2,  "lat_cpu": 185.0},
    # Híbridos — Seção 3 e 10 do material
    {"model_name": "swin_tiny",          "family": "hybrid",  "params": 28.3,  "flops_224": 4.51,  "top1": 81.3, "lat_gpu": 4.2,  "lat_cpu": 52.0},
    {"model_name": "swin_small",         "family": "hybrid",  "params": 49.6,  "flops_224": 8.74,  "top1": 83.0, "lat_gpu": 6.5,  "lat_cpu": 85.0},
    {"model_name": "efficientvit_b1",    "family": "hybrid",  "params": 9.1,   "flops_224": 0.86,  "top1": 79.4, "lat_gpu": 1.9,  "lat_cpu": 20.0},
    # YOLO backbones — Tabela 3, seção 5 do material
    {"model_name": "yolov8n_backbone",   "family": "cnn",    "params": 3.2,   "flops_224": 2.17,  "top1": 37.3, "lat_gpu": 1.8,  "lat_cpu": 25.0},
    {"model_name": "yolov8s_backbone",   "family": "cnn",    "params": 11.2,  "flops_224": 7.15,  "top1": 44.9, "lat_gpu": 2.4,  "lat_cpu": 38.0},
    {"model_name": "yolov8m_backbone",   "family": "cnn",    "params": 25.9,  "flops_224": 19.73, "top1": 50.2, "lat_gpu": 4.1,  "lat_cpu": 72.0},
]

# Resoluções de interesse (seção 4 do material — Tabela 2)
_RESOLUTIONS = [128, 224, 320, 384, 512]

# Fatores de escala de FLOPs por resolução em relação a 224
# FLOPs escalam ≈ (r/224)^2 para CNNs (seção 4), mas com fator sublinear
# devido FC layers e pooling global
_RESOLUTION_FLOPS_FACTOR = {
    128: 0.327,  # (128/224)^2 ≈ 0.327
    224: 1.000,
    320: 2.041,  # (320/224)^2 ≈ 2.041
    384: 2.939,  # (384/224)^2 ≈ 2.939
    512: 5.224,  # (512/224)^2 ≈ 5.224
}

# Hardware — fatores de velocidade relativa
_HARDWARE_SPEED = {
    "v100":    {"gpu_factor": 1.0,   "cpu_factor": None},
    "rtx3090": {"gpu_factor": 0.80,  "cpu_factor": None},
    "cpu":     {"gpu_factor": None,  "cpu_factor": 1.0},
}


class DataPreprocessor:
    """
    Pipeline de geração e pré-processamento de dados de benchmark.

    Gera dataset sintético de benchmarks de arquiteturas de visão computacional
    baseado em dados-âncora extraídos da literatura (Tan & Le, 2019;
    Dosovitskiy et al., 2021; Jocher et al., 2023). Variabilidade entre
    execuções é simulada com ruído gaussiano controlado.

    Attributes:
        random_state: Seed para reprodutibilidade.
        data: DataFrame carregado.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Inicializa o pré-processador.

        Args:
            random_state: Seed para reprodutibilidade.
        """
        self.random_state = random_state
        self.data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Geração de dataset de benchmark
    # ------------------------------------------------------------------

    @staticmethod
    def generate_benchmark_dataset(
        output_path: str = "data/raw/dataset.csv",
        n_runs_per_config: int = 17,
        noise_std: float = 0.03,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Gera dataset sintético de benchmarks de arquiteturas de visão.

        Cria ~5000 registros simulando execuções de benchmark para 20
        arquiteturas em 5 resoluções e 3 configurações de hardware.
        Valores-âncora são extraídos das Tabelas 1-3 do material da aula
        (Tan & Le, 2019; Dosovitskiy et al., 2021; Jocher et al., 2023).

        Conforme Reddi et al. (2020), benchmarks reproduzíveis com múltiplas
        execuções são essenciais para comparações justas entre modelos.

        Args:
            output_path: Caminho para salvar o CSV.
            n_runs_per_config: Execuções por configuração (modelo×resolução×hw).
            noise_std: Desvio-padrão relativo do ruído gaussiano (3%).
            random_state: Seed para reprodutibilidade.

        Returns:
            DataFrame com os dados gerados.
        """
        rng = np.random.RandomState(random_state)
        rows: list[dict[str, Any]] = []
        run_id = 0

        for arch in _ARCHITECTURE_ANCHORS:
            for resolution in _RESOLUTIONS:
                flops_factor = _RESOLUTION_FLOPS_FACTOR[resolution]
                base_flops = arch["flops_224"] * flops_factor

                # ViTs escalam pior com resolução (complexidade quadrática da
                # atenção — seção 1 do material)
                if arch["family"] == "vit":
                    n_patches_224 = (224 // 16) ** 2  # 196
                    n_patches_r = (resolution // 16) ** 2
                    attention_factor = (n_patches_r / n_patches_224) ** 1.5
                    base_flops = arch["flops_224"] * attention_factor

                for hw_name, hw_speed in _HARDWARE_SPEED.items():
                    for _ in range(n_runs_per_config):
                        run_id += 1
                        noise_lat = 1.0 + rng.normal(0, noise_std)
                        noise_tp = 1.0 + rng.normal(0, noise_std)

                        # Calcular latência conforme hardware
                        if hw_name == "cpu":
                            base_lat = arch["lat_cpu"] * (flops_factor ** 0.85)
                            latency = base_lat * noise_lat
                        else:
                            speed = hw_speed["gpu_factor"]
                            base_lat = arch["lat_gpu"] * (flops_factor ** 0.7)
                            latency = base_lat / speed * noise_lat

                        # Throughput = 1000 / latency (imagens/s para batch=1)
                        throughput = (1000.0 / latency) * noise_tp

                        # Memória escala com resolução e parâmetros
                        base_mem = arch["params"] * 4.0 + base_flops * 50.0
                        memory = base_mem * (1.0 + rng.normal(0, 0.02))

                        # Acurácia: leve ganho com resolução maior, com
                        # rendimentos decrescentes (seção 4 do material)
                        acc_boost = 2.5 * np.log(resolution / 224 + 1)
                        accuracy = arch["top1"] + acc_boost
                        accuracy += rng.normal(0, 0.1)  # ruído de medição

                        rows.append({
                            "model_name": arch["model_name"],
                            "model_family": arch["family"],
                            "input_resolution": resolution,
                            "hardware": hw_name,
                            "batch_size": 1,
                            "params_millions": round(arch["params"], 2),
                            "flops_giga": round(max(base_flops, 0.01), 4),
                            "top1_accuracy": round(min(accuracy, 99.0), 2),
                            "latency_ms": round(max(latency, 0.1), 3),
                            "throughput_imgs_sec": round(max(throughput, 1.0), 2),
                            "memory_mb": round(max(memory, 10.0), 2),
                            "run_id": run_id,
                        })

        df = pd.DataFrame(rows)

        # Salvar CSV
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info(
            "Dataset de benchmark gerado: %d registros em %s",
            len(df), output_path,
        )
        return df

    # ------------------------------------------------------------------
    # Geração de dataset de imagens sintéticas
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_images(
        output_dir: str = "data/raw/images",
        n_samples: int = 1000,
        img_size: int = 64,
        n_classes: int = 4,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera dataset sintético de imagens para benchmarking de treinamento.

        Cria imagens com formas geométricas simples representando classes
        diferentes, simulando um cenário de inspeção industrial como o
        case da Siemens mencionado na seção 'Mercado, Cases e Tendências'.

        Args:
            output_dir: Diretório para salvar imagens (opcional).
            n_samples: Número total de imagens.
            img_size: Tamanho da imagem (quadrada).
            n_classes: Número de classes.
            random_state: Seed para reprodutibilidade.

        Returns:
            Tupla (images, labels) com arrays numpy.
            images: shape (n_samples, 3, img_size, img_size), float32 [0, 1]
            labels: shape (n_samples,), int64
        """
        rng = np.random.RandomState(random_state)
        images = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            label = i % n_classes
            labels[i] = label

            # Background com ruído suave
            bg_color = rng.uniform(0.1, 0.3, size=(3, 1, 1))
            img = np.ones((3, img_size, img_size), dtype=np.float32) * bg_color
            img += rng.normal(0, 0.02, img.shape).astype(np.float32)

            cx, cy = img_size // 2, img_size // 2
            r = rng.randint(img_size // 6, img_size // 3)

            if label == 0:
                # Classe 0: círculo (peça normal)
                yy, xx = np.ogrid[:img_size, :img_size]
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
                color = rng.uniform(0.6, 0.9, size=3)
                for c in range(3):
                    img[c][mask] = color[c]

            elif label == 1:
                # Classe 1: retângulo (peça com defeito — arranhão)
                x1 = max(0, cx - r)
                x2 = min(img_size, cx + r)
                y1 = max(0, cy - r)
                y2 = min(img_size, cy + r)
                color = rng.uniform(0.5, 0.8, size=3)
                for c in range(3):
                    img[c, y1:y2, x1:x2] = color[c]
                # Adicionar "arranhão" diagonal
                for offset in range(-1, 2):
                    coords = np.arange(y1, y2)
                    x_coords = np.clip(
                        x1 + (coords - y1) + offset, 0, img_size - 1
                    )
                    for c in range(3):
                        img[c, coords, x_coords.astype(int)] = 0.9

            elif label == 2:
                # Classe 2: triângulo (componente tipo A)
                for y in range(cy - r, cy + r):
                    half_w = max(0, int(r * (1 - abs(y - cy + r) / (2 * r))))
                    x1 = max(0, cx - half_w)
                    x2 = min(img_size, cx + half_w)
                    color = rng.uniform(0.4, 0.7, size=3)
                    for c in range(3):
                        img[c, min(y, img_size - 1), x1:x2] = color[c]

            else:
                # Classe 3: forma irregular (anomalia)
                n_blobs = rng.randint(3, 6)
                for _ in range(n_blobs):
                    bx = rng.randint(0, img_size)
                    by = rng.randint(0, img_size)
                    br = rng.randint(3, img_size // 6)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    mask = ((xx - bx) ** 2 + (yy - by) ** 2) <= br ** 2
                    color = rng.uniform(0.5, 1.0, size=3)
                    for c in range(3):
                        img[c][mask] = color[c]

            images[i] = np.clip(img, 0.0, 1.0)

        # Salvar como .npz se diretório especificado
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / "synthetic_images.npz",
                images=images,
                labels=labels,
            )
            logger.info(
                "Imagens sintéticas geradas: %d amostras em %s",
                n_samples, output_dir,
            )

        return images, labels

    # ------------------------------------------------------------------
    # Carregamento e preparação
    # ------------------------------------------------------------------

    def load_data(self, path: str = "data/raw/dataset.csv") -> pd.DataFrame:
        """
        Carrega dataset de benchmarks a partir de CSV.

        Args:
            path: Caminho do arquivo CSV.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existe.
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em '{path}'. "
                "Execute generate_benchmark_dataset() primeiro."
            )
        self.data = pd.read_csv(path)
        logger.info("Dataset carregado: %d registros de %s", len(self.data), path)
        return self.data

    def prepare_features(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Prepara features para análise — agrega estatísticas por configuração.

        Calcula média, desvio-padrão e percentis de latência e throughput
        por combinação (modelo, resolução, hardware), seguindo a prática
        de múltiplas execuções recomendada por Reddi et al. (2020).

        Args:
            df: DataFrame de entrada. Se None, usa self.data.

        Returns:
            DataFrame agregado com estatísticas por configuração.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")

        group_cols = ["model_name", "model_family", "input_resolution", "hardware"]

        agg_df = df.groupby(group_cols).agg(
            params_millions=("params_millions", "first"),
            flops_giga=("flops_giga", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            latency_mean_ms=("latency_ms", "mean"),
            latency_std_ms=("latency_ms", "std"),
            latency_p50_ms=("latency_ms", "median"),
            latency_p95_ms=("latency_ms", lambda x: np.percentile(x, 95)),
            throughput_mean=("throughput_imgs_sec", "mean"),
            memory_mean_mb=("memory_mb", "mean"),
            n_runs=("run_id", "count"),
        ).reset_index()

        return agg_df

    def split_data(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide dados em conjuntos de treino e teste.

        Utilizado para validação de modelos de previsão de latência
        ou análise de regressão entre FLOPs e latência real.

        Args:
            df: DataFrame de entrada. Se None, usa self.data.
            test_size: Fração reservada para teste.

        Returns:
            Tupla (train_df, test_df).
        """
        from sklearn.model_selection import train_test_split

        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado.")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df["model_family"],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def get_resolution_impact(
        self,
        df: Optional[pd.DataFrame] = None,
        model_name: str = "efficientnet_b0",
        hardware: str = "v100",
    ) -> pd.DataFrame:
        """
        Isola o impacto da resolução para um modelo e hardware fixos.

        Conforme seção 4 do material: dobrar a resolução aumenta FLOPs
        por fator ~4 em camadas convolucionais (crescimento quadrático).

        Args:
            df: DataFrame de entrada.
            model_name: Nome do modelo a filtrar.
            hardware: Hardware a filtrar.

        Returns:
            DataFrame filtrado e ordenado por resolução.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado.")

        filtered = df[
            (df["model_name"] == model_name) & (df["hardware"] == hardware)
        ].copy()

        if filtered.empty:
            raise ValueError(
                f"Nenhum dado para modelo='{model_name}', hardware='{hardware}'."
            )

        return filtered.sort_values("input_resolution").reset_index(drop=True)
