"""
Módulo de definição de modelos com benchmarking integrado de latência.

Implementa wrapper de modelos de classificação com medição automática de
latência de inferência, permitindo comparar a complexidade computacional
de diferentes algoritmos conforme discutido na seção 'Teoria-Chave'.

Referência:
    Reddi, V. J. et al. (2020). MLPerf Inference Benchmark. ISCA '20.
    Dean, J. & Barroso, L. A. (2013). The Tail at Scale.
    Communications of the ACM, 56(2), 74-80.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class LatencyBenchmarkModel:
    """
    Wrapper de modelo de classificação com benchmarking de latência.

    Encapsula diferentes algoritmos de ML e mede automaticamente a latência
    de inferência (L_inferencia), permitindo comparações justas conforme os
    princípios do MLPerf Benchmark (Reddi et al., 2020).

    A complexidade de inferência varia entre algoritmos:
    - Logistic Regression: O(d) — linear no número de features
    - Random Forest: O(t · log n) — t = número de árvores
    - Gradient Boosting: O(t · profundidade) — sequencial por natureza
    - SVM (kernel RBF): O(n_sv · d) — n_sv = número de support vectors

    Attributes:
        model_name: Nome do algoritmo.
        model: Instância do classificador sklearn.
        is_fitted: Se o modelo já foi treinado.
        latency_history: Histórico de medições de latência.
    """

    SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
    }

    def __init__(
        self,
        model_name: str = "logistic_regression",
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        """
        Inicializa o modelo com configuração de benchmarking.

        Args:
            model_name: Nome do algoritmo. Suportados: 'logistic_regression',
                'random_forest', 'gradient_boosting', 'svm'.
            random_state: Seed para reprodutibilidade.
            **model_params: Parâmetros adicionais para o classificador.

        Raises:
            ValueError: Se model_name não é suportado.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não suportado. "
                f"Escolha entre: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.is_fitted = False
        self.latency_history: list[dict[str, Any]] = []

        # Configurar parâmetros padrão
        default_params = self._get_default_params(model_name, random_state)
        default_params.update(model_params)

        model_cls = self.SUPPORTED_MODELS[model_name]
        self.model = model_cls(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LatencyBenchmarkModel:
        """
        Treina o modelo e registra a latência de treinamento.

        A latência de treinamento não afeta diretamente a experiência do
        usuário em produção, mas é relevante para retraining pipelines
        conforme Sculley et al. (2015).

        Args:
            X: Features de treinamento (n_samples, n_features).
            y: Labels de treinamento (n_samples,).

        Returns:
            self para encadeamento.
        """
        start = time.perf_counter()
        self.model.fit(X, y)
        elapsed = time.perf_counter() - start

        self.is_fitted = True
        self._log_latency("fit", X.shape[0], elapsed)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predição com medição de latência de inferência.

        Mede L_inferencia, o componente mais crítico da latência em produção.
        Conforme Crankshaw et al. (2017), a latência de inferência é
        frequentemente o componente dominante em serving systems.

        Args:
            X: Features para predição (n_samples, n_features).

        Returns:
            Array de predições (n_samples,).

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        start = time.perf_counter()
        predictions = self.model.predict(X)
        elapsed = time.perf_counter() - start

        self._log_latency("predict", X.shape[0], elapsed)
        return predictions

    def predict_single(self, x: np.ndarray) -> tuple[int, float]:
        """
        Predição para uma única amostra com medição de latência.

        Simula cenário de serving online onde amostras chegam uma a uma.
        Permite medir a latência individual (tail latency) conforme
        Dean & Barroso (2013), "The Tail at Scale".

        Args:
            x: Feature vector de uma amostra (n_features,) ou (1, n_features).

        Returns:
            Tupla (predição, latência_ms).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        if x.ndim == 1:
            x = x.reshape(1, -1)

        start = time.perf_counter()
        prediction = self.model.predict(x)
        elapsed = time.perf_counter() - start

        latency_ms = elapsed * 1000
        self._log_latency("predict_single", 1, elapsed)
        return int(prediction[0]), latency_ms

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula acurácia do modelo.

        Args:
            X: Features de teste.
            y: Labels verdadeiros.

        Returns:
            Acurácia (0.0 a 1.0).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return float(self.model.score(X, y))

    def benchmark_inference(
        self, X: np.ndarray, n_iterations: int = 100
    ) -> dict[str, float]:
        """
        Executa benchmark de latência de inferência com múltiplas iterações.

        Implementa o protocolo de benchmarking do MLPerf (Reddi et al., 2020):
        múltiplas execuções para capturar a distribuição de latência e calcular
        percentis P50, P95, P99.

        Args:
            X: Features para benchmarking.
            n_iterations: Número de iterações para medir.

        Returns:
            Dicionário com estatísticas de latência (ms):
            - mean, std, min, max, p50, p95, p99
            - throughput (amostras/segundo)
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        latencies: list[float] = []

        # Warm-up: 5 iterações descartadas (evita cold start)
        for _ in range(5):
            self.model.predict(X)

        # Medição real
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.model.predict(X)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # converter para ms

        arr = np.array(latencies)
        total_samples = X.shape[0] * n_iterations
        total_time_s = arr.sum() / 1000

        return {
            "model": self.model_name,
            "n_samples": X.shape[0],
            "n_iterations": n_iterations,
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "throughput_samples_per_sec": total_samples / total_time_s,
        }

    def get_model_info(self) -> dict[str, Any]:
        """
        Retorna informações do modelo relevantes para análise de latência.

        Returns:
            Dicionário com nome, parâmetros e status do modelo.
        """
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "params": self.model.get_params(),
            "n_latency_records": len(self.latency_history),
        }

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------
    @staticmethod
    def _get_default_params(model_name: str, random_state: int) -> dict[str, Any]:
        """Retorna parâmetros padrão para cada modelo."""
        defaults: dict[str, dict[str, Any]] = {
            "logistic_regression": {
                "random_state": random_state,
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            "random_forest": {
                "random_state": random_state,
                "n_estimators": 100,
                "n_jobs": 1,  # single-thread para medição justa de latência
            },
            "gradient_boosting": {
                "random_state": random_state,
                "n_estimators": 100,
                "max_depth": 3,
            },
            "svm": {
                "random_state": random_state,
                "kernel": "rbf",
                "probability": False,
            },
        }
        return defaults.get(model_name, {"random_state": random_state})

    def _log_latency(
        self, operation: str, n_samples: int, elapsed_seconds: float
    ) -> None:
        """Registra medição de latência."""
        self.latency_history.append(
            {
                "operation": operation,
                "n_samples": n_samples,
                "latency_ms": round(elapsed_seconds * 1000, 4),
                "throughput_per_sec": n_samples / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0,
            }
        )
