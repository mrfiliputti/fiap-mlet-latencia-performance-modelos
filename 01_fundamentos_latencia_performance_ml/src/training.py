"""
Módulo de treinamento com profiling de performance.

Implementa o pipeline de treinamento com medição de latência em cada etapa,
cross-validation e hyperparameter tuning. Conforme Sculley et al. (2015),
o pipeline de treinamento é onde a maior parte da dívida técnica se acumula.

Referência:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning
    Systems. NeurIPS '15.
    Li, M. et al. (2014). Scaling Distributed Machine Learning with the
    Parameter Server. OSDI '14.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate

from .model import LatencyBenchmarkModel


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "logistic_regression",
    random_state: int = 42,
    **model_params: Any,
) -> tuple[LatencyBenchmarkModel, dict[str, float]]:
    """
    Treina um modelo e retorna métricas de treinamento com latência.

    Implementa o ciclo básico de treinamento conforme a seção
    'Fundamentos de Performance', medindo o tempo total de fit
    para cada algoritmo.

    Args:
        X_train: Features de treinamento (n_samples, n_features).
        y_train: Labels de treinamento (n_samples,).
        model_name: Nome do algoritmo (ver LatencyBenchmarkModel.SUPPORTED_MODELS).
        random_state: Seed para reprodutibilidade.
        **model_params: Parâmetros adicionais do modelo.

    Returns:
        Tupla (modelo_treinado, métricas_de_treinamento).
    """
    model = LatencyBenchmarkModel(
        model_name=model_name,
        random_state=random_state,
        **model_params,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    accuracy = model.score(X_train, y_train)

    metrics = {
        "model_name": model_name,
        "training_time_s": round(training_time, 4),
        "training_samples": X_train.shape[0],
        "training_accuracy": round(accuracy, 4),
        "samples_per_second": round(X_train.shape[0] / training_time, 2),
    }

    return model, metrics


def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_names: Optional[list[str]] = None,
    random_state: int = 42,
) -> list[tuple[LatencyBenchmarkModel, dict[str, float]]]:
    """
    Treina múltiplos modelos para comparação de latência.

    Implementa o protocolo de comparação de algoritmos descrito na seção
    'Complexidade de Modelos', treinando cada algoritmo sob as mesmas
    condições para comparação justa.

    Args:
        X_train: Features de treinamento.
        y_train: Labels de treinamento.
        model_names: Lista de nomes de modelos. Se None, treina todos.
        random_state: Seed para reprodutibilidade.

    Returns:
        Lista de tuplas (modelo, métricas) para cada algoritmo.
    """
    if model_names is None:
        model_names = list(LatencyBenchmarkModel.SUPPORTED_MODELS.keys())

    results = []
    for name in model_names:
        model, metrics = train_model(
            X_train, y_train, model_name=name, random_state=random_state
        )
        results.append((model, metrics))

    return results


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "logistic_regression",
    cv: int = 5,
    random_state: int = 42,
    **model_params: Any,
) -> dict[str, Any]:
    """
    Realiza cross-validation com medição de tempo por fold.

    Cross-validation permite avaliar a estabilidade do modelo e estimar
    a variância da performance. O tempo por fold também é registrado
    para entender o custo computacional do processo de validação.

    Args:
        X: Features completas.
        y: Labels completos.
        model_name: Nome do algoritmo.
        cv: Número de folds.
        random_state: Seed para reprodutibilidade.
        **model_params: Parâmetros adicionais do modelo.

    Returns:
        Dicionário com scores e tempos de cross-validation.
    """
    model = LatencyBenchmarkModel(
        model_name=model_name,
        random_state=random_state,
        **model_params,
    )

    start = time.perf_counter()
    cv_results = cross_validate(
        model.model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1"],
        return_train_score=True,
    )
    total_time = time.perf_counter() - start

    return {
        "model_name": model_name,
        "cv_folds": cv,
        "test_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "test_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "test_f1_mean": float(np.mean(cv_results["test_f1"])),
        "test_f1_std": float(np.std(cv_results["test_f1"])),
        "train_accuracy_mean": float(np.mean(cv_results["train_accuracy"])),
        "fit_time_mean_s": float(np.mean(cv_results["fit_time"])),
        "fit_time_std_s": float(np.std(cv_results["fit_time"])),
        "score_time_mean_s": float(np.mean(cv_results["score_time"])),
        "total_cv_time_s": round(total_time, 4),
    }


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest",
    param_grid: Optional[dict[str, list[Any]]] = None,
    cv: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Realiza tuning de hiperparâmetros com medição de custo computacional.

    O tuning de hiperparâmetros é uma das etapas mais custosas do pipeline
    de ML. A medição do tempo total permite avaliar o trade-off entre
    a qualidade do modelo e o custo computacional do tuning.

    Conforme Li et al. (2014), estratégias eficientes de busca de
    hiperparâmetros são essenciais para escalar sistemas de ML.

    Args:
        X_train: Features de treinamento.
        y_train: Labels de treinamento.
        model_name: Nome do algoritmo.
        param_grid: Grid de hiperparâmetros. Se None, usa grid padrão.
        cv: Número de folds para cross-validation interna.
        random_state: Seed para reprodutibilidade.

    Returns:
        Dicionário com melhores parâmetros, score e tempo de tuning.
    """
    if param_grid is None:
        param_grid = _get_default_param_grid(model_name)

    model = LatencyBenchmarkModel(
        model_name=model_name, random_state=random_state
    )

    grid_search = GridSearchCV(
        model.model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,  # single-thread para medição justa
        return_train_score=True,
    )

    start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    tuning_time = time.perf_counter() - start

    return {
        "model_name": model_name,
        "best_params": grid_search.best_params_,
        "best_score": round(float(grid_search.best_score_), 4),
        "tuning_time_s": round(tuning_time, 4),
        "n_candidates": len(grid_search.cv_results_["params"]),
        "n_fits": len(grid_search.cv_results_["params"]) * cv,
    }


def _get_default_param_grid(model_name: str) -> dict[str, list[Any]]:
    """
    Retorna grid de hiperparâmetros padrão (compacto para benchmarking rápido).

    Args:
        model_name: Nome do algoritmo.

    Returns:
        Grid de hiperparâmetros.
    """
    grids: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
        },
        "gradient_boosting": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf"],
        },
    }
    return grids.get(model_name, {"C": [1.0]})
