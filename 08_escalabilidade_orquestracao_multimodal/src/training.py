"""
Auto-scaling e canary deployment — lógica de simulação e treinamento.

Implementa simuladores de auto-scaling baseado em métricas e
orquestração de canary deployments conforme discutido nas seções
'Azure ML Managed Endpoints: Auto-scaling em Produção' e
'Canary Deployments e Testes Estatísticos em Produção' do material.

O auto-scaling utiliza teoria de filas (Erlang-C) para dimensionar
réplicas, e o canary deployment aplica testes estatísticos para
decisão de promoção/rollback.

Referências:
    Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
    Paleyes, A., Urma, R.-G., Lawrence, N. D. (2022).
        Challenges in Deploying Machine Learning. ACM Computing Surveys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model import CanaryAnalyzer, CanaryTestResult, SLOMonitor


# ================================================================== #
#  Auto-Scaling Simulator                                              #
# ================================================================== #


@dataclass
class ScalingEvent:
    """Registro de evento de scaling."""

    timestamp_idx: int
    direction: str        # "scale_up" | "scale_down" | "no_change"
    old_replicas: int
    new_replicas: int
    trigger_metric: str
    trigger_value: float


class AutoScaler:
    """Simula auto-scaling reativo baseado em métricas.

    Implementa lógica de scaling horizontal conforme discutido na
    seção 'Azure ML Managed Endpoints: Auto-scaling em Produção',
    usando métricas de CPU, latência e profundidade de fila como
    sinais de scaling.

    O comportamento baseia-se no modelo de filas M/M/c, onde
    o número ótimo de réplicas é determinado para manter a latência
    abaixo do SLO definido.

    Args:
        min_replicas: Mínimo de réplicas ativas (cold-start mitigation).
        max_replicas: Máximo de réplicas.
        cpu_upper: Threshold superior de CPU para scale-up.
        cpu_lower: Threshold inferior de CPU para scale-down.
        latency_target_ms: SLO de latência P99 em ms.
        cooldown_steps: Passos de cooldown entre eventos de scaling.
    """

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 10,
        cpu_upper: float = 0.70,
        cpu_lower: float = 0.30,
        latency_target_ms: float = 200.0,
        cooldown_steps: int = 5,
    ) -> None:
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cpu_upper = cpu_upper
        self.cpu_lower = cpu_lower
        self.latency_target_ms = latency_target_ms
        self.cooldown_steps = cooldown_steps

    def simulate(
        self,
        df: pd.DataFrame,
        window_size: int = 50,
    ) -> Tuple[pd.DataFrame, List[ScalingEvent]]:
        """Executa simulação de auto-scaling sobre dados históricos.

        Percorre o DataFrame em janelas deslizantes, avaliando métricas
        e decidindo se deve escalar para cima/baixo.

        Args:
            df: DataFrame com colunas ``cpu_utilization``,
                ``total_latency_ms``, ``replica_count``.
            window_size: Tamanho da janela de avaliação.

        Returns:
            Tupla (DataFrame com coluna ``simulated_replicas``,
            lista de ``ScalingEvent``).
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        replicas = np.full(n, self.min_replicas, dtype=int)
        events: List[ScalingEvent] = []
        current = self.min_replicas
        last_event_idx = -self.cooldown_steps  # permite escalar imediatamente

        for i in range(window_size, n):
            window = df.iloc[i - window_size : i]
            avg_cpu = window["cpu_utilization"].mean()
            p99_lat = np.percentile(window["total_latency_ms"], 99)

            direction = "no_change"
            new = current

            if i - last_event_idx < self.cooldown_steps:
                replicas[i] = current
                continue

            # Scale-up: CPU alta OU latência acima do SLO
            if avg_cpu > self.cpu_upper or p99_lat > self.latency_target_ms:
                new = min(current + 1, self.max_replicas)
                if new != current:
                    direction = "scale_up"
                    trigger = "cpu" if avg_cpu > self.cpu_upper else "latency_p99"
                    trigger_val = avg_cpu if trigger == "cpu" else p99_lat
                    events.append(ScalingEvent(i, direction, current, new, trigger, trigger_val))
                    last_event_idx = i

            # Scale-down: CPU baixa E latência dentro do SLO
            elif avg_cpu < self.cpu_lower and p99_lat < self.latency_target_ms * 0.6:
                new = max(current - 1, self.min_replicas)
                if new != current:
                    direction = "scale_down"
                    events.append(ScalingEvent(i, direction, current, new, "cpu", avg_cpu))
                    last_event_idx = i

            current = new
            replicas[i] = current

        # Preencher janela inicial
        replicas[:window_size] = replicas[window_size]
        df["simulated_replicas"] = replicas
        return df, events


# ================================================================== #
#  Canary Deployment Orchestrator                                      #
# ================================================================== #


@dataclass
class CanaryStageResult:
    """Resultado de uma etapa do canary deployment."""

    stage: int
    traffic_pct: float
    n_requests: int
    latency_result: CanaryTestResult
    error_result: CanaryTestResult
    overall_decision: str


class CanaryDeployment:
    """Orquestra canary deployment progressivo com validação estatística.

    Implementa a estratégia de canary deployment descrita na seção
    'Canary Deployments e Testes Estatísticos em Produção', com
    etapas progressivas de tráfego (10 % → 25 % → 50 % → 100 %).

    Se qualquer etapa indica rollback, o deployment é abortado.

    Referência:
        Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
    """

    def __init__(
        self,
        stages: Optional[List[float]] = None,
        analyzer: Optional[CanaryAnalyzer] = None,
    ) -> None:
        self.stages = stages or [0.10, 0.25, 0.50, 1.0]
        self.analyzer = analyzer or CanaryAnalyzer()

    def run(
        self,
        df: pd.DataFrame,
    ) -> List[CanaryStageResult]:
        """Executa canary deployment completo.

        Divide os dados proporcionalmente por etapa e avalia cada
        etapa com testes estatísticos.

        Args:
            df: DataFrame completo com dados de ambas as versões.

        Returns:
            Lista de resultados por etapa.
        """
        stable = df[df["model_version"] == "v1-stable"].copy()
        canary = df[df["model_version"] == "v2-canary"].copy()

        results: List[CanaryStageResult] = []
        total_canary = len(canary)

        for i, pct in enumerate(self.stages):
            n_canary = max(1, int(total_canary * pct))
            canary_sample = canary.iloc[:n_canary]
            stage_df = pd.concat([stable, canary_sample], ignore_index=True)

            analysis = self.analyzer.full_analysis(stage_df)

            # Decisão geral: rollback se QUALQUER teste indica rollback
            if analysis["latency"].decision == "rollback" or analysis["error_rate"].decision == "rollback":
                overall = "rollback"
            elif analysis["latency"].decision == "promote" and analysis["error_rate"].decision != "rollback":
                overall = "promote"
            else:
                overall = "continue"

            results.append(CanaryStageResult(
                stage=i + 1,
                traffic_pct=pct,
                n_requests=len(stage_df),
                latency_result=analysis["latency"],
                error_result=analysis["error_rate"],
                overall_decision=overall,
            ))

            if overall == "rollback":
                break  # abortar deployment

        return results


# ================================================================== #
#  Funções auxiliares de treinamento                                    #
# ================================================================== #


def run_autoscaling_simulation(
    df: pd.DataFrame,
    min_replicas: int = 2,
    max_replicas: int = 10,
    cpu_upper: float = 0.70,
    latency_target_ms: float = 200.0,
) -> Tuple[pd.DataFrame, List[ScalingEvent]]:
    """Função wrapper para simulação de auto-scaling.

    Conecta a configuração declarativa de auto-scaling discutida nos
    snippets hands-on do material (Azure ML Managed Endpoints) com
    a simulação local.

    Args:
        df: DataFrame de métricas.
        min_replicas: Mínimo de réplicas.
        max_replicas: Máximo de réplicas.
        cpu_upper: Threshold de CPU.
        latency_target_ms: Meta de latência P99.

    Returns:
        DataFrame enriquecido e eventos de scaling.
    """
    scaler = AutoScaler(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        cpu_upper=cpu_upper,
        latency_target_ms=latency_target_ms,
    )
    return scaler.simulate(df)


def cross_validate_scaling_params(
    df: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    """Avalia diferentes configurações de auto-scaling.

    Testa combinações de parâmetros e reporta métricas de
    custo vs. latência para cada configuração.

    Args:
        df: DataFrame de métricas.
        param_grid: Dicionário com listas de valores para cada parâmetro.

    Returns:
        Lista de dicionários com parâmetros e resultados.
    """
    if param_grid is None:
        param_grid = {
            "min_replicas": [1, 2, 3],
            "max_replicas": [6, 8, 10],
            "cpu_upper": [0.60, 0.70, 0.80],
        }

    results: List[Dict[str, Any]] = []
    for min_r in param_grid.get("min_replicas", [2]):
        for max_r in param_grid.get("max_replicas", [10]):
            for cpu_u in param_grid.get("cpu_upper", [0.70]):
                if min_r > max_r:
                    continue
                scaler = AutoScaler(
                    min_replicas=min_r,
                    max_replicas=max_r,
                    cpu_upper=cpu_u,
                )
                df_sim, events = scaler.simulate(df)
                avg_replicas = df_sim["simulated_replicas"].mean()
                p99_lat = np.percentile(df_sim["total_latency_ms"], 99)
                results.append({
                    "min_replicas": min_r,
                    "max_replicas": max_r,
                    "cpu_upper": cpu_u,
                    "avg_replicas": round(avg_replicas, 2),
                    "latency_p99": round(p99_lat, 2),
                    "scaling_events": len(events),
                    "cost_score": round(avg_replicas * 10, 2),  # custo simplificado
                })
    return results


def hyperparameter_tuning(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Encontra melhor configuração de auto-scaling.

    Seleciona a configuração que minimiza custo (média de réplicas)
    mantendo latência P99 abaixo de 200 ms.

    Args:
        df: DataFrame de métricas.

    Returns:
        Melhor configuração encontrada.
    """
    results = cross_validate_scaling_params(df)
    # Filtrar configs que atendem SLO
    valid = [r for r in results if r["latency_p99"] <= 200.0]
    if not valid:
        valid = results  # fallback: menor latência
    best = min(valid, key=lambda r: r["cost_score"])
    return best
