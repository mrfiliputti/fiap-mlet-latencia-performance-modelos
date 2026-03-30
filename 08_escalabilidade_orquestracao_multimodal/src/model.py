"""
Simuladores de orquestração multimodal e monitoramento de SLOs.

Implementa:
- ``MultimodalOrchestrator``: simula um pipeline multimodal com workers
  de texto, imagem e áudio conforme a arquitetura de referência
  (Figura 1 do material da aula).
- ``SLOMonitor``: calcula SLIs, SLOs e error budgets seguindo o
  framework SRE de Beyer et al. (2016).
- ``CanaryAnalyzer``: executa testes estatísticos (t-test, qui-quadrado)
  para validação de canary deployments conforme Tabela 2 do material.

Referências:
    Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
    Paleyes, A., Urma, R.-G., Lawrence, N. D. (2022).
        Challenges in Deploying Machine Learning. ACM Computing Surveys.
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems. NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import factorial


# ================================================================== #
#  SLO Monitor                                                        #
# ================================================================== #


@dataclass
class SLOResult:
    """Resultado da avaliação de um SLO."""

    name: str
    target: float
    actual: float
    met: bool
    error_budget_total: float
    error_budget_remaining: float
    error_budget_pct: float


class SLOMonitor:
    """Monitora SLIs, SLOs e error budgets para sistemas de ML serving.

    Implementa o framework de SRE formalizado por Beyer et al. (2016),
    adaptado para métricas de inferência de ML conforme discutido na
    seção 'Fundamentos de SRE para Machine Learning' do material.

    Attributes:
        slo_latency_p99_ms: Meta de latência P99 em ms.
        slo_availability: Meta de disponibilidade (fração).
        window_days: Janela de avaliação em dias.
    """

    def __init__(
        self,
        slo_latency_p99_ms: float = 200.0,
        slo_availability: float = 0.999,
        window_days: int = 30,
    ) -> None:
        self.slo_latency_p99_ms = slo_latency_p99_ms
        self.slo_availability = slo_availability
        self.window_days = window_days

    # ---- SLIs --------------------------------------------------------

    def calculate_slis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula Service Level Indicators a partir de métricas.

        SLIs calculados:
        - latency_p50, latency_p95, latency_p99, latency_max
        - availability (1 - taxa de erro)
        - throughput_rps (aprox.)
        - error_rate

        Args:
            df: DataFrame com colunas ``total_latency_ms`` e ``is_error``.

        Returns:
            Dicionário de SLIs.
        """
        latencies = df["total_latency_ms"]
        slis: Dict[str, float] = {
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "latency_max": float(latencies.max()),
            "error_rate": float(df["is_error"].mean()),
            "availability": float(1 - df["is_error"].mean()),
            "throughput_rps": _estimate_throughput(df),
        }
        return slis

    # ---- SLOs --------------------------------------------------------

    def evaluate_slos(self, df: pd.DataFrame) -> List[SLOResult]:
        """Avalia se os SLOs estão sendo cumpridos.

        Retorna resultados para SLO de latência P99 e SLO de
        disponibilidade, incluindo error budgets conforme o modelo
        descrito por Beyer et al. (2016).

        Args:
            df: DataFrame com métricas de serving.

        Returns:
            Lista de ``SLOResult``.
        """
        slis = self.calculate_slis(df)
        results: List[SLOResult] = []

        # SLO de latência P99
        latency_actual = slis["latency_p99"]
        latency_met = latency_actual <= self.slo_latency_p99_ms
        # Error budget: fração de requisições acima do SLO permitida
        total_reqs = len(df)
        allowed_violations = total_reqs * (1 - self.slo_availability)
        actual_violations = (df["total_latency_ms"] > self.slo_latency_p99_ms).sum()
        budget_remaining = max(0.0, allowed_violations - actual_violations)
        budget_pct = (budget_remaining / allowed_violations * 100) if allowed_violations > 0 else 0.0

        results.append(SLOResult(
            name="latency_p99",
            target=self.slo_latency_p99_ms,
            actual=latency_actual,
            met=latency_met,
            error_budget_total=allowed_violations,
            error_budget_remaining=budget_remaining,
            error_budget_pct=budget_pct,
        ))

        # SLO de disponibilidade
        avail_actual = slis["availability"]
        avail_met = avail_actual >= self.slo_availability
        total_minutes = self.window_days * 24 * 60
        allowed_downtime_min = total_minutes * (1 - self.slo_availability)
        used_downtime_min = total_minutes * slis["error_rate"]
        remaining_min = max(0.0, allowed_downtime_min - used_downtime_min)
        budget_pct_avail = (remaining_min / allowed_downtime_min * 100) if allowed_downtime_min > 0 else 0.0

        results.append(SLOResult(
            name="availability",
            target=self.slo_availability,
            actual=avail_actual,
            met=avail_met,
            error_budget_total=allowed_downtime_min,
            error_budget_remaining=remaining_min,
            error_budget_pct=budget_pct_avail,
        ))
        return results

    # ---- Erlang-C ----------------------------------------------------

    @staticmethod
    def erlang_c(c: int, rho: float) -> float:
        """Calcula a probabilidade Erlang-C de espera na fila.

        Implementa a fórmula de Erlang-C para sistemas M/M/c conforme
        apresentado na seção 'Azure ML Managed Endpoints: Auto-scaling
        em Produção' do material da aula:

            C(c, ρ) = [(cρ)^c / c!] / { (cρ)^c/c! * 1/(1-ρ)
                       + Σ_{k=0}^{c-1} (cρ)^k / k! }     com ρ < 1

        Args:
            c: Número de servidores (réplicas).
            rho: Utilização do sistema (λ / (c·μ)).  Deve ser < 1.

        Returns:
            Probabilidade de espera (Erlang-C).
        """
        if rho >= 1.0:
            return 1.0  # sistema instável
        a = c * rho  # intensidade de tráfego
        numerator = (a ** c) / factorial(c, exact=True)
        sum_terms = sum((a ** k) / factorial(k, exact=True) for k in range(c))
        denominator = numerator / (1 - rho) + sum_terms
        return float(numerator / denominator) if denominator != 0 else 1.0

    @staticmethod
    def mean_queue_wait(
        lam: float, mu: float, c: int
    ) -> float:
        """Calcula tempo médio de espera na fila (Wq) via Erlang-C.

        Fórmula:  Wq = C(c, ρ) / (c·μ − λ)

        Conforme apresentado no material da aula na seção sobre
        auto-scaling e teoria de filas.

        Args:
            lam: Taxa de chegada (λ, requisições/s).
            mu: Taxa de serviço por servidor (μ, requisições/s).
            c: Número de servidores.

        Returns:
            Wq em segundos. Retorna ``inf`` se sistema saturado.
        """
        rho = lam / (c * mu)
        if rho >= 1.0:
            return float("inf")
        ec = SLOMonitor.erlang_c(c, rho)
        return ec / (c * mu - lam)


# ================================================================== #
#  Canary Analyzer                                                     #
# ================================================================== #


@dataclass
class CanaryTestResult:
    """Resultado de um teste estatístico de canary deployment."""

    metric: str
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    decision: str  # "rollback", "promote", "continue"


class CanaryAnalyzer:
    """Analisa canary deployments usando testes estatísticos.

    Implementa a Tabela 2 ('Framework de Decisão para Canary Deployment')
    do material da aula, aplicando teste-t para latência e qui-quadrado
    para taxa de erro.

    Referência:
        Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        latency_degradation_threshold: float = 0.10,
        error_increase_threshold: float = 0.50,
    ) -> None:
        self.alpha = alpha
        self.latency_degradation_threshold = latency_degradation_threshold
        self.error_increase_threshold = error_increase_threshold

    def test_latency(
        self,
        stable_latencies: np.ndarray,
        canary_latencies: np.ndarray,
    ) -> CanaryTestResult:
        """Teste-t para comparação de latências entre versões.

        A estatística t é calculada como:
            t = (X̄₁ − X̄₂) / √(s₁²/n₁ + s₂²/n₂)

        conforme descrito na seção 'Canary Deployments e Testes
        Estatísticos em Produção' do material da aula.

        Args:
            stable_latencies: Latências da versão stable.
            canary_latencies: Latências da versão canary.

        Returns:
            Resultado do teste com decisão (rollback/promote/continue).
        """
        stat, p_value = stats.ttest_ind(
            canary_latencies, stable_latencies, equal_var=False
        )
        significant = p_value < self.alpha

        # Verificar se há degradação real
        mean_diff_pct = (
            (canary_latencies.mean() - stable_latencies.mean())
            / stable_latencies.mean()
        )
        if significant and mean_diff_pct > self.latency_degradation_threshold:
            decision = "rollback"
        elif significant and mean_diff_pct < -self.latency_degradation_threshold:
            decision = "promote"
        else:
            decision = "continue"

        return CanaryTestResult(
            metric="latency_p99",
            test_name="welch_t_test",
            statistic=float(stat),
            p_value=float(p_value),
            significant=significant,
            decision=decision,
        )

    def test_error_rate(
        self,
        stable_errors: int,
        stable_total: int,
        canary_errors: int,
        canary_total: int,
    ) -> CanaryTestResult:
        """Teste qui-quadrado para comparação de taxas de erro.

        Implementa o teste qui-quadrado de independência conforme
        descrito na seção 'Canary Deployments e Testes Estatísticos
        em Produção' do material da aula.

        Args:
            stable_errors: Erros na versão stable.
            stable_total: Total de requisições stable.
            canary_errors: Erros na versão canary.
            canary_total: Total de requisições canary.

        Returns:
            Resultado do teste com decisão.
        """
        # Tabela de contingência 2×2
        table = np.array([
            [stable_errors, stable_total - stable_errors],
            [canary_errors, canary_total - canary_errors],
        ])
        stat, p_value, _, _ = stats.chi2_contingency(table)
        significant = p_value < self.alpha

        stable_rate = stable_errors / stable_total if stable_total > 0 else 0
        canary_rate = canary_errors / canary_total if canary_total > 0 else 0
        increase_pct = (
            (canary_rate - stable_rate) / stable_rate if stable_rate > 0 else 0
        )

        if significant and increase_pct > self.error_increase_threshold:
            decision = "rollback"
        elif significant and increase_pct < -self.error_increase_threshold:
            decision = "promote"
        else:
            decision = "continue"

        return CanaryTestResult(
            metric="error_rate",
            test_name="chi_squared",
            statistic=float(stat),
            p_value=float(p_value),
            significant=significant,
            decision=decision,
        )

    def full_analysis(
        self, df: pd.DataFrame
    ) -> Dict[str, CanaryTestResult]:
        """Executa análise completa de canary sobre DataFrame de serving.

        Args:
            df: DataFrame com colunas ``model_version``,
                ``total_latency_ms``, ``is_error``.

        Returns:
            Dicionário com resultados dos testes ``latency`` e ``error_rate``.
        """
        stable = df[df["model_version"] == "v1-stable"]
        canary = df[df["model_version"] == "v2-canary"]

        latency_result = self.test_latency(
            stable["total_latency_ms"].values,
            canary["total_latency_ms"].values,
        )
        error_result = self.test_error_rate(
            stable_errors=int(stable["is_error"].sum()),
            stable_total=len(stable),
            canary_errors=int(canary["is_error"].sum()),
            canary_total=len(canary),
        )
        return {"latency": latency_result, "error_rate": error_result}


# ================================================================== #
#  Multimodal Orchestrator (simulador)                                 #
# ================================================================== #


@dataclass
class InferenceResult:
    """Resultado de inferência de um worker."""

    modality: str
    latency_ms: float
    confidence: float
    is_error: bool
    prediction: Any = None


class MultimodalOrchestrator:
    """Simula um pipeline de inferência multimodal.

    Implementa a arquitetura de referência (Figura 1) com
    API Gateway → Request Router → Workers (Text/Image/Audio) → Aggregator,
    conforme discutido na seção 'Orquestração de Pipelines Multimodais'.

    Suporta fusão tardia (late fusion) de resultados, onde cada
    modalidade é processada independentemente e os resultados são
    combinados no Aggregator.

    Referência:
        Paleyes, A., Urma, R.-G., Lawrence, N. D. (2022).
            Challenges in Deploying Machine Learning. ACM Computing Surveys.
    """

    def __init__(
        self,
        fusion_strategy: str = "late",
        seed: int = 42,
    ) -> None:
        """
        Args:
            fusion_strategy: Estratégia de fusão ('late', 'early', 'intermediate').
            seed: Semente para reprodutibilidade.
        """
        if fusion_strategy not in ("late", "early", "intermediate"):
            raise ValueError(f"Estratégia inválida: {fusion_strategy}")
        self.fusion_strategy = fusion_strategy
        self.rng = np.random.default_rng(seed)
        self._workers: Dict[str, Dict[str, Any]] = {
            "text":  {"mean_latency": 35, "std_latency": 10, "error_rate": 0.01},
            "image": {"mean_latency": 60, "std_latency": 20, "error_rate": 0.02},
            "audio": {"mean_latency": 80, "std_latency": 25, "error_rate": 0.015},
        }

    def _simulate_worker(self, modality: str) -> InferenceResult:
        """Simula execução de um worker especializado."""
        cfg = self._workers[modality]
        latency = max(1.0, self.rng.normal(cfg["mean_latency"], cfg["std_latency"]))
        is_error = bool(self.rng.random() < cfg["error_rate"])
        confidence = float(self.rng.beta(8, 2)) if not is_error else 0.0
        return InferenceResult(
            modality=modality,
            latency_ms=round(latency, 2),
            confidence=round(confidence, 4),
            is_error=is_error,
        )

    def predict(
        self, modalities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Executa inferência multimodal (simulada).

        Processa as modalidades solicitadas em paralelo (simulado) e
        aplica fusão tardia (late fusion) para combinar resultados.

        Args:
            modalities: Lista de modalidades a processar.
                Se ``None``, processa todas.

        Returns:
            Dicionário com resultados individuais e agregados.
        """
        if modalities is None:
            modalities = list(self._workers.keys())

        results = {m: self._simulate_worker(m) for m in modalities}

        # Late fusion: média ponderada de confidências
        valid = [r for r in results.values() if not r.is_error]
        if valid:
            fused_confidence = float(np.mean([r.confidence for r in valid]))
            total_latency = max(r.latency_ms for r in results.values())  # paralelo
        else:
            fused_confidence = 0.0
            total_latency = sum(r.latency_ms for r in results.values())

        return {
            "individual": {m: r.__dict__ for m, r in results.items()},
            "fused_confidence": round(fused_confidence, 4),
            "total_latency_ms": round(total_latency, 2),
            "has_error": any(r.is_error for r in results.values()),
            "fusion_strategy": self.fusion_strategy,
        }

    def score(self, n_requests: int = 100) -> Dict[str, float]:
        """Calcula métricas agregadas sobre N requisições.

        Args:
            n_requests: Número de requisições a simular.

        Returns:
            Dicionário com P50, P95, P99, error_rate.
        """
        latencies = []
        errors = 0
        for _ in range(n_requests):
            result = self.predict()
            latencies.append(result["total_latency_ms"])
            if result["has_error"]:
                errors += 1
        latencies_arr = np.array(latencies)
        return {
            "latency_p50": float(np.percentile(latencies_arr, 50)),
            "latency_p95": float(np.percentile(latencies_arr, 95)),
            "latency_p99": float(np.percentile(latencies_arr, 99)),
            "error_rate": errors / n_requests,
            "mean_latency": float(latencies_arr.mean()),
        }


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def _estimate_throughput(df: pd.DataFrame) -> float:
    """Estima throughput (req/s) a partir de timestamps."""
    if "timestamp" not in df.columns or len(df) < 2:
        return 0.0
    duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    return len(df) / duration if duration > 0 else 0.0
