"""
Módulo de simuladores de paralelismo e serving para inferência de LLMs.

Implementa simuladores pedagógicos de:
- Tensor Parallelism (Shoeybi et al., 2019 — Megatron-LM)
- Pipeline Parallelism com micro-batching (Huang et al., 2019 — GPipe)
- PagedAttention KV-Cache Manager (Kwon et al., 2023 — vLLM)
- Continuous Batching Scheduler (Yu et al., 2022 — Orca)
- Speculative Decoding (Leviathan et al., 2023)

Estes simuladores executam em CPU e modelam o *comportamento* das
estratégias de paralelismo sem requerer múltiplas GPUs, permitindo
compreensão profunda dos conceitos em ambiente pedagógico.

Referências:
    Shoeybi, M. et al. (2019). Megatron-LM: Training Multi-Billion
        Parameter Language Models Using Model Parallelism.
    Huang, Y. et al. (2019). GPipe: Efficient Training of Giant Neural
        Networks using Pipeline Parallelism. NeurIPS 2019.
    Kwon, W. et al. (2023). Efficient Memory Management for Large Language
        Model Serving with PagedAttention. SOSP 2023.
    Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.
    Leviathan, Y. et al. (2023). Fast Inference from Transformers via
        Speculative Decoding. ICML 2023.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =========================================================================
# 1. Tensor Parallelism Simulator
# =========================================================================

class TensorParallelismSimulator:
    """
    Simula tensor parallelism conforme Megatron-LM (Shoeybi et al., 2019).

    Implementa particionamento de matrizes por colunas e por linhas,
    conforme discutido na seção 'Tensor Parallelism para Large Language
    Models' do material da aula. Uma matriz W ∈ R^{m×n} é dividida em p
    partições e cada 'dispositivo' processa sua partição.

    O overhead de comunicação (all-reduce) é simulado com um fator
    de latência configurável, modelando a dependência da eficiência
    na largura de banda de interconexão (NVLink vs. InfiniBand).

    Attributes:
        num_devices: Número de GPUs simuladas.
        comm_latency_factor: Fator de overhead de comunicação (0–1).
    """

    def __init__(
        self,
        num_devices: int = 2,
        comm_latency_factor: float = 0.05,
    ) -> None:
        if num_devices < 1:
            raise ValueError("num_devices must be >= 1")
        self.num_devices = num_devices
        self.comm_latency_factor = comm_latency_factor

    def partition_column_wise(
        self, weight_matrix: np.ndarray
    ) -> list[np.ndarray]:
        """Particiona matriz de pesos por colunas (Megatron-LM column split).

        Conforme seção 'Tensor Parallelism para LLMs': W = [W_1 | W_2 | ... | W_p],
        onde cada W_i ∈ R^{m × n/p}. Cada dispositivo computa Y_i = X @ W_i.

        Args:
            weight_matrix: Matriz (m, n) a particionar.

        Returns:
            Lista de p partições de forma (m, n/p).
        """
        m, n = weight_matrix.shape
        if n % self.num_devices != 0:
            raise ValueError(
                f"Dimensão de colunas ({n}) deve ser divisível por "
                f"num_devices ({self.num_devices})"
            )
        return np.array_split(weight_matrix, self.num_devices, axis=1)

    def partition_row_wise(
        self, weight_matrix: np.ndarray
    ) -> list[np.ndarray]:
        """Particiona matriz de pesos por linhas (Megatron-LM row split).

        Conforme seção 'Tensor Parallelism para LLMs': a matriz é dividida
        ao longo das linhas e uma operação all-reduce é necessária para somar
        contribuições parciais.

        Args:
            weight_matrix: Matriz (m, n) a particionar.

        Returns:
            Lista de p partições de forma (m/p, n).
        """
        m, n = weight_matrix.shape
        if m % self.num_devices != 0:
            raise ValueError(
                f"Dimensão de linhas ({m}) deve ser divisível por "
                f"num_devices ({self.num_devices})"
            )
        return np.array_split(weight_matrix, self.num_devices, axis=0)

    def simulate_forward_column_parallel(
        self, X: np.ndarray, weight_matrix: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Simula forward pass com particionamento por colunas.

        Cada dispositivo computa Y_i = X @ W_i, resultados são concatenados.
        Implementa padrão usado em projeções Q, K, V do self-attention
        (Shoeybi et al., 2019).

        Args:
            X: Input de forma (batch, m).
            weight_matrix: Pesos de forma (m, n).

        Returns:
            Tupla (output, metrics) onde output tem forma (batch, n).
        """
        partitions = self.partition_column_wise(weight_matrix)

        partial_results = []
        device_times = []
        for i, W_i in enumerate(partitions):
            start = time.perf_counter()
            Y_i = X @ W_i
            device_times.append(time.perf_counter() - start)
            partial_results.append(Y_i)

        # Simula comunicação de concatenação
        start = time.perf_counter()
        output = np.concatenate(partial_results, axis=1)
        comm_time = time.perf_counter() - start

        # Simula overhead adicional de comunicação
        simulated_comm = max(device_times) * self.comm_latency_factor

        metrics = {
            "device_compute_times": device_times,
            "max_compute_time": max(device_times),
            "comm_time_actual": comm_time,
            "comm_time_simulated": simulated_comm,
            "total_time": max(device_times) + simulated_comm,
            "speedup_vs_single": self._compute_single_device_time(X, weight_matrix)
            / (max(device_times) + simulated_comm + 1e-12),
        }
        return output, metrics

    def simulate_forward_row_parallel(
        self, X: np.ndarray, weight_matrix: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Simula forward pass com particionamento por linhas + all-reduce.

        Cada dispositivo recebe subset de X e computa parcial, seguido de
        all-reduce (soma). Implementa padrão da projeção de saída do
        self-attention (Shoeybi et al., 2019).

        Args:
            X: Input de forma (batch, m).
            weight_matrix: Pesos de forma (m, n).

        Returns:
            Tupla (output, metrics) com resultado e métricas de tempo.
        """
        partitions = self.partition_row_wise(weight_matrix)
        m = weight_matrix.shape[0]
        chunk_size = m // self.num_devices

        partial_results = []
        device_times = []
        for i, W_i in enumerate(partitions):
            X_i = X[:, i * chunk_size: (i + 1) * chunk_size]
            start = time.perf_counter()
            Y_i = X_i @ W_i
            device_times.append(time.perf_counter() - start)
            partial_results.append(Y_i)

        # All-reduce: soma das contribuições parciais
        start = time.perf_counter()
        output = sum(partial_results)
        comm_time = time.perf_counter() - start
        simulated_comm = max(device_times) * self.comm_latency_factor

        metrics = {
            "device_compute_times": device_times,
            "max_compute_time": max(device_times),
            "comm_time_actual": comm_time,
            "comm_time_simulated": simulated_comm,
            "total_time": max(device_times) + simulated_comm,
        }
        return output, metrics

    def _compute_single_device_time(
        self, X: np.ndarray, W: np.ndarray
    ) -> float:
        """Tempo de referência com dispositivo único."""
        start = time.perf_counter()
        _ = X @ W
        return time.perf_counter() - start


# =========================================================================
# 2. Pipeline Parallelism Simulator
# =========================================================================

@dataclass
class PipelineStage:
    """Representa um estágio do pipeline (grupo de camadas).

    Conforme discutido na seção 'Pipeline Parallelism e Micro-batching'
    (Huang et al., 2019 — GPipe).

    Attributes:
        stage_id: Identificador do estágio.
        weight: Matriz de pesos simulada.
        compute_time_ms: Tempo simulado de computação (ms).
    """

    stage_id: int
    weight: np.ndarray
    compute_time_ms: float = 0.0


class PipelineParallelismSimulator:
    """
    Simula pipeline parallelism com micro-batching (GPipe).

    Implementa o conceito de pipeline bubble e a técnica de micro-batching
    conforme discutido na seção 'Pipeline Parallelism e Micro-batching'
    do material da aula. A fração de tempo perdida em bubbles é
    aproximadamente (P-1)/(M+P-1), onde P = estágios e M = micro-batches.

    Referência:
        Huang, Y. et al. (2019). GPipe: Efficient Training of Giant
        Neural Networks using Pipeline Parallelism. NeurIPS 2019.

    Attributes:
        num_stages: Número de estágios no pipeline.
        stage_compute_ms: Tempo de computação por estágio (ms).
    """

    def __init__(
        self,
        num_stages: int = 4,
        stage_compute_ms: float = 10.0,
        hidden_dim: int = 64,
    ) -> None:
        self.num_stages = num_stages
        self.stage_compute_ms = stage_compute_ms
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(42)

        # Cria pesos simulados para cada estágio
        self.stages = [
            PipelineStage(
                stage_id=i,
                weight=self.rng.standard_normal((hidden_dim, hidden_dim)).astype(
                    np.float32
                ),
                compute_time_ms=stage_compute_ms,
            )
            for i in range(num_stages)
        ]

    def compute_bubble_fraction(self, num_micro_batches: int) -> float:
        """Calcula fração de tempo perdida em pipeline bubble.

        Conforme seção 'Pipeline Parallelism e Micro-batching':
        bubble_fraction ≈ (P-1) / (M + P - 1)

        Args:
            num_micro_batches: Número de micro-batches (M).

        Returns:
            Fração de tempo desperdiçada (0–1).
        """
        p = self.num_stages
        m = num_micro_batches
        return (p - 1) / (m + p - 1)

    def simulate_pipeline(
        self,
        batch: np.ndarray,
        num_micro_batches: int = 4,
    ) -> tuple[list[np.ndarray], dict]:
        """Simula execução de pipeline com micro-batching.

        Divide o batch em micro-batches que fluem pelo pipeline,
        rastreando utilização de dispositivos e bubbles.

        Args:
            batch: Dados de entrada (batch_size, hidden_dim).
            num_micro_batches: Número de micro-batches (M).

        Returns:
            Tupla (outputs, metrics) com resultados por micro-batch e métricas.
        """
        batch_size = batch.shape[0]
        micro_batch_size = batch_size // num_micro_batches
        if micro_batch_size == 0:
            micro_batch_size = 1
            num_micro_batches = batch_size

        micro_batches = [
            batch[i * micro_batch_size: (i + 1) * micro_batch_size]
            for i in range(num_micro_batches)
        ]

        # Tabela de scheduling: schedule[t][stage] = micro_batch_id ou None
        p = self.num_stages
        m = num_micro_batches
        total_steps = m + p - 1

        schedule = np.full((total_steps, p), -1, dtype=int)
        for mb_id in range(m):
            for stage_id in range(p):
                step = mb_id + stage_id
                schedule[step, stage_id] = mb_id

        # Executa forward por estágio
        outputs: list[np.ndarray] = []
        stage_activations: dict[int, np.ndarray] = {}
        compute_start = time.perf_counter()

        for mb_id in range(m):
            x = micro_batches[mb_id].copy()
            for stage in self.stages:
                x = x @ stage.weight
                x = np.maximum(x, 0)  # ReLU simples
            outputs.append(x)

        compute_time = time.perf_counter() - compute_start

        bubble_fraction = self.compute_bubble_fraction(m)
        busy_slots = int(np.sum(schedule >= 0))
        total_slots = total_steps * p

        metrics = {
            "num_stages": p,
            "num_micro_batches": m,
            "total_pipeline_steps": total_steps,
            "bubble_fraction": bubble_fraction,
            "busy_slots": busy_slots,
            "total_slots": total_slots,
            "utilization": busy_slots / total_slots,
            "compute_time_s": compute_time,
            "schedule": schedule,
        }
        return outputs, metrics


# =========================================================================
# 3. PagedAttention KV-Cache Manager
# =========================================================================

@dataclass
class KVBlock:
    """Bloco de KV-Cache no estilo PagedAttention.

    Cada bloco armazena até block_size tokens de Key e Value cache.
    Conforme Kwon et al. (2023), blocos são a unidade de alocação
    de memória.

    Attributes:
        block_id: Identificador físico do bloco.
        is_free: Se o bloco está disponível para alocação.
        ref_count: Contagem de referências (para prefix sharing).
    """

    block_id: int
    is_free: bool = True
    ref_count: int = 0


class PagedKVCacheManager:
    """
    Gerenciador de KV-Cache com PagedAttention.

    Implementa alocação paginada de memória para KV-Cache conforme
    proposto por Kwon et al. (2023) na seção 'PagedAttention e
    Gerenciamento de KV-Cache' do material da aula.

    Memória é gerenciada em blocos de tamanho fixo. Cada sequência mantém
    uma block table mapeando posições lógicas para blocos físicos.
    Blocos podem ser compartilhados entre sequências (prefix sharing).

    Referência:
        Kwon, W. et al. (2023). Efficient Memory Management for
        Large Language Model Serving with PagedAttention. SOSP 2023.

    Attributes:
        block_size: Tokens por bloco.
        num_blocks: Total de blocos físicos disponíveis.
        blocks: Lista de blocos físicos.
        block_tables: Mapeamento sequence_id -> lista de block_ids.
    """

    def __init__(self, num_blocks: int = 256, block_size: int = 16) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.block_tables: dict[int, list[int]] = {}

    @property
    def free_blocks(self) -> int:
        """Número de blocos livres."""
        return sum(1 for b in self.blocks if b.is_free)

    @property
    def utilization(self) -> float:
        """Fração de blocos em uso."""
        used = sum(1 for b in self.blocks if not b.is_free)
        return used / self.num_blocks if self.num_blocks > 0 else 0.0

    def allocate(self, sequence_id: int, num_tokens: int) -> bool:
        """Aloca blocos para uma nova sequência.

        Calcula blocos necessários como ceil(num_tokens / block_size)
        conforme a fórmula descrita na seção PagedAttention do material.

        Args:
            sequence_id: ID da sequência.
            num_tokens: Número de tokens a alocar.

        Returns:
            True se alocação bem-sucedida, False se memória insuficiente.
        """
        needed = int(np.ceil(num_tokens / self.block_size))
        free = [b for b in self.blocks if b.is_free]

        if len(free) < needed:
            return False  # Out of memory

        allocated_ids: list[int] = []
        for i in range(needed):
            block = free[i]
            block.is_free = False
            block.ref_count = 1
            allocated_ids.append(block.block_id)

        self.block_tables[sequence_id] = allocated_ids
        return True

    def extend(self, sequence_id: int, additional_tokens: int) -> bool:
        """Estende alocação de uma sequência com tokens adicionais.

        Simula crescimento dinâmico durante geração, alocando novos
        blocos conforme necessário. É a vantagem chave do PagedAttention
        sobre alocação contígua.

        Args:
            sequence_id: ID da sequência existente.
            additional_tokens: Novos tokens a acomodar.

        Returns:
            True se extensão bem-sucedida.
        """
        if sequence_id not in self.block_tables:
            return False

        current_blocks = len(self.block_tables[sequence_id])
        current_capacity = current_blocks * self.block_size

        # Calcula quantos tokens já estão usando blocos alocados
        # (seria necessário tracking preciso; simplificamos)
        new_total_needed = current_blocks + int(
            np.ceil(additional_tokens / self.block_size)
        )
        extra_needed = new_total_needed - current_blocks

        if extra_needed <= 0:
            return True

        free = [b for b in self.blocks if b.is_free]
        if len(free) < extra_needed:
            return False

        for i in range(extra_needed):
            block = free[i]
            block.is_free = False
            block.ref_count = 1
            self.block_tables[sequence_id].append(block.block_id)

        return True

    def free(self, sequence_id: int) -> int:
        """Libera blocos de uma sequência completada.

        Args:
            sequence_id: ID da sequência a liberar.

        Returns:
            Número de blocos liberados.
        """
        if sequence_id not in self.block_tables:
            return 0

        block_ids = self.block_tables.pop(sequence_id)
        freed = 0
        for bid in block_ids:
            block = self.blocks[bid]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.is_free = True
                block.ref_count = 0
                freed += 1
        return freed

    def share_prefix(
        self, source_seq_id: int, target_seq_id: int, num_prefix_blocks: int
    ) -> bool:
        """Compartilha blocos de prefixo entre sequências.

        Implementa prefix sharing conforme descrito na seção PagedAttention:
        'Requisições que compartilham o mesmo prefixo (como um system prompt)
        podem referenciar os mesmos blocos físicos de KV-Cache'.

        Args:
            source_seq_id: Sequência de origem com o prefixo.
            target_seq_id: Sequência destino que compartilha o prefixo.
            num_prefix_blocks: Número de blocos do prefixo a compartilhar.

        Returns:
            True se compartilhamento bem-sucedido.
        """
        if source_seq_id not in self.block_tables:
            return False

        source_blocks = self.block_tables[source_seq_id]
        if num_prefix_blocks > len(source_blocks):
            return False

        shared_ids = source_blocks[:num_prefix_blocks]
        for bid in shared_ids:
            self.blocks[bid].ref_count += 1

        if target_seq_id not in self.block_tables:
            self.block_tables[target_seq_id] = []
        self.block_tables[target_seq_id] = shared_ids.copy() + self.block_tables.get(
            target_seq_id, []
        )
        return True

    def get_memory_stats(self) -> dict:
        """Retorna estatísticas de memória.

        Returns:
            Dicionário com métricas de utilização.
        """
        used_blocks = sum(1 for b in self.blocks if not b.is_free)
        shared_blocks = sum(1 for b in self.blocks if b.ref_count > 1)
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.num_blocks - used_blocks,
            "shared_blocks": shared_blocks,
            "utilization": used_blocks / self.num_blocks,
            "block_size": self.block_size,
            "total_token_capacity": self.num_blocks * self.block_size,
        }


# =========================================================================
# 4. Continuous Batching Scheduler
# =========================================================================

@dataclass
class ScheduledRequest:
    """Requisição no scheduler de continuous batching.

    Attributes:
        request_id: Identificador da requisição.
        prompt_tokens: Tokens de prompt.
        max_gen_tokens: Máximo de tokens a gerar.
        tokens_generated: Tokens já gerados.
        arrival_step: Step em que a requisição chegou.
        start_step: Step em que começou a ser processada.
        finish_step: Step em que completou (-1 se em andamento).
    """

    request_id: int
    prompt_tokens: int
    max_gen_tokens: int
    tokens_generated: int = 0
    arrival_step: int = 0
    start_step: int = -1
    finish_step: int = -1

    @property
    def is_complete(self) -> bool:
        return self.tokens_generated >= self.max_gen_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.tokens_generated


class ContinuousBatchingScheduler:
    """
    Scheduler de continuous batching (iteration-level scheduling).

    Implementa o conceito de continuous batching conforme Yu et al. (2022)
    — Orca — descrito na seção 'Continuous Batching e Scheduling Dinâmico'
    do material da aula. A cada iteração (geração de 1 token), requisições
    completas são removidas e novas são admitidas da fila.

    Diferente de static batching, onde todas requisições devem aguardar
    a mais longa, continuous batching mantém o batch maximamente preenchido.

    Referência:
        Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for
        Transformer-Based Generative Models. OSDI 2022.

    Attributes:
        max_batch_size: Tamanho máximo do batch ativo.
        max_total_tokens: Limite de tokens totais em memória.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_total_tokens: int = 4096,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens

        self.waiting_queue: list[ScheduledRequest] = []
        self.active_batch: list[ScheduledRequest] = []
        self.completed: list[ScheduledRequest] = []
        self.current_step: int = 0

        # Métricas
        self._step_history: list[dict] = []

    def add_request(self, request: ScheduledRequest) -> None:
        """Adiciona requisição à fila de espera.

        Args:
            request: Requisição a enfileirar.
        """
        request.arrival_step = self.current_step
        self.waiting_queue.append(request)

    def _current_token_count(self) -> int:
        """Total de tokens no batch ativo."""
        return sum(r.total_tokens for r in self.active_batch)

    def _can_admit(self, request: ScheduledRequest) -> bool:
        """Verifica se requisição pode ser admitida no batch."""
        if len(self.active_batch) >= self.max_batch_size:
            return False
        new_total = self._current_token_count() + request.prompt_tokens
        return new_total <= self.max_total_tokens

    def step(self) -> dict:
        """Executa uma iteração de scheduling.

        Implementa o loop central do continuous batching:
        1. Gera 1 token para cada requisição ativa
        2. Remove requisições completadas
        3. Admite novas requisições da fila

        Returns:
            Métricas da iteração (batch_size, completions, admissions, etc.).
        """
        # 1. Gera um token para cada requisição ativa
        for req in self.active_batch:
            req.tokens_generated += 1

        # 2. Remove completadas
        completed_this_step = [r for r in self.active_batch if r.is_complete]
        for req in completed_this_step:
            req.finish_step = self.current_step
            self.completed.append(req)
        self.active_batch = [r for r in self.active_batch if not r.is_complete]

        # 3. Admite novas requisições da fila (FCFS)
        admitted = 0
        while self.waiting_queue and self._can_admit(self.waiting_queue[0]):
            req = self.waiting_queue.pop(0)
            req.start_step = self.current_step
            self.active_batch.append(req)
            admitted += 1

        metrics = {
            "step": self.current_step,
            "active_batch_size": len(self.active_batch),
            "waiting_queue_size": len(self.waiting_queue),
            "completed_this_step": len(completed_this_step),
            "admitted_this_step": admitted,
            "total_tokens_in_batch": self._current_token_count(),
            "total_completed": len(self.completed),
        }
        self._step_history.append(metrics)
        self.current_step += 1
        return metrics

    def run_to_completion(self, max_steps: int = 10000) -> list[dict]:
        """Executa scheduler até todas requisições completarem.

        Args:
            max_steps: Limite de segurança de iterações.

        Returns:
            Histórico de métricas por step.
        """
        for _ in range(max_steps):
            self.step()
            if not self.active_batch and not self.waiting_queue:
                break
        return self._step_history

    def get_summary(self) -> dict:
        """Retorna resumo de performance do scheduler.

        Returns:
            Dicionário com métricas agregadas.
        """
        if not self.completed:
            return {"total_completed": 0}

        latencies = []
        for req in self.completed:
            wait_time = req.start_step - req.arrival_step
            processing_time = req.finish_step - req.start_step
            latencies.append(wait_time + processing_time)

        latencies_arr = np.array(latencies)
        return {
            "total_completed": len(self.completed),
            "total_steps": self.current_step,
            "avg_latency_steps": float(np.mean(latencies_arr)),
            "p50_latency_steps": float(np.percentile(latencies_arr, 50)),
            "p95_latency_steps": float(np.percentile(latencies_arr, 95)),
            "p99_latency_steps": float(np.percentile(latencies_arr, 99)),
            "throughput_req_per_step": len(self.completed) / max(self.current_step, 1),
            "avg_batch_utilization": float(
                np.mean([h["active_batch_size"] for h in self._step_history])
            )
            / self.max_batch_size,
        }


# =========================================================================
# 5. Static Batching Scheduler (para comparação)
# =========================================================================

class StaticBatchingScheduler:
    """
    Scheduler de static batching (baseline para comparação).

    Implementa o batching tradicional descrito como inferior na seção
    'Continuous Batching e Scheduling Dinâmico': agrupa requisições e
    processa simultaneamente até que TODAS completem. Requisições curtas
    aguardam as longas, desperdiçando capacidade.

    Attributes:
        max_batch_size: Tamanho máximo do batch.
    """

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size
        self.completed: list[ScheduledRequest] = []
        self.total_steps: int = 0

    def process_requests(
        self, requests: list[ScheduledRequest]
    ) -> list[dict]:
        """Processa requisições com static batching.

        Args:
            requests: Lista de requisições.

        Returns:
            Histórico de métricas.
        """
        history: list[dict] = []
        remaining = list(requests)
        step = 0

        while remaining:
            batch = remaining[: self.max_batch_size]
            remaining = remaining[self.max_batch_size:]

            # Processa até a requisição mais longa completar
            max_gen = max(r.max_gen_tokens for r in batch)
            for r in batch:
                r.start_step = step
                r.tokens_generated = r.max_gen_tokens
                r.finish_step = step + max_gen

            step += max_gen

            for r in batch:
                self.completed.append(r)

            history.append(
                {
                    "batch_size": len(batch),
                    "max_gen_in_batch": max_gen,
                    "wasted_tokens": sum(max_gen - r.max_gen_tokens for r in batch),
                    "step_after": step,
                }
            )

        self.total_steps = step
        return history


# =========================================================================
# 6. Speculative Decoding Simulator
# =========================================================================

class SpeculativeDecoder:
    """
    Simulador de Speculative Decoding.

    Implementa o algoritmo de speculative decoding conforme Leviathan et al.
    (2023), discutido na seção 'Speculative Decoding para Redução de Latência'
    do material da aula.

    Um modelo draft (menor e mais rápido) gera K tokens candidatos, que são
    verificados em paralelo pelo modelo principal. O speedup depende da
    acceptance rate α do draft.

    Referência:
        Leviathan, Y. et al. (2023). Fast Inference from Transformers via
        Speculative Decoding. ICML 2023.
        Chen, C. et al. (2023). Accelerating Large Language Model Decoding
        with Speculative Sampling.

    Attributes:
        draft_acceptance_rate: Probabilidade de aceitar um token do draft.
        draft_speed_factor: Razão de velocidade draft/principal (< 1 = mais rápido).
        speculation_length: Número de tokens K gerados pelo draft por rodada.
    """

    def __init__(
        self,
        draft_acceptance_rate: float = 0.8,
        draft_speed_factor: float = 0.15,
        speculation_length: int = 5,
        main_model_latency_ms: float = 50.0,
        seed: int = 42,
    ) -> None:
        if not 0 < draft_acceptance_rate <= 1.0:
            raise ValueError("draft_acceptance_rate must be in (0, 1]")
        self.draft_acceptance_rate = draft_acceptance_rate
        self.draft_speed_factor = draft_speed_factor
        self.speculation_length = speculation_length
        self.main_model_latency_ms = main_model_latency_ms
        self.rng = np.random.default_rng(seed)

    def simulate_generation(
        self, total_tokens: int
    ) -> dict:
        """Simula geração de tokens com speculative decoding.

        A cada rodada:
        1. Draft gera K tokens candidatos
        2. Main model verifica todos K em paralelo
        3. Tokens aceitos/rejeitados segundo acceptance rate

        Args:
            total_tokens: Número total de tokens a gerar.

        Returns:
            Métricas de geração: rodadas, tokens aceitos, latência estimada.
        """
        tokens_generated = 0
        rounds = 0
        total_accepted = 0
        total_draft_calls = 0
        total_verify_calls = 0

        while tokens_generated < total_tokens:
            rounds += 1
            k = min(self.speculation_length, total_tokens - tokens_generated)

            # Draft gera K candidatos
            total_draft_calls += k

            # Verificação: cada token aceito com probabilidade α
            accepted = 0
            for i in range(k):
                if self.rng.random() < self.draft_acceptance_rate:
                    accepted += 1
                else:
                    # Rejeição: gera 1 token com modelo principal
                    accepted += 1  # Token corrigido pelo modelo principal
                    break

            total_accepted += accepted
            total_verify_calls += 1  # 1 chamada de verificação paralela
            tokens_generated += accepted

        # Latência estimada
        draft_latency = self.main_model_latency_ms * self.draft_speed_factor
        autoregressive_latency = total_tokens * self.main_model_latency_ms
        speculative_latency = rounds * (
            self.speculation_length * draft_latency + self.main_model_latency_ms
        )

        return {
            "total_tokens": total_tokens,
            "rounds": rounds,
            "tokens_per_round": total_tokens / max(rounds, 1),
            "total_draft_calls": total_draft_calls,
            "total_verify_calls": total_verify_calls,
            "autoregressive_latency_ms": autoregressive_latency,
            "speculative_latency_ms": speculative_latency,
            "speedup": autoregressive_latency / max(speculative_latency, 1e-6),
            "effective_acceptance_rate": total_accepted / max(total_draft_calls, 1),
        }

    def compare_with_autoregressive(
        self,
        token_counts: list[int],
    ) -> list[dict]:
        """Compara speculative decoding vs. geração autoregressiva padrão.

        Args:
            token_counts: Lista de quantidades de tokens a gerar.

        Returns:
            Lista de dicionários com métricas comparativas.
        """
        results = []
        for n_tokens in token_counts:
            result = self.simulate_generation(n_tokens)
            results.append(result)
        return results


# =========================================================================
# 7. Amdahl & Gustafson Utilities
# =========================================================================

def amdahl_speedup(serial_fraction: float, num_processors: int) -> float:
    """Calcula speedup segundo Lei de Amdahl.

    Conforme discutido na seção 'Fundamentos de Paralelismo em Deep Learning':
    se 10% do código é sequencial, speedup máximo é 10×.

    Referência:
        Discussão em 'Saiba Mais', seção 1 do material da aula.

    Args:
        serial_fraction: Fração do trabalho que é sequencial (0–1).
        num_processors: Número de processadores paralelos.

    Returns:
        Speedup teórico.
    """
    return 1.0 / (serial_fraction + (1 - serial_fraction) / num_processors)


def gustafson_speedup(serial_fraction: float, num_processors: int) -> float:
    """Calcula speedup segundo Lei de Gustafson.

    Conforme Gustafson (1988), discutido na seção 'Fundamentos de Paralelismo':
    em workloads escaláveis, ganhos podem ser maiores que o limite de Amdahl.

    Referência:
        Gustafson, J. L. (1988). Reevaluating Amdahl's Law.
        Communications of the ACM, 31(5), 532-533.

    Args:
        serial_fraction: Fração serial do trabalho.
        num_processors: Número de processadores.

    Returns:
        Speedup segundo Gustafson.
    """
    return num_processors - serial_fraction * (num_processors - 1)
