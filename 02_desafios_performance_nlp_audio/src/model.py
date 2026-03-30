"""
Modelos de atenção para demonstração de desafios de performance.

Implementa os mecanismos de self-attention discutidos na seção
'Fundamentos do Mecanismo de Self-Attention' e variantes eficientes
da taxonomia proposta por Tay et al. (2022) em 'Efficient Transformers:
A Survey'.

Referências:
    Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
    Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient
        Exact Attention with IO-Awareness. NeurIPS.
    Katharopoulos, A. et al. (2020). Transformers are RNNs: Fast
        Autoregressive Transformers with Linear Attention. ICML.
    Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is
        All You Need. arXiv:1911.02150.
    Tay, Y. et al. (2022). Efficient Transformers: A Survey.
        ACM Computing Surveys, v. 55, n. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ================================================================== #
#  1. Vanilla Scaled Dot-Product Attention                           #
# ================================================================== #

class VanillaAttention:
    """Self-attention com complexidade O(n²·d) / memória O(n²).

    Implementa a formulação clássica de Vaswani et al. (2017):

        Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    conforme apresentado na seção 'Fundamentos do Mecanismo de
    Self-Attention' do material da aula.

    A complexidade quadrática decorre da materialização completa da
    matriz de scores Q K^T ∈ ℝ^{n×n}.
    """

    def __init__(self, d_model: int, num_heads: int = 1) -> None:
        """
        Args:
            d_model: Dimensão do modelo (projeções Q, K, V).
            num_heads: Número de cabeças de atenção.
        """
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisível por num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projeções lineares (pesos aleatórios para demos)
        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax numericamente estável (subtrai max antes de exp)."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula atenção scaled dot-product.

        Implementa:  softmax(Q K^T / √d_k) V

        O fator de escala √d_k é crucial para estabilidade numérica,
        evitando que produtos escalares grandes empurrem a softmax para
        regiões de gradientes extremamente pequenos (Vaswani et al., 2017).

        Args:
            Q: Queries,  shape (..., n, d_k).
            K: Keys,     shape (..., n, d_k).
            V: Values,   shape (..., n, d_v).
            mask: Máscara booleana, shape (..., n, n). True = posição mascarada.

        Returns:
            Tupla (output, attention_weights).
        """
        d_k = Q.shape[-1]
        # Q K^T → shape (..., n, n)  — complexidade O(n²·d)
        scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax — requer materializar n×n na memória → O(n²)
        attn_weights = self._softmax(scores, axis=-1)

        # Multiplicar por V → (..., n, d_v)
        output = attn_weights @ V
        return output, attn_weights

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass multi-head.

        Args:
            X: Entrada, shape (batch, seq_len, d_model).
            mask: Máscara opcional.

        Returns:
            Tupla (output, attention_weights).
        """
        batch, seq_len, _ = X.shape

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Reshape para multi-head: (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenar cabeças
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        output = output @ self.W_o

        return output, attn_weights

    def memory_footprint_bytes(self, seq_length: int, batch_size: int = 1) -> int:
        """Estima memória para a matriz de atenção n×n.

        Conforme discutido na seção 'Fundamentos do Mecanismo de
        Self-Attention': a matriz QK^T é n×n, com O(n²) de memória.
        """
        # attention scores: batch × heads × n × n, cada elem = 4 bytes (float32)
        return batch_size * self.num_heads * seq_length * seq_length * 4


# ================================================================== #
#  2. Linear Attention (Kernel-Based)                                #
# ================================================================== #

class LinearAttention:
    """Atenção linear com complexidade O(n·d²) usando kernel trick.

    Implementa a formulação de Katharopoulos et al. (2020) discutida
    na seção 'Kernel-Based Methods' da taxonomia de atenção eficiente:

        Attention(Q, K, V) = φ(Q) (φ(K)^T V) / (φ(Q) Σ φ(K_j)^T)

    onde φ é uma feature map (aqui usamos ELU + 1 como aproximação).
    A associatividade permite computar φ(K)^T V primeiro em O(n·d²),
    seguido de multiplicação por φ(Q).
    """

    def __init__(self, d_model: int, num_heads: int = 1) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisível por num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)

    @staticmethod
    def _feature_map(x: np.ndarray) -> np.ndarray:
        """Feature map φ(x) = ELU(x) + 1 (sempre positiva)."""
        return np.where(x > 0, x + 1, np.exp(x))

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, None]:
        """Forward pass com atenção linear.

        Complexidade: O(n·d²) em vez de O(n²·d), vantajoso quando n >> d.

        Args:
            X: Entrada, shape (batch, seq_len, d_model).
            mask: Não utilizado (mantido para interface uniforme).

        Returns:
            Tupla (output, None) — pesos de atenção não são materializados.
        """
        batch, seq_len, _ = X.shape

        Q = self._feature_map(X @ self.W_q)
        K = self._feature_map(X @ self.W_k)
        V = X @ self.W_v

        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Associatividade: φ(K)^T V → (batch, heads, d_k, d_k)
        KV = K.swapaxes(-2, -1) @ V

        # φ(Q) · (φ(K)^T V) → (batch, heads, n, d_k)
        numerator = Q @ KV

        # Normalização: φ(Q) Σ φ(K_j)^T → (batch, heads, n, 1)
        K_sum = K.sum(axis=-2, keepdims=True)       # (batch, heads, 1, d_k)
        denominator = (Q * K_sum).sum(axis=-1, keepdims=True)  # dot per query
        denominator = np.maximum(denominator, 1e-6)

        output = numerator / denominator

        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        output = output @ self.W_o

        return output, None

    def memory_footprint_bytes(self, seq_length: int, batch_size: int = 1) -> int:
        """Atenção linear não materializa n×n → memória O(n·d)."""
        return batch_size * self.num_heads * seq_length * self.d_k * 4


# ================================================================== #
#  3. Local (Windowed) Attention                                     #
# ================================================================== #

class LocalAttention:
    """Atenção local com janela deslizante.

    Implementa o padrão *Fixed Patterns — Local Attention* da taxonomia
    de Tay et al. (2022): cada token atende apenas a uma janela de w
    tokens vizinhos, reduzindo complexidade para O(n·w·d).

    Similar à abordagem do Longformer (Beltagy et al., 2020).
    """

    def __init__(
        self, d_model: int, window_size: int = 64, num_heads: int = 1
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisível por num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, None]:
        """Forward pass com atenção local por janela deslizante.

        Complexidade: O(n·w·d) onde w = window_size << n.

        Args:
            X: Entrada (batch, seq_len, d_model).
            mask: Não utilizado.

        Returns:
            Tupla (output, None).
        """
        batch, seq_len, _ = X.shape
        w = self.window_size

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Processamento simplificado (single-head para clareza)
        output = np.zeros_like(Q)
        for i in range(seq_len):
            start = max(0, i - w // 2)
            end = min(seq_len, i + w // 2 + 1)

            q_i = Q[:, i : i + 1, :]        # (batch, 1, d)
            k_w = K[:, start:end, :]         # (batch, w', d)
            v_w = V[:, start:end, :]         # (batch, w', d)

            scores = (q_i @ k_w.swapaxes(-2, -1)) / math.sqrt(self.d_model)
            weights = self._softmax(scores, axis=-1)
            output[:, i, :] = (weights @ v_w).squeeze(1)

        output = output @ self.W_o
        return output, None

    def memory_footprint_bytes(self, seq_length: int, batch_size: int = 1) -> int:
        """Memória O(n·w) em vez de O(n²)."""
        w = min(self.window_size, seq_length)
        return batch_size * self.num_heads * seq_length * w * 4


# ================================================================== #
#  4. Tiled Attention (Flash Attention simplificado)                 #
# ================================================================== #

class TiledAttention:
    """Atenção com tiling — demonstração pedagógica do princípio Flash Attention.

    Implementa o conceito de *IO-aware tiling* proposto por Dao et al.
    (2022) na seção 'Flash Attention: IO-Awareness como Princípio de
    Design'. O algoritmo processa Q, K, V em blocos que caberiam na
    SRAM, computando softmax de forma *online* com estatísticas
    correntes (máximo e soma) para renormalização incremental.

    NOTA: Esta é uma implementação pedagógica em NumPy para demonstrar
    o algoritmo. Flash Attention real opera em CUDA com acesso direto
    à hierarquia de memória da GPU (SRAM vs HBM).

    Referência:
        Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient
        Exact Attention with IO-Awareness. NeurIPS.
    """

    def __init__(self, d_model: int, block_size: int = 64) -> None:
        self.d_model = d_model
        self.block_size = block_size

        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)

    def forward(
        self, X: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, None]:
        """Forward pass com tiling e softmax online.

        Processa blocos de tamanho B, mantendo estatísticas
        correntes (m_i, l_i) para normalização incremental da softmax,
        sem nunca materializar a matriz n×n completa.

        Args:
            X: Entrada (batch, seq_len, d_model).
            mask: Não utilizado nesta versão simplificada.

        Returns:
            Tupla (output, None).
        """
        batch, n, d = X.shape
        B = self.block_size
        scale = 1.0 / math.sqrt(d)

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        O = np.zeros((batch, n, d), dtype=np.float64)
        # Estatísticas para softmax online (Dao et al., 2022)
        m = np.full((batch, n, 1), -np.inf)  # max corrente
        l = np.zeros((batch, n, 1))           # soma corrente

        n_blocks = math.ceil(n / B)
        for j in range(n_blocks):
            j_start = j * B
            j_end = min((j + 1) * B, n)

            K_j = K[:, j_start:j_end, :]  # (batch, B_j, d)
            V_j = V[:, j_start:j_end, :]

            for i in range(n_blocks):
                i_start = i * B
                i_end = min((i + 1) * B, n)

                Q_i = Q[:, i_start:i_end, :]  # (batch, B_i, d)

                # Scores do bloco: (batch, B_i, B_j)
                S_ij = (Q_i @ K_j.swapaxes(-2, -1)) * scale

                # Softmax online: atualizar max e soma
                m_ij = S_ij.max(axis=-1, keepdims=True)  # (batch, B_i, 1)
                m_old = m[:, i_start:i_end, :]
                m_new = np.maximum(m_old, m_ij)

                # Renormalizar soma anterior
                exp_old = np.exp(m_old - m_new)
                exp_new = np.exp(S_ij - m_new)

                l_old = l[:, i_start:i_end, :]
                l_new = exp_old * l_old + exp_new.sum(axis=-1, keepdims=True)

                # Atualizar output com renormalização
                O[:, i_start:i_end, :] = (
                    (exp_old * l_old) * O[:, i_start:i_end, :]
                    + exp_new @ V_j
                ) / np.maximum(l_new, 1e-10)

                m[:, i_start:i_end, :] = m_new
                l[:, i_start:i_end, :] = l_new

        return O.astype(np.float32), None

    def memory_footprint_bytes(self, seq_length: int, batch_size: int = 1) -> int:
        """Flash Attention: memória O(n) em vez de O(n²).

        Apenas blocos de tamanho B são mantidos em memória por vez,
        mais estatísticas (m, l) de tamanho O(n).
        """
        B = self.block_size
        # Blocos Q_i, K_j, V_j + estatísticas m, l
        block_mem = batch_size * B * self.d_model * 3 * 4
        stats_mem = batch_size * seq_length * 2 * 4
        return block_mem + stats_mem


# ================================================================== #
#  5. KV-Cache para geração autoregressiva                          #
# ================================================================== #

@dataclass
class KVCache:
    """Cache de Keys e Values para geração autoregressiva eficiente.

    Implementa o mecanismo de KV-Cache descrito na seção 'KV-Cache e
    Geração Autoregressiva' do material da aula:

        K_cached = [K_1; K_2; ...; K_t]
        V_cached = [V_1; V_2; ...; V_t]

    Para gerar posição t+1, apenas Q_{t+1} é computado; K e V anteriores
    são recuperados do cache. Isso reduz o custo marginal de cada token
    de O(t²·d) para O(t·d) (Shazeer, 2019; Touvron et al., 2023).

    Referência:
        Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head
            is All You Need. arXiv:1911.02150.
        Touvron, H. et al. (2023). LLaMA 2: Open Foundation and
            Fine-Tuned Chat Models. arXiv:2307.09288.
    """

    d_model: int
    num_layers: int = 1
    num_heads: int = 1
    # Listas de cache por camada: K e V acumulados
    keys: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Inicializar listas vazias por camada
        if not self.keys:
            self.keys = [None] * self.num_layers  # type: ignore[list-item]
            self.values = [None] * self.num_layers  # type: ignore[list-item]

    def update(
        self, layer_idx: int, new_key: np.ndarray, new_value: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adiciona novo K, V ao cache e retorna cache acumulado.

        K_cached ← [K_cached; K_{t+1}]
        V_cached ← [V_cached; V_{t+1}]

        Args:
            layer_idx: Índice da camada.
            new_key: Novo key, shape (batch, 1, d_model).
            new_value: Novo value, shape (batch, 1, d_model).

        Returns:
            Tupla (cached_keys, cached_values) acumulados.
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_key
            self.values[layer_idx] = new_value
        else:
            self.keys[layer_idx] = np.concatenate(
                [self.keys[layer_idx], new_key], axis=1
            )
            self.values[layer_idx] = np.concatenate(
                [self.values[layer_idx], new_value], axis=1
            )
        return self.keys[layer_idx], self.values[layer_idx]

    def current_length(self, layer_idx: int = 0) -> int:
        """Retorna número de tokens no cache."""
        if self.keys[layer_idx] is None:
            return 0
        return self.keys[layer_idx].shape[1]

    def memory_bytes(self) -> int:
        """Calcula memória total do cache.

        Conforme discutido no material: custo O(n·L·H·d), onde
        L = camadas, H = cabeças, d = dimensão por cabeça.
        """
        total = 0
        for layer_idx in range(self.num_layers):
            if self.keys[layer_idx] is not None:
                total += self.keys[layer_idx].nbytes
                total += self.values[layer_idx].nbytes
        return total

    def clear(self) -> None:
        """Limpa o cache (nova sequência)."""
        self.keys = [None] * self.num_layers  # type: ignore[list-item]
        self.values = [None] * self.num_layers  # type: ignore[list-item]


# ================================================================== #
#  6. Transformer Block simplificado                                 #
# ================================================================== #

class SimpleTransformerBlock:
    """Bloco Transformer com self-attention + FFN.

    Implementa a decomposição de latência descrita na seção
    'Decomposição da Latência em Transformers': embedding → attention
    scores → FFN → layer norm. As camadas FFN, com dimensão 4× maior
    que d_model, respondem por ~60 % dos FLOPs totais em inferência
    (Shoeybi et al., 2019).

    Referência:
        Shoeybi, M. et al. (2019). Megatron-LM: Training Multi-Billion
            Parameter Language Models Using Model Parallelism.
    """

    def __init__(self, d_model: int, num_heads: int = 1, ff_dim: Optional[int] = None) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * d_model  # Padrão: 4× d_model

        self.attention = VanillaAttention(d_model, num_heads)

        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_ff1 = rng.normal(0, scale, (d_model, self.ff_dim)).astype(np.float32)
        self.b_ff1 = np.zeros(self.ff_dim, dtype=np.float32)
        self.W_ff2 = rng.normal(0, scale, (self.ff_dim, d_model)).astype(np.float32)
        self.b_ff2 = np.zeros(d_model, dtype=np.float32)

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization — introduz pontos de sincronização que
        limitam paralelismo, conforme observado na decomposição de
        latência."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.

        Args:
            X: Entrada (batch, seq_len, d_model).

        Returns:
            Saída (batch, seq_len, d_model).
        """
        # Sub-camada 1: Self-Attention
        normed = self._layer_norm(X)
        attn_out, _ = self.attention.forward(normed)
        X = X + attn_out  # Conexão residual

        # Sub-camada 2: Feed-Forward Network
        normed = self._layer_norm(X)
        ff_out = self._relu(normed @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        X = X + ff_out  # Conexão residual

        return X
