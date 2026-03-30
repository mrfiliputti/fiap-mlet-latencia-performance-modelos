"""
Pré-processamento de dados para experimentos de NLP e Áudio.

Implementa as técnicas de tokenização, padding/packing de sequências e
extração de features de áudio (STFT, Mel Spectrogram, MFCC) discutidas
na seção 'Saiba Mais' do material da Aula 02.

Referências:
    Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
    Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale
        Weak Supervision. ICML.
    Ott, M. et al. (2019). fairseq: A Fast, Extensible Toolkit for
        Sequence Modeling. NAACL-HLT.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ #
#  Classe principal: DataPreprocessor                                 #
# ------------------------------------------------------------------ #
class DataPreprocessor:
    """Gerencia geração, carga e pré-processamento dos dados da aula.

    Conecta-se à seção *Processamento Digital de Sinais para Áudio* e
    *Estratégias de Batching para Sequências Variáveis* do documento 04.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ---- Geração de dataset sintético -------------------------------- #

    def generate_dataset(
        self,
        n_samples: int = 5000,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Gera dataset sintético com metadados de sequência e áudio.

        O dataset é projetado para exercitar os conceitos de complexidade
        quadrática do self-attention (Vaswani et al., 2017) e extração
        de features de áudio (Radford et al., 2023).

        Args:
            n_samples: Número de amostras a gerar.
            output_path: Caminho para salvar CSV. Se None, salva em
                ``data/raw/dataset.csv``.

        Returns:
            DataFrame com o dataset gerado.
        """
        rng = self.rng

        seq_lengths = rng.choice([32, 64, 128, 256, 512, 1024], size=n_samples)
        d_models = rng.choice([64, 128, 256], size=n_samples)
        num_heads = rng.choice([1, 2, 4, 8], size=n_samples)

        signal_freq = rng.uniform(80.0, 4000.0, size=n_samples).round(1)
        signal_dur = rng.uniform(0.5, 10.0, size=n_samples).round(2)
        sample_rates = rng.choice([8000, 16000, 22050], size=n_samples)
        n_mels = rng.choice([40, 80, 128], size=n_samples)
        noise_level = rng.uniform(0.0, 0.5, size=n_samples).round(3)

        attention_types = rng.choice(
            ["vanilla", "linear", "sparse", "local"], size=n_samples
        )

        # Latência simulada: cresce quadraticamente com seq_length para vanilla
        # e linearmente para linear, conforme discutido na tabela comparativa
        # de abordagens de atenção eficiente (Tay et al., 2022).
        base_latency = (seq_lengths.astype(float) ** 2) / 1e6 * d_models
        type_factors = np.where(
            attention_types == "vanilla",
            1.0,
            np.where(
                attention_types == "linear",
                0.15,
                np.where(attention_types == "sparse", 0.35, 0.25),
            ),
        )
        vanilla_latency = (base_latency * type_factors + rng.normal(0, 0.5, n_samples)).clip(0.1)

        # Memória: O(n²) para vanilla, O(n) para linear  (seção Flash Attention)
        memory_mb = (
            seq_lengths.astype(float) ** 2
            * num_heads
            / 1e6
            * type_factors
            + rng.normal(0, 0.3, n_samples)
        ).clip(0.01).round(3)

        categories = np.where(
            seq_lengths <= 128,
            "short_seq",
            np.where(seq_lengths <= 512, "medium_seq", "long_seq"),
        )

        df = pd.DataFrame(
            {
                "sample_id": np.arange(n_samples),
                "seq_length": seq_lengths,
                "d_model": d_models,
                "num_heads": num_heads,
                "signal_freq_hz": signal_freq,
                "signal_duration_s": signal_dur,
                "sample_rate": sample_rates,
                "n_mels": n_mels,
                "noise_level": noise_level,
                "attention_type": attention_types,
                "vanilla_latency_ms": vanilla_latency.round(3),
                "memory_mb": memory_mb,
                "category": categories,
            }
        )

        if output_path is None:
            output_path = str(BASE_DIR / "data" / "raw" / "dataset.csv")
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    # ---- Carga e limpeza --------------------------------------------- #

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Carrega dataset CSV.

        Args:
            path: Caminho do CSV. Se None, usa ``data/raw/dataset.csv``.

        Returns:
            DataFrame carregado.
        """
        if path is None:
            path = str(BASE_DIR / "data" / "raw" / "dataset.csv")
        return pd.read_csv(path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove valores nulos e garante tipos corretos.

        Args:
            df: DataFrame bruto.

        Returns:
            DataFrame limpo.
        """
        df = df.dropna().copy()
        int_cols = ["sample_id", "seq_length", "d_model", "num_heads", "sample_rate", "n_mels"]
        for col in int_cols:
            df[col] = df[col].astype(int)
        return df.reset_index(drop=True)

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separa features numéricas e alvo (latência vanilla).

        Implementa pré-processamento básico conforme a seção
        *Decomposição da Latência em Transformers* do material.

        Args:
            df: DataFrame limpo.

        Returns:
            Tupla (X, y) com features e alvo.
        """
        feature_cols = [
            "seq_length",
            "d_model",
            "num_heads",
            "signal_freq_hz",
            "signal_duration_s",
            "sample_rate",
            "n_mels",
            "noise_level",
        ]
        X = df[feature_cols].copy()
        y = df["vanilla_latency_ms"].copy()
        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Divide dados em treino, validação e teste.

        Args:
            X: Features.
            y: Alvo.
            test_size: Fração para teste.
            val_size: Fração para validação (do total).

        Returns:
            Dicionário com X_train, X_val, X_test, y_train, y_val, y_test.
        """
        from sklearn.model_selection import train_test_split

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val, random_state=self.seed
        )
        return {
            "X_train": X_train.values,
            "X_val": X_val.values,
            "X_test": X_test.values,
            "y_train": y_train.values,
            "y_val": y_val.values,
            "y_test": y_test.values,
        }


# ------------------------------------------------------------------ #
#  Funções de pré-processamento de sequências                        #
# ------------------------------------------------------------------ #

def pad_sequences(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica padding a sequências de comprimento variável.

    Implementa a estratégia de *padding tradicional* descrita na seção
    'Estratégias de Batching para Sequências Variáveis' do documento 04,
    onde tokens [PAD] são adicionados para equalizar comprimentos e uma
    attention mask indica posições válidas.

    Args:
        sequences: Lista de arrays 1-D de comprimentos variados.
        max_length: Comprimento alvo. Se None, usa o maior comprimento.
        pad_value: Valor de preenchimento.

    Returns:
        Tupla (padded, mask) com shapes (batch, max_length).
    """
    if max_length is None:
        max_length = max(len(s) for s in sequences)

    batch_size = len(sequences)
    padded = np.full((batch_size, max_length), pad_value, dtype=np.float32)
    mask = np.zeros((batch_size, max_length), dtype=np.float32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
        mask[i, :length] = 1.0

    return padded, mask


def sorted_batch_indices(
    lengths: List[int], batch_size: int
) -> List[List[int]]:
    """Retorna índices de batches ordenados por comprimento de sequência.

    Implementa *sorted batching* (Ott et al., 2019) conforme discutido
    na seção 'Estratégias de Batching para Sequências Variáveis',
    agrupando sequências de comprimentos similares para minimizar padding.

    Args:
        lengths: Comprimento de cada sequência no dataset.
        batch_size: Tamanho de cada batch.

    Returns:
        Lista de listas de índices por batch.
    """
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [
        sorted_idx[i : i + batch_size]
        for i in range(0, len(sorted_idx), batch_size)
    ]
    return batches


# ------------------------------------------------------------------ #
#  Funções de processamento de áudio                                 #
# ------------------------------------------------------------------ #

def generate_synthetic_audio(
    freq_hz: float = 440.0,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    noise_level: float = 0.0,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, int]:
    """Gera sinal de áudio sintético (senóide + ruído).

    Parâmetros de sample_rate alinham-se com padrões do Whisper
    (Radford et al., 2023): 16 kHz é a taxa padrão para ASR, conforme
    o Teorema de Nyquist-Shannon para fala humana (~8 kHz de banda).

    Args:
        freq_hz: Frequência fundamental em Hz.
        duration_s: Duração em segundos.
        sample_rate: Taxa de amostragem.
        noise_level: Amplitude do ruído gaussiano aditivo.
        seed: Seed para reprodutibilidade.

    Returns:
        Tupla (signal, sample_rate).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    if noise_level > 0:
        signal += (noise_level * rng.normal(size=len(t))).astype(np.float32)
    return signal, sample_rate


def compute_stft(
    signal: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Calcula a STFT (Short-Time Fourier Transform) do sinal.

    Implementa a equação da STFT discutida na seção
    'Processamento Digital de Sinais para Áudio' do documento 04:

        X[m, k] = Σ x[n + mH] · w[n] · exp(-j2πkn/N)

    Os parâmetros padrão (n_fft=400, hop_length=160) correspondem a
    janelas de 25 ms com stride de 10 ms para áudio a 16 kHz,
    configuração padrão do Whisper (Radford et al., 2023).

    Args:
        signal: Sinal temporal 1-D.
        n_fft: Tamanho da FFT (janela).
        hop_length: Deslocamento entre janelas.

    Returns:
        Magnitude do STFT com shape (n_fft//2 + 1, n_frames).
    """
    # Número de frames
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    if n_frames <= 0:
        raise ValueError(
            f"Sinal muito curto ({len(signal)} amostras) para n_fft={n_fft}"
        )

    # Janela de Hann
    window = np.hanning(n_fft).astype(np.float32)

    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for m in range(n_frames):
        start = m * hop_length
        frame = signal[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft_matrix[:, m] = np.abs(spectrum)

    return stft_matrix


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Calcula espectrograma Mel a partir de um sinal de áudio.

    Implementa a conversão para escala Mel descrita na seção
    'Processamento Digital de Sinais para Áudio':

        m = 2595 · log10(1 + f / 700)

    Esta escala é perceptualmente motivada: o sistema auditivo humano
    possui resolução frequencial aproximadamente logarítmica.
    Configuração padrão do Whisper: 80 filtros Mel, n_fft=400 (25 ms),
    hop_length=160 (10 ms) a 16 kHz (Radford et al., 2023).

    Args:
        signal: Sinal temporal 1-D.
        sample_rate: Taxa de amostragem em Hz.
        n_mels: Número de filtros Mel.
        n_fft: Tamanho da FFT.
        hop_length: Deslocamento entre janelas.

    Returns:
        Espectrograma Mel em dB, shape (n_mels, n_frames).
    """
    stft_mag = compute_stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Construir banco de filtros Mel (triângulares)
    mel_filterbank = _build_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    mel_spec = mel_filterbank @ (stft_mag ** 2)  # power spectrogram
    mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    return mel_db


def compute_mfcc(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Calcula MFCCs (Mel-Frequency Cepstral Coefficients).

    Aplica DCT ao log do espectrograma Mel, conforme descrito na seção
    'Processamento Digital de Sinais para Áudio'. Embora espectrogramas
    Mel brutos tenham demonstrado desempenho superior com redes neurais
    profundas (Radford et al., 2023), MFCCs permanecem úteis para
    sistemas clássicos e como features compactas.

    Args:
        signal: Sinal temporal 1-D.
        sample_rate: Taxa de amostragem.
        n_mfcc: Número de coeficientes cepstrais.
        n_mels: Número de filtros Mel.
        n_fft: Tamanho da FFT.
        hop_length: Deslocamento entre janelas.

    Returns:
        MFCCs com shape (n_mfcc, n_frames).
    """
    mel_db = compute_mel_spectrogram(
        signal, sample_rate, n_mels, n_fft, hop_length
    )
    # DCT tipo II (scipy-style)
    from scipy.fft import dct
    mfccs = dct(mel_db, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfccs


# ------------------------------------------------------------------ #
#  Utilitários internos                                              #
# ------------------------------------------------------------------ #

def _hz_to_mel(hz: float) -> float:
    """Converte Hz para escala Mel (seção 'Processamento Digital de
    Sinais para Áudio': m = 2595 · log10(1 + f/700))."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Converte escala Mel para Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    """Constrói banco de filtros Mel triangulares.

    Segue a formulação descrita na seção 'Processamento Digital de
    Sinais para Áudio' com filtros espaçados na escala Mel e
    geometria triangular.

    Args:
        sample_rate: Taxa de amostragem.
        n_fft: Tamanho da FFT.
        n_mels: Número de filtros.

    Returns:
        Matriz de filtros com shape (n_mels, n_fft//2 + 1).
    """
    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank
