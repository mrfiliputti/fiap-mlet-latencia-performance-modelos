"""
Testes unitários para src/data_preprocessing.py.

Cobre geração de dataset, padding de sequências, sorted batching,
e funções de processamento de áudio (STFT, Mel, MFCC).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    DataPreprocessor,
    compute_mel_spectrogram,
    compute_mfcc,
    compute_stft,
    generate_synthetic_audio,
    pad_sequences,
    sorted_batch_indices,
)


# ---- Fixtures -------------------------------------------------------- #

@pytest.fixture
def preprocessor():
    return DataPreprocessor(seed=42)


@pytest.fixture
def sample_signal():
    """Sinal sintético de 440 Hz, 1 segundo, 16 kHz."""
    signal, sr = generate_synthetic_audio(
        freq_hz=440, duration_s=1.0, sample_rate=16000, seed=42
    )
    return signal, sr


# ---- Tests: DataPreprocessor ----------------------------------------- #

def test_generate_dataset_shape(preprocessor):
    """generate_dataset retorna DataFrame com n_samples linhas e 13 colunas."""
    df = preprocessor.generate_dataset(n_samples=100, output_path=os.devnull)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df.shape[1] == 13


def test_generate_dataset_columns(preprocessor):
    """Verifica colunas esperadas no dataset gerado."""
    df = preprocessor.generate_dataset(n_samples=10, output_path=os.devnull)
    expected = {
        "sample_id", "seq_length", "d_model", "num_heads",
        "signal_freq_hz", "signal_duration_s", "sample_rate",
        "n_mels", "noise_level", "attention_type",
        "vanilla_latency_ms", "memory_mb", "category",
    }
    assert set(df.columns) == expected


def test_generate_dataset_saves_csv(preprocessor):
    """Verifica que o CSV é salvo em disco."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        preprocessor.generate_dataset(n_samples=50, output_path=path)
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 50
    finally:
        os.unlink(path)


def test_clean_data_removes_nan(preprocessor):
    """clean_data remove linhas com NaN."""
    df = preprocessor.generate_dataset(n_samples=50, output_path=os.devnull)
    df.loc[0, "seq_length"] = np.nan
    df.loc[3, "d_model"] = np.nan
    cleaned = preprocessor.clean_data(df)
    assert len(cleaned) == 48
    assert cleaned.isna().sum().sum() == 0


def test_prepare_features_shapes(preprocessor):
    """prepare_features retorna X e y com dimensões corretas."""
    df = preprocessor.generate_dataset(n_samples=200, output_path=os.devnull)
    X, y = preprocessor.prepare_features(df)
    assert X.shape == (200, 8)
    assert y.shape == (200,)


def test_split_data_sizes(preprocessor):
    """split_data divide corretamente em treino/val/teste."""
    df = preprocessor.generate_dataset(n_samples=1000, output_path=os.devnull)
    X, y = preprocessor.prepare_features(df)
    splits = preprocessor.split_data(X, y, test_size=0.2, val_size=0.1)
    total = (
        len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    )
    assert total == 1000
    assert len(splits["X_test"]) == 200  # 20%


# ---- Tests: Padding e Batching --------------------------------------- #

def test_pad_sequences_shape():
    """pad_sequences produz shape (batch, max_length)."""
    seqs = [np.array([1, 2, 3]), np.array([4, 5])]
    padded, mask = pad_sequences(seqs)
    assert padded.shape == (2, 3)
    assert mask.shape == (2, 3)


def test_pad_sequences_values():
    """pad_sequences preenche com zeros e mask indica posições válidas."""
    seqs = [np.array([1, 2]), np.array([3, 4, 5])]
    padded, mask = pad_sequences(seqs, pad_value=0.0)
    np.testing.assert_array_equal(padded[0], [1, 2, 0])
    np.testing.assert_array_equal(mask[0], [1, 1, 0])
    np.testing.assert_array_equal(mask[1], [1, 1, 1])


def test_sorted_batch_indices_groups():
    """sorted_batch_indices produz batches de tamanho correto."""
    lengths = [10, 5, 20, 3, 15, 8]
    batches = sorted_batch_indices(lengths, batch_size=2)
    assert len(batches) == 3
    assert all(len(b) == 2 for b in batches)


# ---- Tests: Áudio ---------------------------------------------------- #

def test_generate_synthetic_audio_shape():
    """Sinal gerado tem número correto de amostras."""
    signal, sr = generate_synthetic_audio(
        freq_hz=440, duration_s=2.0, sample_rate=16000
    )
    assert sr == 16000
    assert len(signal) == 32000


def test_compute_stft_shape(sample_signal):
    """STFT produz (n_fft//2 + 1, n_frames)."""
    signal, sr = sample_signal
    stft = compute_stft(signal, n_fft=400, hop_length=160)
    expected_freqs = 400 // 2 + 1  # 201
    expected_frames = 1 + (len(signal) - 400) // 160
    assert stft.shape == (expected_freqs, expected_frames)


def test_compute_mel_spectrogram_shape(sample_signal):
    """Mel spectrogram tem shape (n_mels, n_frames)."""
    signal, sr = sample_signal
    mel = compute_mel_spectrogram(signal, sr, n_mels=80, n_fft=400, hop_length=160)
    assert mel.shape[0] == 80
    assert mel.shape[1] > 0


def test_compute_mfcc_shape(sample_signal):
    """MFCCs tem shape (n_mfcc, n_frames)."""
    signal, sr = sample_signal
    mfccs = compute_mfcc(signal, sr, n_mfcc=13, n_mels=80) 
    assert mfccs.shape[0] == 13
    assert mfccs.shape[1] > 0


def test_compute_stft_short_signal_raises():
    """STFT lança erro se sinal for mais curto que n_fft."""
    short_signal = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="Sinal muito curto"):
        compute_stft(short_signal, n_fft=400, hop_length=160)
