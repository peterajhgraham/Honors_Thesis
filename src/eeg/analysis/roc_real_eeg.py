"""
ROC Analysis with Real-EEG-Layered Noise

Extends the synthetic ROC analysis by replacing simple Gaussian or pink
noise with structured noise derived from (or modelled after) real EEG
recordings. A correlation-based template matching detector is used to
generate realistic ROC curves across a wide range of noise levels.

Functions
---------
generate_clean_signal          : Create a synthetic EEG signal with preset patterns.
generate_frequency_band_signal : Generate a single-band Gaussian-windowed sinusoid.
generate_realistic_roc_data    : Full ROC experiment with correlation-based detection.
create_perfect_roc_dataset     : Convenience wrapper for the experiment.
test_noise_patterns            : Generate multiple noise realisations via circular shift.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from .noise import generate_clean_eeg_noise, add_noise_to_signal, MAX_SHIFT

# Default EEG parameters
SAMPLING_RATE = 250
DURATION = 3.333
FREQUENCIES = [2, 6, 10, 20, 40]
AMPLITUDES = [1.0, 0.8, 0.6, 0.4, 0.3]
WAVE_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
NOISE_LEVELS = [
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
    10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
]


def generate_clean_signal(sequence_type=None):
    """Generate a synthetic EEG signal with preset token patterns.

    Parameters
    ----------
    sequence_type : int or None
        Pattern selector (1, 2, or 3). Random if None.

    Returns
    -------
    signal : np.ndarray
        Clean wavelet-concatenated signal.
    """
    patterns = {
        1: [0, 0, 2, 3, 4],
        2: [1, 0, 1, 3, 4],
        3: [2, 0, 2, 3, 4],
    }

    if sequence_type is None or sequence_type not in patterns:
        sequence_type = np.random.choice([1, 2, 3])

    sequence = patterns[sequence_type]
    chunk_length = int(DURATION * SAMPLING_RATE)
    signal = []

    for token in sequence:
        t_env = np.linspace(-1, 1, chunk_length)
        gaussian_envelope = np.exp(-(t_env**2))
        f = FREQUENCIES[token]
        a = AMPLITUDES[token]
        t = np.arange(0, DURATION, 1 / SAMPLING_RATE)
        wavelet = a * np.sin(2 * np.pi * f * t)[:chunk_length]
        signal.append(wavelet * gaussian_envelope)

    return np.concatenate(signal)


def generate_frequency_band_signal(wave_index, length=833):
    """Generate a Gaussian-windowed sinusoid for a specific frequency band.

    Parameters
    ----------
    wave_index : int
        Index into ``FREQUENCIES`` / ``AMPLITUDES``.
    length : int
        Number of samples.

    Returns
    -------
    signal : np.ndarray
        Windowed sinusoid.
    """
    f = FREQUENCIES[wave_index]
    a = AMPLITUDES[wave_index]
    t = np.arange(0, length / SAMPLING_RATE, 1 / SAMPLING_RATE)
    t_norm = np.linspace(-1, 1, length)
    gaussian_envelope = np.exp(-(t_norm**2))
    return a * np.sin(2 * np.pi * f * t[:length]) * gaussian_envelope


def generate_realistic_roc_data(n_samples=100, n_thresholds=100):
    """Run a correlation-based detection experiment to produce ROC data.

    For each noise level and wave type, ``n_samples`` signal-present and
    signal-absent trials are generated using circularly shifted EEG-like
    noise. A template correlation score serves as the detector output.

    Parameters
    ----------
    n_samples : int
        Number of trials per condition.
    n_thresholds : int
        Number of interpolated ROC points.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: ``Wave``, ``Noise``, ``AUC``, ``FPR``, ``TPR``.
    """
    results = []
    shifts = np.linspace(0, MAX_SHIFT, n_samples, dtype=int)

    for noise_level in tqdm(NOISE_LEVELS, desc="Processing noise levels"):
        for wave_idx, wave in enumerate(WAVE_NAMES):
            y_true, y_scores = [], []

            for i in range(n_samples):
                shift = shifts[i]
                baseline_noise = (
                    generate_clean_eeg_noise(833, shift) * noise_level
                )

                # Signal-present trial
                clean_signal = generate_frequency_band_signal(wave_idx)
                noisy_signal = clean_signal + baseline_noise
                noisy_signal = (noisy_signal - np.mean(noisy_signal)) / np.std(
                    noisy_signal
                )
                template = generate_frequency_band_signal(wave_idx)
                template = (template - np.mean(template)) / np.std(template)
                corr = np.correlate(noisy_signal, template, mode="valid")[0]
                y_true.append(1)
                y_scores.append(corr)

                # Signal-absent trial
                noise_only = baseline_noise.copy()
                noise_only = (noise_only - np.mean(noise_only)) / np.std(noise_only)
                corr = np.correlate(noise_only, template, mode="valid")[0]
                y_true.append(0)
                y_scores.append(corr)

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Interpolate to fixed number of points
            if len(fpr) > n_thresholds:
                new_fpr = np.linspace(0, 1, n_thresholds)
                if len(np.unique(fpr)) < len(fpr):
                    unique_indices = np.unique(fpr, return_index=True)[1]
                    unique_fpr = fpr[np.sort(unique_indices)]
                    unique_tpr = tpr[np.sort(unique_indices)]
                    new_tpr = np.interp(new_fpr, unique_fpr, unique_tpr)
                else:
                    new_tpr = np.interp(new_fpr, fpr, tpr)
                fpr, tpr = new_fpr, new_tpr

            results.append(
                {
                    "Wave": wave,
                    "Noise": noise_level,
                    "AUC": roc_auc,
                    "FPR": fpr,
                    "TPR": tpr,
                }
            )

    return pd.DataFrame(results)


def create_perfect_roc_dataset():
    """Convenience wrapper that runs the full realistic ROC experiment."""
    return generate_realistic_roc_data(n_samples=100)


def test_noise_patterns(clean_signal, noise_level, num_patterns=100):
    """Generate multiple noisy versions of a signal via circular shift.

    Parameters
    ----------
    clean_signal : np.ndarray
        Clean signal to corrupt.
    noise_level : float
        Noise scaling factor.
    num_patterns : int
        Number of distinct noise realisations.

    Returns
    -------
    noisy_signals : list of np.ndarray
        Noisy signal variants.
    """
    noisy_signals = []
    shifts = np.linspace(0, MAX_SHIFT, num_patterns, dtype=int)
    for shift in shifts:
        noisy_signals.append(add_noise_to_signal(clean_signal, noise_level, shift))
    return noisy_signals
