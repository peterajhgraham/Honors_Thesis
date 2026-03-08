"""
Synthetic EEG Signal Generation

Generates synthetic EEG-like signals by superimposing sinusoidal
components at the five canonical frequency bands (delta, theta, alpha,
beta, gamma) with additive Gaussian noise.

Functions
---------
generate_synthetic_eeg_with_multiple_waves : Create a multi-band EEG signal.
generate_token_chunk                       : Generate a single sinusoidal chunk.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_eeg_with_multiple_waves(
    duration=10, sampling_rate=250, noise_level=0.5
):
    """Generate a synthetic multi-band EEG signal.

    Creates sinusoidal components at representative frequencies for
    delta (2.25 Hz), theta (6 Hz), alpha (10 Hz), beta (21 Hz), and
    gamma (65 Hz) bands. Gaussian noise is added to simulate real
    recording conditions.

    Parameters
    ----------
    duration : float
        Signal length in seconds.
    sampling_rate : int
        Samples per second.
    noise_level : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    time : np.ndarray
        Time axis in seconds.
    eeg_signal : np.ndarray
        Composite noisy EEG signal.
    """
    time = np.arange(0, duration, 1 / sampling_rate)

    # Generate different EEG frequency components (choose average of range)
    delta_wave = np.sin(2 * np.pi * 2.25 * time)   # Delta  (0.5-4 Hz)
    theta_wave = np.sin(2 * np.pi * 6 * time)       # Theta  (4-8 Hz)
    alpha_wave = np.sin(2 * np.pi * 10 * time)      # Alpha  (8-12 Hz)
    beta_wave = np.sin(2 * np.pi * 21 * time)       # Beta   (12-30 Hz)
    gamma_wave = np.sin(2 * np.pi * 65 * time)      # Gamma  (30-100 Hz)

    eeg_signal = delta_wave + theta_wave + alpha_wave + beta_wave + gamma_wave

    # np.random.randn produces a standard normal distribution (mean=0, std=1)
    noise = noise_level * np.random.randn(len(time))

    return time, eeg_signal + noise


def generate_token_chunk(frequency, amplitude, duration, sampling_rate):
    """Generate a single sinusoidal chunk representing one token.

    Parameters
    ----------
    frequency : float
        Frequency of the sinusoid in Hz.
    amplitude : float
        Peak amplitude.
    duration : float
        Chunk duration in seconds.
    sampling_rate : int
        Samples per second.

    Returns
    -------
    chunk : np.ndarray
        Sinusoidal signal.
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t)
