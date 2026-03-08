"""
Wavelet-Based EEG Signal Generation

Generates synthetic EEG signals using Morlet-style wavelets. Each token
in a sequence is mapped to a sinusoidal burst at its characteristic
frequency, windowed by a Gaussian envelope. This creates time-localised
oscillatory events that mimic the transient nature of real EEG activity.

Functions
---------
generate_wavelet_signal : Map a token sequence to a concatenated wavelet signal.
"""

import numpy as np


def generate_wavelet_signal(sequence, frequencies, amplitudes, duration, sampling_rate):
    """Generate a wavelet-based EEG signal from a token sequence.

    Each token index selects a frequency and amplitude. The corresponding
    sinusoidal burst is windowed by a Gaussian envelope and concatenated
    to produce the full signal.

    Parameters
    ----------
    sequence : array-like of int
        Token indices selecting frequency/amplitude pairs.
    frequencies : list of float
        Centre frequencies in Hz for each token type.
    amplitudes : list of float
        Peak amplitudes for each token type.
    duration : float
        Duration of each token chunk in seconds.
    sampling_rate : int
        Samples per second.

    Returns
    -------
    signal : np.ndarray
        Concatenated wavelet signal.
    """
    signal = []
    chunk_length = int(duration * sampling_rate)

    for token in sequence:
        t_env = np.linspace(-1, 1, chunk_length)
        gaussian_envelope = np.exp(-(t_env**2))

        f = frequencies[token]
        a = amplitudes[token]
        t = np.arange(0, duration, 1 / sampling_rate)
        wavelet = a * np.sin(2 * np.pi * f * t)
        wavelet = wavelet[:chunk_length]
        signal.append(wavelet * gaussian_envelope)

    return np.concatenate(signal)
