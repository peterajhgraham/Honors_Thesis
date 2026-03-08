"""
Short-Time Fourier Transform (STFT) Decomposition

Decomposes a synthetic EEG signal into its constituent frequency bands
(delta, theta, alpha, beta, gamma) using the STFT, then reconstructs
each band via the inverse STFT. Also provides spectrogram computation
and visualisation utilities.

Functions
---------
reconstruct_band : Isolate and reconstruct a single frequency band via STFT.
compute_stft     : Compute the STFT of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft


def reconstruct_band(frequencies, Zxx, fmin, fmax, fs=250, window="hann",
                     nperseg=256, noverlap=128):
    """Isolate and reconstruct a single frequency band from an STFT.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis from ``scipy.signal.stft``.
    Zxx : np.ndarray
        Complex STFT matrix.
    fmin, fmax : float
        Lower and upper frequency bounds of the band in Hz.
    fs : int
        Sampling rate in Hz.
    window : str
        Window function name.
    nperseg : int
        Segment length used in the original STFT.
    noverlap : int
        Overlap used in the original STFT.

    Returns
    -------
    reconstructed_signal : np.ndarray
        Time-domain signal containing only the selected band.
    """
    band_mask = (frequencies >= fmin) & (frequencies <= fmax)
    Zxx_band = np.copy(Zxx)
    Zxx_band[~band_mask, :] = 0
    _, reconstructed_signal = istft(
        Zxx_band, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )
    return reconstructed_signal


def compute_stft(signal, sampling_rate, nperseg=256):
    """Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sampling_rate : int
        Sampling rate in Hz.
    nperseg : int
        Length of each STFT segment.

    Returns
    -------
    f : np.ndarray
        Frequency axis.
    t : np.ndarray
        Time axis.
    Zxx : np.ndarray
        Complex STFT matrix.
    """
    f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=nperseg)
    return f, t, Zxx
