"""
Discrete Wavelet Transform (DWT) Decomposition

Performs multi-level wavelet decomposition and perfect reconstruction
using the Daubechies-4 (``db4``) wavelet. This provides an alternative
time-frequency decomposition to the STFT, with better time resolution
at high frequencies and better frequency resolution at low frequencies.

Functions
---------
dwt_decompose_reconstruct : Decompose and perfectly reconstruct a signal via DWT.
"""

import pywt


def dwt_decompose_reconstruct(signal, wavelet="db4", level=4):
    """Decompose and reconstruct a signal using the Discrete Wavelet Transform.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal to decompose.
    wavelet : str
        Wavelet family name (default: Daubechies-4).
    level : int
        Number of decomposition levels.

    Returns
    -------
    reconstructed_signal : np.ndarray
        Perfectly reconstructed signal from wavelet coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal
