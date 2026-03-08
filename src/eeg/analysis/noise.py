"""
Noise Generation Utilities

Provides generators for white noise, pink (1/f) noise, and structured
EEG-like noise built from a deterministic harmonic basis that can be
circularly shifted to produce reproducible yet varied noise realisations.

Functions
---------
generate_pink_noise     : Generate 1/f^alpha (pink) noise via FFT filtering.
generate_clean_eeg_noise: Generate structured EEG-like noise with circular shifting.
add_noise_to_signal     : Add scaled noise to a clean signal.
"""

import numpy as np

# Maximum circular shift for noise-pattern variation
MAX_SHIFT = 88_000

# Fixed sampling rate used across the project
SAMPLING_RATE = 250


def generate_pink_noise(length, alpha=1.0):
    """Generate pink noise with a 1/f^alpha power spectrum.

    The signal is constructed by shaping white noise in the frequency
    domain with a ``1/f^(alpha/2)`` amplitude filter, then transforming
    back to the time domain.

    Parameters
    ----------
    length : int
        Number of samples to generate.
    alpha : float
        Spectral exponent (1.0 = classic pink noise).

    Returns
    -------
    colored_noise : np.ndarray
        Pink-noise signal normalised to unit variance.
    """
    fft_length = int(2 ** np.ceil(np.log2(length)))
    white_noise = np.random.normal(0, 1, fft_length)

    freq = np.fft.rfftfreq(fft_length)
    freq[0] = freq[1]  # avoid division by zero at DC

    f_filter = 1 / (freq ** (alpha / 2))
    white_noise_fft = np.fft.rfft(white_noise)
    colored_noise_fft = white_noise_fft * f_filter
    colored_noise = np.fft.irfft(colored_noise_fft)
    colored_noise = colored_noise[:length]
    colored_noise = colored_noise / np.std(colored_noise) * np.std(white_noise[:length])

    return colored_noise


def generate_clean_eeg_noise(length, shift=0):
    """Generate structured EEG-like noise using a harmonic basis.

    Builds a signal from fixed sinusoidal components at physiologically
    relevant frequencies (0.5 -- 30 Hz) with 1/sqrt(f) amplitude scaling.
    A circular shift of each component introduces pattern diversity
    without changing the spectral profile.

    Parameters
    ----------
    length : int
        Number of samples.
    shift : int
        Circular shift applied to each harmonic component.

    Returns
    -------
    normalized : np.ndarray
        Zero-mean, unit-variance noise signal.
    """
    base_noise = np.zeros(length)

    for i, f in enumerate([0.5, 1, 2, 4, 8, 12, 16, 20, 30]):
        phase = i * np.pi / 4
        amplitude = 1.0 / np.sqrt(f)
        t = np.arange(length) / SAMPLING_RATE
        component = amplitude * np.sin(2 * np.pi * f * t + phase)

        if shift > 0:
            shift_amount = shift % len(component)
            component = np.roll(component, shift_amount)

        base_noise += component

    # Deterministic high-frequency component
    large_noise_array = np.random.RandomState(42).normal(0, 0.2, length * 100)
    start_idx = shift % (len(large_noise_array) - length)
    white_noise = large_noise_array[start_idx : start_idx + length]

    combined = base_noise + white_noise
    normalized = (combined - np.mean(combined)) / np.std(combined)
    return normalized


def add_noise_to_signal(clean_signal, noise_level, shift=0):
    """Add scaled EEG-like noise to a clean signal.

    Parameters
    ----------
    clean_signal : np.ndarray
        Noise-free signal.
    noise_level : float
        Multiplicative scaling factor for the noise.
    shift : int
        Circular shift for noise-pattern variation.

    Returns
    -------
    noisy_signal : np.ndarray
        Signal with additive noise.
    """
    noise = generate_clean_eeg_noise(len(clean_signal), shift)
    noise = (noise - np.mean(noise)) / np.std(noise)
    return clean_signal + noise_level * noise
