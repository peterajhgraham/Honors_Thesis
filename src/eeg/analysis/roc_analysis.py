"""
ROC Analysis for EEG Frequency-Band Detection

Evaluates the detectability of each canonical EEG frequency band under
increasing levels of additive noise. For each noise level the Short-Time
Fourier Transform is computed, band-specific power is extracted, and
Receiver Operating Characteristic (ROC) curves are constructed against
ground-truth wavelet locations. The Area Under the Curve (AUC) and
precision are reported as summary statistics.

Functions
---------
calculate_roc_for_thresholds : Sweep detection thresholds and compute ROC metrics.
threshold_stft               : Binary threshold of STFT band power.
run_eeg_wavelet_experiments  : Full experimental loop over noise levels.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.signal import stft
from tqdm import tqdm


def calculate_roc_for_thresholds(signal, ground_truth, thresholds, wave_type,
                                  noise_level):
    """Compute ROC statistics over a range of detection thresholds.

    Parameters
    ----------
    signal : np.ndarray
        Continuous detection score (e.g., band power).
    ground_truth : np.ndarray
        Binary ground-truth labels (1 = signal present).
    thresholds : np.ndarray
        Detection thresholds to evaluate.
    wave_type : str
        Name of the frequency band (for bookkeeping).
    noise_level : float
        Current noise level (for bookkeeping).

    Returns
    -------
    results : list of dict
        One entry per threshold with keys ``Wave``, ``Noise``,
        ``Threshold``, ``AUC``, ``Precision``, ``FPR``, ``TPR``.
    """
    results = []

    for threshold in thresholds:
        detected = signal > threshold
        fpr, tpr, _ = roc_curve(ground_truth, signal)
        roc_auc = auc(fpr, tpr)

        true_positives = np.sum(detected & ground_truth.astype(bool))
        false_positives = np.sum(detected & ~ground_truth.astype(bool))
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        results.append(
            {
                "Wave": wave_type,
                "Noise": noise_level,
                "Threshold": threshold,
                "AUC": roc_auc,
                "Precision": precision,
                "FPR": fpr,
                "TPR": tpr,
            }
        )

    return results


def threshold_stft(Zxx_noisy, amplitude_threshold, row_indices):
    """Apply a binary threshold to mean STFT band power.

    Parameters
    ----------
    Zxx_noisy : np.ndarray
        Complex STFT matrix.
    amplitude_threshold : float
        Detection threshold.
    row_indices : np.ndarray
        Frequency-bin indices for the target band.

    Returns
    -------
    detected : np.ndarray of bool
        Boolean detection mask over STFT time bins.
    """
    return np.mean(np.abs(Zxx_noisy[row_indices, :]), axis=0) > amplitude_threshold


def run_eeg_wavelet_experiments(
    noise_levels,
    frequencies,
    amplitudes,
    wave_names,
    frequency_boundaries,
    n_tokens,
    seqlen,
    duration,
    sampling_rate,
    generate_wavelet_signal_fn,
    get_ngram_sequence_fn,
    n_repeats=5,
    generate_noise_fn=None,
):
    """Run the full experimental loop over noise levels.

    For each noise level and repeat, a random token sequence is generated,
    converted to a wavelet signal, corrupted with noise, decomposed via
    STFT, and evaluated with ROC analysis for each frequency band.

    Parameters
    ----------
    noise_levels : list of float
        Noise scaling factors to test.
    frequencies : list of float
        Centre frequencies for each token type.
    amplitudes : list of float
        Peak amplitudes for each token type.
    wave_names : list of str
        Human-readable band names.
    frequency_boundaries : list of float
        Edges of the canonical EEG bands.
    n_tokens : int
        Number of distinct token types.
    seqlen : int
        Token-sequence length.
    duration : float
        Duration per token chunk in seconds.
    sampling_rate : int
        Samples per second.
    generate_wavelet_signal_fn : callable
        Function to create wavelet signals from a token sequence.
    get_ngram_sequence_fn : callable
        Function to generate a random token sequence.
    n_repeats : int
        Independent repetitions per noise level.
    generate_noise_fn : callable or None
        Custom noise generator; defaults to ``np.random.randn``.

    Returns
    -------
    results_df : pd.DataFrame
        Per-threshold ROC results for all conditions.
    """
    all_results = []

    for noise_level in tqdm(noise_levels, desc="Processing noise levels"):
        max_threshold = max(3 * noise_level, 1.0)
        thresholds = np.linspace(0, max_threshold, 20)

        for repeat in range(n_repeats):
            sequence = get_ngram_sequence_fn(5, n_tokens, seqlen)
            morlet_signal = generate_wavelet_signal_fn(
                sequence, frequencies, amplitudes, duration, sampling_rate
            )

            if generate_noise_fn is not None:
                noise = noise_level * generate_noise_fn(len(morlet_signal))
            else:
                noise = noise_level * np.random.randn(len(morlet_signal))
            morlet_signal_noisy = morlet_signal + noise

            # Ground-truth wavelet locations
            wavelet_indices = [np.where(sequence == i)[0] for i in range(n_tokens)]
            chunk_length = int(duration * sampling_rate)
            where_frequency = np.zeros([n_tokens, chunk_length * seqlen])

            for i, indices in enumerate(wavelet_indices):
                for j in indices:
                    where_frequency[i, chunk_length * j : chunk_length * (j + 1)] = 1

            # STFT
            nperseg = min(1024, max(256, int(256 * (1 + noise_level / 10))))
            noverlap = int(nperseg * 0.75)
            f, t, Zxx_noisy = stft(
                morlet_signal_noisy, fs=sampling_rate,
                nperseg=nperseg, noverlap=noverlap,
            )

            # Align ground truth to STFT time bins
            time_step = duration * seqlen / t.shape[0]
            aligned_truth = []
            for i in range(n_tokens):
                interp_truth = np.zeros(t.shape[0])
                for j in range(t.shape[0]):
                    time_point = j * time_step
                    idx = int(time_point * sampling_rate)
                    if idx < where_frequency.shape[1]:
                        interp_truth[j] = where_frequency[i, idx]
                aligned_truth.append(interp_truth)

            # Extract frequency-band indices
            band_indices = []
            for i in range(len(frequency_boundaries) - 1):
                indices = np.where(
                    (f >= frequency_boundaries[i]) & (f < frequency_boundaries[i + 1])
                )[0]
                band_indices.append(indices)

            # Evaluate each band
            for i in range(n_tokens):
                smooth_window = min(7, max(3, int(1 + noise_level / 50)))
                band_power = np.mean(np.abs(Zxx_noisy[band_indices[i], :]), axis=0)

                if i == 4:  # Gamma
                    band_power = np.convolve(
                        band_power,
                        np.ones(smooth_window) / smooth_window,
                        mode="same",
                    )
                elif noise_level > 1.0:
                    band_power = np.convolve(
                        band_power,
                        np.ones(smooth_window) / smooth_window,
                        mode="same",
                    )

                results = calculate_roc_for_thresholds(
                    band_power, aligned_truth[i], thresholds,
                    wave_names[i], noise_level,
                )
                all_results.extend(results)

    return pd.DataFrame(all_results)
