"""
Experiment: Synthetic EEG Generation, STFT Band Decomposition, and DWT

1. Generates a multi-band synthetic EEG signal.
2. Decomposes it into delta/theta/alpha/beta/gamma via STFT band-pass filtering.
3. Demonstrates DWT decomposition and perfect reconstruction.
4. Renders spectrogram and per-band time-domain plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from src.eeg.synthetic.signal_generation import (
    generate_synthetic_eeg_with_multiple_waves,
    generate_token_chunk,
)
from src.eeg.decomposition.stft_decomposition import reconstruct_band, compute_stft
from src.eeg.decomposition.dwt_decomposition import dwt_decompose_reconstruct

# ------------------------------------------------------------------ #
#  1. Generate synthetic EEG
# ------------------------------------------------------------------ #
time, synthetic_eeg = generate_synthetic_eeg_with_multiple_waves()

plt.figure(figsize=(10, 4))
plt.plot(time, synthetic_eeg)
plt.title("Synthetic EEG Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# ------------------------------------------------------------------ #
#  2. STFT band decomposition
# ------------------------------------------------------------------ #
window = "hann"
nperseg = 256
noverlap = 128

frequencies, times, Zxx = stft(
    synthetic_eeg, fs=250, window=window, nperseg=nperseg, noverlap=noverlap
)

bands = {
    "Delta (0.5-4 Hz)": (0.5, 4, "blue"),
    "Theta (4-8 Hz)": (4, 8, "green"),
    "Alpha (8-12 Hz)": (8, 12, "red"),
    "Beta (12-30 Hz)": (12, 30, "purple"),
    "Gamma (30-100 Hz)": (30, 100, "orange"),
}

reconstructed = {}
for name, (fmin, fmax, color) in bands.items():
    reconstructed[name] = reconstruct_band(
        frequencies, Zxx, fmin, fmax,
        fs=250, window=window, nperseg=nperseg, noverlap=noverlap,
    )

min_length = min(len(time), *(len(r) for r in reconstructed.values()))

plt.figure(figsize=(12, 10))
plt.subplot(6, 1, 1)
plt.plot(time[:min_length], synthetic_eeg[:min_length])
plt.title("Original EEG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

for i, (name, (fmin, fmax, color)) in enumerate(bands.items(), start=2):
    plt.subplot(6, 1, i)
    plt.plot(time[:min_length], reconstructed[name][:min_length], color=color)
    plt.title(f"Reconstructed {name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------ #
#  3. DWT decomposition and reconstruction
# ------------------------------------------------------------------ #
reconstructed_eeg = dwt_decompose_reconstruct(synthetic_eeg)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, synthetic_eeg, label="Original")
plt.title("Original EEG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(
    time[: len(reconstructed_eeg)], reconstructed_eeg,
    label="Reconstructed", color="orange",
)
plt.title("Reconstructed EEG Signal from DWT")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------ #
#  4. Spectrogram
# ------------------------------------------------------------------ #
f, t, Zxx = compute_stft(synthetic_eeg, sampling_rate=250)

plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
plt.title("STFT Magnitude Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar()
plt.show()

# ------------------------------------------------------------------ #
#  5. Token-chunked signal with noise & event detection
# ------------------------------------------------------------------ #
duration = 10
sampling_rate = 250
freq_map = {"delta": 2, "alpha": 10, "gamma": 40}
amp_map = {"delta": 2.0, "alpha": 1.5, "gamma": 0.8}

delta_chunk = generate_token_chunk(freq_map["delta"], amp_map["delta"],
                                   duration / 3, sampling_rate)
alpha_chunk = generate_token_chunk(freq_map["alpha"], amp_map["alpha"],
                                   duration / 3, sampling_rate)
gamma_chunk = generate_token_chunk(freq_map["gamma"], amp_map["gamma"],
                                   duration / 3, sampling_rate)
eeg_no_noise = np.concatenate((delta_chunk, alpha_chunk, gamma_chunk))

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(eeg_no_noise)) / sampling_rate, eeg_no_noise)
plt.title("Synthetic EEG Signal without Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# STFT without noise
frequencies, times, Zxx = stft(eeg_no_noise, fs=sampling_rate, nperseg=128)
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, np.abs(Zxx), shading="gouraud")
plt.title("STFT Magnitude — Without Noise")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.show()

# Add noise and re-compute
noise_level = 0.5
eeg_noisy = eeg_no_noise + noise_level * np.random.randn(len(eeg_no_noise))
frequencies, times, Zxx_noisy = stft(eeg_noisy, fs=sampling_rate, nperseg=128)

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, np.abs(Zxx_noisy), shading="gouraud")
plt.title("STFT Magnitude — With Noise")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.show()

# Threshold-based event detection
amplitude_threshold = 0.25
event_mask = np.abs(Zxx_noisy) > amplitude_threshold

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, np.abs(Zxx_noisy), shading="gouraud")
plt.contour(times, frequencies, event_mask, colors="red", linewidths=0.5)
plt.title("Detected Events in Noisy STFT Signal (Thresholded)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()
