"""
Experiment: ROC Analysis under White Noise

Evaluates EEG frequency-band detection performance across 12 noise
levels (0.1 to 500.0) using additive white Gaussian noise. Produces
ROC curves, AUC-vs-noise plots (log and linear scale), precision
curves, and a performance summary table.
"""

import numpy as np

from src.sequences.ngram import get_ngram_sequence_simple
from src.eeg.synthetic.wavelet_generation import generate_wavelet_signal
from src.eeg.analysis.roc_analysis import run_eeg_wavelet_experiments
from src.eeg.analysis.plotting import (
    plot_roc_curves,
    plot_auc_vs_noise,
    plot_precision_vs_noise,
    visualize_sample_signals,
)

# --- Parameters ---
DURATION = 3.333
SAMPLING_RATE = 250
FREQUENCIES = [2, 6, 10, 20, 40]
AMPLITUDES = [1.0, 0.8, 0.6, 0.4, 0.3]
N_TOKENS = 5
SEQLEN = 100
NOISE_LEVELS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
WAVE_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
FREQ_BOUNDARIES = [0, 4, 8, 12, 30, 100]

# --- Visualise sample signals ---
visualize_sample_signals(
    FREQUENCIES, AMPLITUDES, WAVE_NAMES, [0.5, 5.0, 50.0, 500.0],
    generate_wavelet_signal, get_ngram_sequence_simple,
    DURATION, SAMPLING_RATE,
    save_path="eeg_signal_samples_extended.png",
)

# --- Run experiments ---
results_df = run_eeg_wavelet_experiments(
    noise_levels=NOISE_LEVELS,
    frequencies=FREQUENCIES,
    amplitudes=AMPLITUDES,
    wave_names=WAVE_NAMES,
    frequency_boundaries=FREQ_BOUNDARIES,
    n_tokens=N_TOKENS,
    seqlen=SEQLEN,
    duration=DURATION,
    sampling_rate=SAMPLING_RATE,
    generate_wavelet_signal_fn=generate_wavelet_signal,
    get_ngram_sequence_fn=get_ngram_sequence_simple,
    n_repeats=3,
)

# --- Plots ---
plot_roc_curves(
    results_df, WAVE_NAMES, [0.1, 1.0, 10.0, 100.0],
    save_path="eeg_wavelet_roc_curves_extended.png",
)
plot_auc_vs_noise(
    results_df, WAVE_NAMES, NOISE_LEVELS, log_scale=True,
    save_path="eeg_wavelet_auc_vs_noise_log.png",
)
plot_auc_vs_noise(
    results_df, WAVE_NAMES, NOISE_LEVELS, log_scale=False,
    save_path="eeg_wavelet_auc_vs_noise_linear.png",
)
plot_precision_vs_noise(
    results_df, WAVE_NAMES, NOISE_LEVELS, log_scale=True,
    save_path="eeg_wavelet_precision_vs_noise_log.png",
)

# --- Summary ---
performance_summary = results_df.groupby(["Wave", "Noise"]).apply(
    lambda x: x.loc[x["Precision"].idxmax()]
)[["AUC", "Precision", "Threshold"]].reset_index()

print("\nPerformance Summary (Best Precision for Each Wave Type and Noise Level):")
print(performance_summary.to_string(index=False))

print("\nNoise levels where each wave type reaches AUC <= 0.5:")
for wave in WAVE_NAMES:
    wave_res = performance_summary[performance_summary["Wave"] == wave]
    below = wave_res[wave_res["AUC"] <= 0.5]
    if not below.empty:
        nl = below["Noise"].min()
        a = below.loc[below["Noise"] == nl, "AUC"].values[0]
        print(f"  {wave}: Noise level = {nl} (AUC = {a:.3f})")
    else:
        mx = wave_res["Noise"].max()
        a = wave_res.loc[wave_res["Noise"] == mx, "AUC"].values[0]
        print(f"  {wave}: Did not reach AUC <= 0.5 (Min AUC = {a:.3f} at noise {mx})")
