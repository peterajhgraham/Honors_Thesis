"""
EEG Experiment Plotting Utilities

Publication-quality plotting functions for ROC curves, AUC-vs-noise
summaries, precision curves, signal sample visualisations, and
wave-component breakdowns. A consistent colour palette maps each
canonical EEG band to a fixed colour across all figures.

Functions
---------
get_wave_colors               : Return colour list for wave names.
plot_roc_curves               : ROC subplots at selected noise levels.
plot_auc_vs_noise             : AUC score vs noise level.
plot_precision_vs_noise       : Precision vs noise level.
visualize_sample_signals      : Clean vs noisy signal grid.
create_beautiful_roc_curves   : ROC curves from real-EEG-noise experiments.
create_beautiful_signal_examples : Signal visualisations with EEG-like noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Canonical colour palette
WAVE_COLORS = {
    "Delta": "#4169E1",   # royal blue
    "Theta": "#FF8C00",   # dark orange
    "Alpha": "#228B22",   # forest green
    "Beta": "#DC143C",    # crimson
    "Gamma": "#9932CC",   # dark orchid
}

MARKERS = ["o", "s", "D", "^", "p"]


def get_wave_colors(wave_names):
    """Return a list of hex colours for the given wave names."""
    return [WAVE_COLORS[w] for w in wave_names]


# ------------------------------------------------------------------ #
#  ROC curve plots
# ------------------------------------------------------------------ #

def plot_roc_curves(results_df, wave_names, noise_levels, save_path=None):
    """Plot ROC subplots at selected noise levels.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``Wave``, ``Noise``, ``Precision``, ``FPR``,
        ``TPR``, and ``AUC`` columns.
    wave_names : list of str
        Band names to plot.
    noise_levels : list of float
        Subset of noise levels to display (one subplot each).
    save_path : str or None
        If given, save figure to this path.
    """
    colors = get_wave_colors(wave_names)
    n = len(noise_levels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(7 * cols, 5 * rows))
    for i, noise in enumerate(noise_levels):
        plt.subplot(rows, cols, i + 1)
        for j, wave in enumerate(wave_names):
            subset = results_df[
                (results_df["Wave"] == wave) & (results_df["Noise"] == noise)
            ]
            if not subset.empty:
                best = subset.loc[subset["Precision"].idxmax()]
                plt.plot(
                    best["FPR"], best["TPR"],
                    label=f"{wave} (AUC={best['AUC']:.2f})",
                    color=colors[j], linewidth=2.5,
                )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves — Noise Level {noise}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------------ #
#  AUC vs noise
# ------------------------------------------------------------------ #

def plot_auc_vs_noise(results_df, wave_names, noise_levels, log_scale=True,
                      save_path=None):
    """Plot AUC score vs noise level for each wave type."""
    colors = get_wave_colors(wave_names)
    plt.figure(figsize=(12, 7))

    for j, wave in enumerate(wave_names):
        best_results = []
        for noise in noise_levels:
            subset = results_df[
                (results_df["Wave"] == wave) & (results_df["Noise"] == noise)
            ]
            if not subset.empty:
                best = subset.loc[subset["Precision"].idxmax()]
                best_results.append({"Noise": noise, "AUC": best["AUC"]})

        df = pd.DataFrame(best_results)
        plt.plot(
            df["Noise"], df["AUC"],
            marker=MARKERS[j], markersize=10,
            markeredgecolor="white", markeredgewidth=1,
            label=wave, color=colors[j], linewidth=2.5,
        )

    if log_scale:
        plt.xscale("log")
    plt.xlabel("Noise Level" + (" (log scale)" if log_scale else ""), fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.title("AUC Score vs Noise Level for Different Wave Types", fontsize=14)
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.7, linewidth=1.5,
                label="Random Chance (AUC = 0.5)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.45, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------------ #
#  Precision vs noise
# ------------------------------------------------------------------ #

def plot_precision_vs_noise(results_df, wave_names, noise_levels, log_scale=True,
                            save_path=None):
    """Plot precision vs noise level for each wave type."""
    colors = get_wave_colors(wave_names)
    plt.figure(figsize=(12, 7))

    for j, wave in enumerate(wave_names):
        best_results = []
        for noise in noise_levels:
            subset = results_df[
                (results_df["Wave"] == wave) & (results_df["Noise"] == noise)
            ]
            if not subset.empty:
                best = subset.loc[subset["Precision"].idxmax()]
                best_results.append({"Noise": noise, "Precision": best["Precision"]})

        df = pd.DataFrame(best_results)
        plt.plot(
            df["Noise"], df["Precision"],
            marker=MARKERS[j], markersize=10,
            markeredgecolor="white", markeredgewidth=1,
            label=wave, color=colors[j], linewidth=2.5,
        )

    if log_scale:
        plt.xscale("log")
    plt.xlabel("Noise Level" + (" (log scale)" if log_scale else ""), fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision vs Noise Level for Different Wave Types", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------------ #
#  Signal visualisation helpers
# ------------------------------------------------------------------ #

def visualize_sample_signals(frequencies, amplitudes, wave_names, noise_levels,
                             generate_wavelet_fn, get_seq_fn, duration,
                             sampling_rate, noise_fn=None, save_path=None):
    """Render a grid of clean vs noisy signal examples."""
    colors = get_wave_colors(wave_names)
    plt.figure(figsize=(15, 20))

    plt.subplot(6, 2, 1)
    for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        t = np.arange(0, 1, 1 / sampling_rate)
        wavelet = amp * np.sin(2 * np.pi * freq * t)
        t_g = np.linspace(-1, 1, len(wavelet))
        envelope = np.exp(-(t_g**2))
        plt.plot(
            wavelet * envelope,
            label=f"{wave_names[i]} ({freq} Hz)",
            color=colors[i], linewidth=2,
        )
    plt.title("Sample of All Wave Types")
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    sample_levels = noise_levels[:4] if len(noise_levels) >= 4 else noise_levels
    for i, nl in enumerate(sample_levels):
        seq = get_seq_fn(5, len(frequencies), 5)
        clean = generate_wavelet_fn(seq, frequencies, amplitudes, duration, sampling_rate)
        if noise_fn is not None:
            noisy = clean + nl * noise_fn(len(clean))
        else:
            noisy = clean + nl * np.random.randn(len(clean))

        plt.subplot(6, 2, i * 2 + 3)
        plt.plot(clean, color="black", linewidth=1.5)
        plt.title("Clean Signal Sample")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        plt.subplot(6, 2, i * 2 + 4)
        plt.plot(noisy, color="black", linewidth=1.5)
        plt.title(f"Noisy Signal (Noise={nl})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def create_beautiful_roc_curves(results_df, wave_names, save_prefix="eeg"):
    """Create ROC and AUC-vs-noise plots from a real-EEG-noise experiment."""
    colors = WAVE_COLORS

    # ROC subplots
    sample_noise = [0.1, 1.0, 10.0, 100.0]
    plt.figure(figsize=(15, 10))
    for i, noise in enumerate(sample_noise):
        plt.subplot(2, 2, i + 1)
        for wave in wave_names:
            subset = results_df[
                (results_df["Wave"] == wave) & (results_df["Noise"] == noise)
            ]
            if not subset.empty:
                row = subset.iloc[0]
                plt.plot(
                    row["FPR"], row["TPR"],
                    label=f"{wave} (AUC={row['AUC']:.2f})",
                    color=colors[wave], linewidth=2.5,
                )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=10)
        plt.ylabel("True Positive Rate", fontsize=10)
        plt.title(f"ROC Curves — EEG Noise Level {noise}", fontsize=12)
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc_curves.png", dpi=300)
    plt.show()

    # AUC vs noise
    plt.figure(figsize=(12, 7))
    markers = ["o", "s", "D", "^", "p"]
    for j, wave in enumerate(wave_names):
        auc_by_noise = (
            results_df[results_df["Wave"] == wave]
            .groupby("Noise")["AUC"]
            .mean()
        )
        plt.plot(
            auc_by_noise.index, auc_by_noise.values,
            marker=markers[j], markersize=8,
            markerfacecolor=colors[wave], markeredgecolor="white",
            markeredgewidth=1, label=wave, color=colors[wave], linewidth=2,
        )
    plt.xscale("log")
    plt.xlabel("EEG Noise Level (log scale)", fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.title("AUC Score vs EEG Noise Level for Different Brain Waves", fontsize=14)
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.7, linewidth=1.5,
                label="Random Chance (AUC = 0.5)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.45, 1.05])
    plt.savefig(f"{save_prefix}_auc_vs_noise.png", dpi=300)
    plt.show()
