"""
Experiment: ROC Analysis with Real-EEG-Layered Noise

Uses structured EEG-like noise (deterministic harmonic basis with
circular shifting) to produce 100 distinct noise realisations per
condition. A correlation-based template detector is evaluated to
generate realistic ROC curves and AUC summaries.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eeg.analysis.roc_real_eeg import (
    generate_clean_signal,
    generate_frequency_band_signal,
    create_perfect_roc_dataset,
    test_noise_patterns,
    WAVE_NAMES,
    NOISE_LEVELS,
    MAX_SHIFT,
)
from src.eeg.analysis.noise import generate_clean_eeg_noise
from src.eeg.analysis.plotting import (
    create_beautiful_signal_examples,
    create_beautiful_roc_curves,
    WAVE_COLORS,
)

# ------------------------------------------------------------------ #
#  1. Signal visualisations
# ------------------------------------------------------------------ #

# Create beautiful clean vs. noisy signal examples
from src.eeg.analysis.roc_real_eeg import (
    FREQUENCIES, AMPLITUDES, DURATION, SAMPLING_RATE,
)

# Custom signal examples
plt.figure(figsize=(15, 15))
colors = WAVE_COLORS

plt.subplot(4, 2, 1)
for i, (freq, amp, wave) in enumerate(zip(FREQUENCIES, AMPLITUDES, WAVE_NAMES)):
    t = np.arange(0, 1, 1 / SAMPLING_RATE)
    sig = amp * np.sin(2 * np.pi * freq * t)
    plt.plot(sig[:100], label=f"{wave} ({freq} Hz)", color=colors[wave], linewidth=2)
plt.title("Brain Wave Types", fontsize=12)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

sample_noise_levels = [0.5, 5.0, 50.0]
shifts = [1000, 25000, 60000]
for i, (nl, shift) in enumerate(zip(sample_noise_levels, shifts)):
    from src.eeg.analysis.noise import add_noise_to_signal
    clean = generate_clean_signal(i + 1)
    noisy = add_noise_to_signal(clean, nl, shift)

    plt.subplot(4, 2, i * 2 + 3)
    plt.plot(clean, "k-", linewidth=1)
    plt.title("Clean Signal Sample", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, i * 2 + 4)
    plt.plot(noisy, "k-", linewidth=0.8)
    plt.title(f"Noisy Signal (EEG Noise Level={nl})", fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eeg_signal_samples.png", dpi=300)
plt.show()

# ------------------------------------------------------------------ #
#  2. Multiple noise patterns
# ------------------------------------------------------------------ #
clean_signal = generate_clean_signal(1)
noisy_signals = test_noise_patterns(clean_signal, noise_level=1.0)

plt.figure(figsize=(15, 10))
for i in range(min(9, len(noisy_signals))):
    plt.subplot(3, 3, i + 1)
    plt.plot(noisy_signals[i], "k-", linewidth=0.8)
    plt.title(f"Noise Pattern {i + 1}", fontsize=10)
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multiple_noise_patterns.png", dpi=300)
plt.show()
print(f"Generated {len(noisy_signals)} different noise patterns")

# ------------------------------------------------------------------ #
#  3. Realistic ROC curves
# ------------------------------------------------------------------ #
print("Generating realistic ROC curves with multiple noise patterns...")
results_df = create_perfect_roc_dataset()
create_beautiful_roc_curves(results_df, WAVE_NAMES, save_prefix="eeg")

# ------------------------------------------------------------------ #
#  4. ROC variability across trials
# ------------------------------------------------------------------ #
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(15, 10))
demo_noise = 5.0

for trial in range(5):
    plt.subplot(1, 5, trial + 1)
    shift_start = trial * MAX_SHIFT // 10
    shifts = np.linspace(shift_start, shift_start + MAX_SHIFT // 10, 20, dtype=int)

    for wave_idx, wave in enumerate(WAVE_NAMES):
        y_true, y_scores = [], []
        for shift in shifts:
            clean = generate_frequency_band_signal(wave_idx)
            noise = generate_clean_eeg_noise(len(clean), shift) * demo_noise

            noisy = clean + noise
            noisy = (noisy - np.mean(noisy)) / np.std(noisy)
            template = generate_frequency_band_signal(wave_idx)
            template = (template - np.mean(template)) / np.std(template)
            y_true.append(1)
            y_scores.append(np.correlate(noisy, template, mode="valid")[0])

            noise_only = (noise - np.mean(noise)) / np.std(noise)
            y_true.append(0)
            y_scores.append(np.correlate(noise_only, template, mode="valid")[0])

        fpr, tpr, _ = roc_curve(np.array(y_true), np.array(y_scores))
        plt.plot(fpr, tpr, label=wave[:1], color=colors[wave], linewidth=1.5)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title(f"Trial {trial + 1}", fontsize=10)
    if trial == 0:
        plt.ylabel("True Positive Rate", fontsize=9)
        plt.legend(loc="lower right", fontsize=8)

plt.suptitle(
    "Variability in ROC Curves with Different Noise Patterns (Noise Level = 5.0)",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("roc_curve_variability.png", dpi=300)
plt.show()

print("All visualisations complete!")
