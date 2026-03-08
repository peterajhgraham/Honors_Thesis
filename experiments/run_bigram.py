"""
Experiment: Bigram Sequence Generation and Validation

Generates a bigram token sequence, validates that empirical transition
probabilities converge to the sampled conditional distribution, and
produces heatmap visualisations comparing the two.
"""

import numpy as np
from src.sequences.bigram import generate_bigram_sequence, plot_heatmap

# --- Generate sequence ---
n_tokens = 5
sequence, probs_cond = generate_bigram_sequence(
    n_tokens=n_tokens, seqlen=10_000, sparsity=0
)

# --- Validation tests ---
assert len(sequence) == 10_000, "Sequence length is incorrect."
assert np.all((sequence >= 1) & (sequence <= n_tokens)), "Tokens are out of range."
assert sequence[0] != 0, "The first token should not be zero."

# Compute empirical transition probabilities
empirical_prob = np.zeros_like(probs_cond)
for i in range(n_tokens):
    for j in range(n_tokens):
        loc_i = np.where(sequence == i + 1)[0]
        loc_j = np.where(sequence == j + 1)[0]
        count = sum(1 for k in loc_i if k + 1 in loc_j)
        empirical_prob[i, j] = count / len(loc_i)

print("Relative error (empirical vs. true):")
print(abs(empirical_prob - probs_cond) / probs_cond)
print("All tests passed!")

# --- Visualise ---
labels = [f"Token {i + 1}" for i in range(n_tokens)]
plot_heatmap(empirical_prob, title="Empirical Probability Heatmap",
             x_labels=labels, y_labels=labels)
plot_heatmap(probs_cond, title="Conditional Probability Heatmap",
             x_labels=labels, y_labels=labels)
