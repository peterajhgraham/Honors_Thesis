"""
Bigram Sequence Generation

Generates random token sequences governed by learned bigram (conditional)
probability distributions. A Dirichlet prior is used to sample both the
initial token distribution and the row-stochastic transition matrix.

Functions
---------
generate_bigram_sequence : Generate a sequence of tokens with bigram statistics.
plot_heatmap             : Render an annotated heatmap for a probability matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_bigram_sequence(n_tokens=5, seqlen=1000, sparsity=0):
    """Generate a random sequence with random bigram statistics.

    Parameters
    ----------
    n_tokens : int
        Number of distinct tokens (vocabulary size).
    seqlen : int
        Length of the generated sequence.
    sparsity : float
        Probability of zeroing out any given token position.

    Returns
    -------
    sequence : np.ndarray
        Integer token sequence of shape ``(seqlen,)``.
    probs_cond : np.ndarray
        Conditional probability matrix of shape ``(n_tokens, n_tokens)``.
    """
    # Initial distribution of tokens
    probs_initial = np.random.dirichlet(np.ones(n_tokens))

    # Initialize conditional probability matrix
    probs_cond = np.zeros((n_tokens, n_tokens))

    # Set up conditional distributions
    for token_n in range(n_tokens):
        probs_cond[token_n, :] = np.random.dirichlet(np.ones(n_tokens))

    # Initialize sequence
    sequence = np.zeros(seqlen, dtype=int)

    # Sample initial token
    sequence[0] = np.random.choice(n_tokens, 1, p=probs_initial) + 1

    # Generate subsequent samples
    for pos in range(1, seqlen):
        sequence[pos] = np.random.choice(
            n_tokens, 1, p=probs_cond[(sequence[pos - 1] - 1), :]
        ) + 1

    # Make some tokens zero (sparsity)
    mask = np.random.rand(seqlen) < sparsity
    sequence[mask] = 0

    return sequence, probs_cond


def plot_heatmap(matrix, title="Heatmap", x_labels=None, y_labels=None):
    """Plot an annotated heatmap for a probability matrix.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array to visualize.
    title : str
        Plot title.
    x_labels, y_labels : list of str or None
        Tick labels for columns and rows.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, cmap="coolwarm", cbar=True,
        xticklabels=x_labels, yticklabels=y_labels,
    )
    plt.title(title)
    plt.show()
