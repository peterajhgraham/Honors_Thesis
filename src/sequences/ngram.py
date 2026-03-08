"""
N-gram Sequence Generation

Provides utilities for generating token sequences governed by higher-order
n-gram transition probabilities. Both a full Dirichlet-sampled version
and a simplified uniform-random version are included.

Functions
---------
get_ngram_sequence        : Generate a sequence using full n-gram transition tables.
get_ngram_sequence_simple : Generate a uniformly random token sequence.
"""

import numpy as np


def get_ngram_sequence(ngram, n_tokens, seqlen):
    """Generate a token sequence using full n-gram transition probabilities.

    A multidimensional transition tensor of shape ``(n_tokens,) * ngram``
    is sampled from Dirichlet distributions. The first ``ngram`` tokens are
    drawn from marginals; subsequent tokens are drawn conditioned on the
    preceding ``ngram - 1`` tokens.

    Parameters
    ----------
    ngram : int
        Order of the n-gram model.
    n_tokens : int
        Number of distinct tokens.
    seqlen : int
        Length of the sequence to generate.

    Returns
    -------
    sequence : np.ndarray
        Integer token sequence of shape ``(seqlen,)``.
    """
    probs = np.zeros([n_tokens] * ngram)
    for indices in np.ndindex(*probs.shape[:-1]):
        probs[(*indices, slice(None))] = np.random.dirichlet(np.ones(n_tokens))

    sequence = np.zeros(seqlen, dtype=int)
    for pos in range(ngram):
        sequence[pos] = np.random.choice(
            n_tokens,
            p=np.mean(probs, axis=tuple(range(ngram - (pos + 1))))[
                tuple(sequence[:pos])
            ],
        )
    for pos in range(ngram, seqlen):
        sequence[pos] = np.random.choice(
            n_tokens, p=probs[tuple(sequence[pos - ngram + 1 : pos])]
        )
    return sequence


def get_ngram_sequence_simple(ngram, n_tokens, seqlen):
    """Generate a uniformly random token sequence (simplified n-gram).

    Parameters
    ----------
    ngram : int
        Unused; kept for API compatibility.
    n_tokens : int
        Number of distinct tokens.
    seqlen : int
        Length of the sequence to generate.

    Returns
    -------
    sequence : np.ndarray
        Integer token sequence of shape ``(seqlen,)``.
    """
    return np.random.choice(n_tokens, seqlen)
