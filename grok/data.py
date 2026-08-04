"""Dataset: all p^2 pairs (a, b) as sequences [a, b, =], labelled fn(a, b).

The full dataset is small enough (p^2 = 12769 sequences) to hold in memory
and train on as a single batch, so there is no DataLoader — just arrays.
"""

import random

import mlx.core as mx
import numpy as np

from .config import Config


def make_dataset(cfg: Config):
    """Return (tokens, labels) for all p^2 inputs in lexicographic order.

    tokens: (p^2, 3) int32 — rows are [a, b, p] where token p is "=".
    labels: (p^2,)   int32 — fn(a, b).

    The batch dimension enumerates (a, b) lexicographically, so it can be
    reshaped to (p, p, ...) with axis 0 = a and axis 1 = b.
    """
    p = cfg.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = np.stack([a, b, np.full(p * p, p)], axis=1)
    labels = cfg.fn(a, b)
    return mx.array(tokens, dtype=mx.int32), mx.array(labels, dtype=mx.int32)


def train_test_split(cfg: Config):
    """Boolean masks (is_train, is_test) over the lexicographic batch.

    Uses python's `random` with cfg.data_seed, replicating the original torch
    codebase exactly — so checkpoints converted from the original run see
    the identical split.
    """
    p = cfg.p
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.seed(cfg.data_seed)
    random.shuffle(pairs)
    div = int(cfg.frac_train * len(pairs))
    train_set = set(pairs[:div])

    is_train = np.array([(i, j) in train_set for i in range(p) for j in range(p)])
    return is_train, ~is_train
