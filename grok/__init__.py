"""Grokking on modular addition — MLX port of the "Progress Measures for
Grokking via Mechanistic Interpretability" codebase (Nanda et al., 2023).

Modules:
    config    — hyperparameters for the model and training run
    data      — modular-arithmetic dataset generation and train/test split
    model     — 1-layer transformer with activation caching
    train     — training loop with checkpointing (`python -m grok.train`)
    fourier   — Fourier basis over Z_p and projections used in the analysis
    metrics   — losses, accuracies, progress measures, checkpoint sweeps
    plotting  — vanilla-plotly helpers (imshow, lines, fourier heatmaps, ...)
"""

from .config import Config
from .data import make_dataset, train_test_split
from .model import Transformer
from .fourier import Fourier

__all__ = ["Config", "make_dataset", "train_test_split", "Transformer", "Fourier"]
