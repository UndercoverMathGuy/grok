"""Losses, accuracies, progress measures, and checkpoint sweeps.

All functions here work on numpy arrays in float64 — float32 log_softmax
underflows once the model is very confident, which matters because late-
training losses reach ~1e-8. `all_logits(model, tokens)` is the bridge from
an MLX model to numpy: logits at the final position for the full p^2 batch,
with the "=" logit dropped so the output lives in R^p.
"""

import numpy as np

from .fourier import Fourier


def _log_softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def all_logits(model, tokens):
    """(p^2, p) float64 logits at the final position, "=" logit dropped."""
    import mlx.core as mx

    logits = model(tokens)[:, -1, :-1]
    mx.eval(logits)
    return np.array(logits, dtype=np.float64)


def cross_entropy_high_precision(logits, labels, mask=None):
    """Mean CE in float64. `mask` restricts to a boolean subset (e.g. is_train)."""
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    logprobs = _log_softmax(logits)
    return -logprobs[np.arange(len(labels)), labels].mean()


def accuracy(logits, labels, mask=None):
    correct = logits.argmax(axis=-1) == labels
    if mask is not None:
        correct = correct[mask]
    return correct.mean()


# ---------------------------------------------------------------------------
# Progress measures (Section 5 of the paper)
# ---------------------------------------------------------------------------

def circuit_only_logits(logits, fourier: Fourier, key_freqs, center_to=None):
    """Keep only the cos/sin(w(a+b)) components for the key frequencies.

    If `center_to` is given (the original logits), add back the mean logit as
    a bias correction, as in the paper.
    """
    out = sum(fourier.freq_contribution(logits, freq) for freq in key_freqs)
    if center_to is not None:
        out = out + center_to.mean(axis=0, keepdims=True) - out.mean(axis=0, keepdims=True)
    return out


def circuit_excluded_logits(logits, fourier: Fourier, freqs):
    """Remove the cos/sin(w(a+b)) components for the given frequencies."""
    out = logits.copy()
    for freq in freqs:
        out -= fourier.freq_contribution(logits, freq)
    return out


def cos_w_product_a_plus_b_minus_c_coeffs(logits, fourier: Fourier):
    """Coefficient of cos(w(a+b-c)) in the logits, for every frequency w."""
    return np.array(
        [
            (fourier.cos_w_product_a_plus_b_minus_c(freq) * logits).sum()
            for freq in range(1, fourier.p // 2 + 1)
        ]
    )


def kl_divergence(p_logits, q_logits, temp=1.0, mask=None):
    """Mean KL(softmax(p/T) || softmax(q/T)) per row, in float64.

    Use this (not the float32 training loss) when analysing how far a
    student's logit distribution is from a teacher's at a given checkpoint.
    """
    if mask is not None:
        p_logits, q_logits = p_logits[mask], q_logits[mask]
    p_logp = _log_softmax(np.asarray(p_logits, dtype=np.float64) / temp)
    q_logp = _log_softmax(np.asarray(q_logits, dtype=np.float64) / temp)
    return (np.exp(p_logp) * (p_logp - q_logp)).sum(axis=-1).mean()


def freq_energy(logits, fourier: Fourier):
    """(p//2,) squared norm of the logits' projection onto each frequency's
    cos/sin(w(a+b)) plane — the spectrum whose rotation the distillation
    experiment tracks. Row k-1 is frequency k."""
    return np.array(
        [
            (fourier.freq_contribution(logits, freq) ** 2).sum()
            for freq in range(1, fourier.p // 2 + 1)
        ]
    )


def freq_coeffs_and_energy(logits, fourier: Fourier):
    """(coeffs, energy) over all frequencies in one pass.

    Equivalent to cos_w_product_a_plus_b_minus_c_coeffs + freq_energy but
    computed via one stacked matmul instead of one projection per frequency —
    fast enough to call every few epochs during training (the spectral
    logger). Row k-1 is frequency k.
    """
    n = fourier.p // 2
    u_cos = np.stack([fourier.cos_a_plus_b(k) for k in range(1, n + 1)])
    u_sin = np.stack([fourier.sin_a_plus_b(k) for k in range(1, n + 1)])
    c = u_cos @ logits  # (n, p): each freq's cos(w(a+b)) component per output
    s = u_sin @ logits
    energy = (c**2).sum(axis=1) + (s**2).sum(axis=1)
    # cos(w(a+b-c)) = cos(w(a+b))cos(wc) + sin(w(a+b))sin(wc), so the
    # normalized template contracts to (c . cos_wc + s . sin_wc) / sqrt(2).
    b_cos, b_sin = fourier.basis[1::2], fourier.basis[2::2]
    coeffs = ((c * b_cos).sum(axis=1) + (s * b_sin).sum(axis=1)) / np.sqrt(2)
    return coeffs, energy


def sum_sq_weights(model):
    """Total sum of squared parameters (the weight-decay energy)."""
    from mlx.utils import tree_flatten

    return sum(float((v**2).sum()) for _, v in tree_flatten(model.parameters()))


# ---------------------------------------------------------------------------
# Neuron frequency clustering (Section 4.2)
# ---------------------------------------------------------------------------

def neuron_freq_clusters(neuron_acts, fourier: Fourier, threshold=0.85):
    """Cluster MLP neurons by their dominant frequency.

    neuron_acts: (p^2, d_mlp) final-position MLP activations on the full batch.

    Returns (neuron_freqs, neuron_frac_explained, key_freqs):
      neuron_freqs        — (d_mlp,) dominant freq per neuron; -1 if no single
                            frequency explains > threshold of its variance
      neuron_frac_explained — (d_mlp,) that max fraction of variance
      key_freqs           — sorted unique dominant frequencies
    """
    centered = neuron_acts - neuron_acts.mean(axis=0, keepdims=True)
    fourier_acts = fourier.fft2d(centered)  # (p^2, d_mlp)
    norms = (fourier_acts**2).sum(axis=0)

    # Candidate frequencies 1..p//2-1: the original searches range(1, p//2),
    # excluding p//2 itself.
    candidate_freqs = np.arange(1, fourier.p // 2)
    freq_idx = np.stack([fourier.freq_indices_2d(f) for f in candidate_freqs])
    # (n_freqs, d_mlp): fraction of each neuron's variance on each freq's terms
    explained = (fourier_acts[freq_idx] ** 2).sum(axis=1) / norms

    neuron_frac_explained = explained.max(axis=0)
    neuron_freqs = candidate_freqs[explained.argmax(axis=0)]
    key_freqs = np.unique(neuron_freqs)
    neuron_freqs[neuron_frac_explained < threshold] = -1
    return neuron_freqs, neuron_frac_explained, key_freqs


# ---------------------------------------------------------------------------
# Sweeping metrics over training checkpoints
# ---------------------------------------------------------------------------

def sweep_checkpoints(model, ckpt_epochs, metric_fn):
    """Evaluate metrics at every saved checkpoint.

    ckpt_epochs: [(epoch, path)] from train.checkpoint_epochs.
    metric_fn:   fn(model) -> {name: scalar or array}. Compute expensive
                 shared quantities (e.g. logits) once inside it and derive
                 every metric from them — the model runs once per checkpoint.

    Returns {name: np.array with leading checkpoint dim} plus an 'epochs'
    array. Restores the final checkpoint's weights before returning.
    """
    from tqdm.auto import tqdm

    rows = {}
    for epoch, path in tqdm(ckpt_epochs):
        model.load_weights(str(path))
        for name, value in metric_fn(model).items():
            rows.setdefault(name, []).append(value)
    model.load_weights(str(ckpt_epochs[-1][1]))
    out = {name: np.array(vals) for name, vals in rows.items()}
    out["epochs"] = np.array([e for e, _ in ckpt_epochs])
    return out
