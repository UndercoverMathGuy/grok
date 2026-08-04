"""Fourier analysis over Z_p, the core lens of the grokking paper.

`Fourier(p)` bundles the 1D basis (constant, cos 1, sin 1, cos 2, ...) with
the projections used throughout the analysis. Everything is numpy — analysis
is cheap and numpy keeps it dead simple to poke at. Convert model weights
with plotting.to_numpy first.

Conventions: a "flattened" tensor has leading dim p^2 enumerating inputs
(a, b) lexicographically; "square" means that dim is unflattened to (p, p).
"""

import numpy as np


class Fourier:
    def __init__(self, p: int):
        self.p = p
        basis = [np.ones(p) / np.sqrt(p)]
        self.names = ["Const"]
        for k in range(1, p // 2 + 1):
            for trig, name in [(np.cos, "cos"), (np.sin, "sin")]:
                vec = trig(2 * np.pi * k * np.arange(p) / p)
                basis.append(vec / np.linalg.norm(vec))
                self.names.append(f"{name} {k}")
        # (p, p): row i is the i-th normalized basis vector
        self.basis = np.stack(basis).astype(np.float64)

    def fft1d(self, x):
        """Transform a tensor whose last dim has length p into the Fourier basis."""
        return x @ self.basis.T

    def fft2d(self, mat):
        """2D transform of a (p^2, ...) or (p, p, ...) tensor over its input dims.

        Output has the same shape as the input.
        """
        p, shape = self.p, mat.shape
        mat = mat.reshape(p, p, -1)
        mat_by_output = mat.transpose(2, 0, 1)
        out = self.basis @ mat_by_output @ self.basis.T
        return out.transpose(1, 2, 0).reshape(shape)

    def basis_vecs_2d(self, x_index, y_index):
        """2D basis vector (length p^2): outer product of two 1D components."""
        return np.outer(self.basis[x_index], self.basis[y_index]).flatten()

    def cos_a_plus_b(self, freq):
        """Unit vector in R^{p^2} for cos(w_freq * (x + y))."""
        return (
            self.basis_vecs_2d(2 * freq - 1, 2 * freq - 1)
            - self.basis_vecs_2d(2 * freq, 2 * freq)
        ) / np.sqrt(2)

    def sin_a_plus_b(self, freq):
        """Unit vector in R^{p^2} for sin(w_freq * (x + y))."""
        return (
            self.basis_vecs_2d(2 * freq, 2 * freq - 1)
            + self.basis_vecs_2d(2 * freq - 1, 2 * freq)
        ) / np.sqrt(2)

    def freq_contribution(self, tensor, freq):
        """Project a (p^2, ...) tensor onto the cos+sin(w(x+y)) plane for freq."""
        out = np.zeros_like(tensor, dtype=np.float64)
        for vec in [self.cos_a_plus_b(freq), self.sin_a_plus_b(freq)]:
            out += vec[:, None] * (vec @ tensor)
        return out

    def freq_indices_2d(self, freq):
        """Flat (p*p) indices of the 8 linear+quadratic 2D terms of a frequency:
        cos/sin on x alone, on y alone, and the 4 quadratic cross terms."""
        p, c, s = self.p, 2 * freq - 1, 2 * freq
        return np.array([c, s, c * p, s * p, c * p + c, c * p + s, s * p + c, s * p + s])

    def cos_w_product_a_plus_b_minus_c(self, freq):
        """cos(w(a + b - c)) as a normalized (p^2, p) logit template."""
        p = self.p
        a = np.arange(p)[:, None, None]
        b = np.arange(p)[None, :, None]
        c = np.arange(p)[None, None, :]
        w = 2 * np.pi * freq / p
        template = np.cos(w * (a + b - c)).reshape(p * p, p)
        return template / np.linalg.norm(template)
