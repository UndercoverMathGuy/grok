"""Surrogate-flow proxy: does a minimal quadratic network trained on the SAME
mask reproduce the transformer's mask-specific frequency favorites?

Model (Gromov-style): logits(c|a,b) = sum_n U[c,n] * (V[n,a] + W[n,b])^2
MSE on centered one-hot labels, train rows only, full-batch GD + momentum.
MLX (GPU) version — gradients written by hand, scatter done via one-hot
matmuls. Read the audition spectrum (cos(w_k(a+b-c)) amplitude of full-grid
logits) at memorization (train acc ~ 1), for many inits on one mask.

Readouts:
  1. within-toy consistency: do different inits of the TOY agree on top-8?
  2. transfer: do toy favorites match the transformer's popularity on dseed0?
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from grok.config import Config
from grok.data import train_test_split
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

P = 113
N = 256          # neurons
SIGMA = 0.3      # init scale
LR = 1e-3        # Adam (matches the transformer's optimizer family)
B1, B2, EPS = 0.9, 0.98, 1e-8
MAX_STEPS = 20000


def spectrum(logits, p):
    """|cos(w_k(a+b-c)) amplitude| per k from full-grid (p^2, p) logits."""
    a, b = np.divmod(np.arange(p * p), p)
    x = (a[:, None] + b[:, None] - np.arange(p)[None, :]) % p
    Lx = np.zeros(p)
    np.add.at(Lx, x.ravel(), logits.ravel())
    Lx /= p * p
    ks = np.arange(1, p // 2 + 1)
    amp = (2.0 / p) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                                            * np.arange(p)[None, :] / p)).sum(1)
    return np.abs(amp)


def run_toy(seed, is_train, p=P):
    rng = np.random.default_rng(seed)
    V = mx.array(rng.normal(0, SIGMA, (N, p)).astype(np.float32))
    W = mx.array(rng.normal(0, SIGMA, (N, p)).astype(np.float32))
    U = mx.array(rng.normal(0, SIGMA / np.sqrt(N), (p, N)).astype(np.float32))

    a, b = np.divmod(np.arange(p * p), p)
    ta, tb = a[is_train], b[is_train]
    n = ta.size
    labels_np = (ta + tb) % p
    Oa = np.zeros((n, p), np.float32); Oa[np.arange(n), ta] = 1
    Ob = np.zeros((n, p), np.float32); Ob[np.arange(n), tb] = 1
    y_np = np.full((n, p), -1.0 / p, np.float32)
    y_np[np.arange(n), labels_np] += 1.0
    Oa, Ob, y = mx.array(Oa), mx.array(Ob), mx.array(y_np)
    labels = mx.array(labels_np)

    zeros = lambda: (mx.zeros_like(V), mx.zeros_like(W), mx.zeros_like(U))
    state = (V, W, U, *zeros(), *zeros(), mx.array(0.0))

    def step(V, W, U, mV, mW, mU, sV, sW, sU, t):
        H = V @ Oa.T + W @ Ob.T              # (N, n)
        A = H * H
        R = A.T @ U.T - y                    # (n, p)
        gU = R.T @ A.T / n                   # (p, N)
        B = (U.T @ R.T) * (2 * H) / n        # (N, n)
        gV = B @ Oa
        gW = B @ Ob
        t = t + 1
        out = []
        for P_, g, m, s in ((V, gV, mV, sV), (W, gW, mW, sW), (U, gU, mU, sU)):
            m = B1 * m + (1 - B1) * g
            s = B2 * s + (1 - B2) * g * g
            mhat = m / (1 - B1 ** t)
            shat = s / (1 - B2 ** t)
            out.append((P_ - LR * mhat / (mx.sqrt(shat) + EPS), m, s))
        (V, mV, sV), (W, mW, sW), (U, mU, sU) = out
        return V, W, U, mV, mW, mU, sV, sW, sU, t

    cstep = mx.compile(step)

    def train_acc(V, W, U):
        H = V @ Oa.T + W @ Ob.T
        out = (H * H).T @ U.T
        return float((mx.argmax(out, axis=1) == labels).mean())

    last = 0
    for s in range(MAX_STEPS):
        state = cstep(*state)
        if s % 100 == 99:
            mx.eval(*state)
            acc = train_acc(*state[:3])
            last = s
            if acc >= 0.999:
                break
    V, W, U = state[:3]

    # full-grid logits at memorization
    Vn, Wn, Un = (np.array(t, dtype=np.float64) for t in (V, W, U))
    Hf = Vn[:, a] + Wn[:, b]
    logits = (Hf * Hf).T @ Un.T
    acc = (logits[is_train].argmax(1) == labels_np).mean()
    return spectrum(logits, p), acc, last


def main():
    cfg = Config(p=P, data_seed=0)
    is_train, _ = train_test_split(cfg)

    # transformer popularity on this mask
    pop = np.zeros(P // 2)
    n_tf = 0
    for d, c in discover():
        if (c.p, c.data_seed) != (P, 0):
            continue
        coeffs, acc, _ = final_coeffs_and_acc(d, c)
        if acc < 0.99:
            continue
        n_tf += 1
        for k in committee_from_coeffs(coeffs):
            pop[k - 1] += 1
    tf_fav = (np.argsort(pop)[::-1][:8] + 1).tolist()
    print(f"transformer popularity (n={n_tf} runs), top-8: {tf_fav}", flush=True)

    tops = []
    for seed in range(10):
        sp, acc, steps = run_toy(seed, is_train)
        top8 = (np.argsort(sp)[::-1][:8] + 1).tolist()
        tops.append(set(top8))
        print(f"toy seed {seed}: acc {acc:.3f} @ step {steps:4d}  "
              f"top-8 {sorted(top8)}", flush=True)

    # 1. within-toy consistency
    ov = [len(tops[i] & tops[j]) for i in range(len(tops))
          for j in range(i + 1, len(tops))]
    # null: two random 8-subsets of 56 share 8*8/56 = 1.14
    print(f"\nwithin-toy top-8 overlap: mean {np.mean(ov):.2f} (chance 1.14)")

    # 2. transfer to transformer favorites
    toy_pop = np.zeros(P // 2)
    for t in tops:
        for k in t:
            toy_pop[k - 1] += 1
    from scipy.stats import spearmanr
    rho, pv = spearmanr(toy_pop, pop)
    print(f"toy popularity vs transformer popularity: Spearman rho={rho:+.3f} "
          f"(p={pv:.4f})")
    toy_fav = (np.argsort(toy_pop)[::-1][:8] + 1).tolist()
    print(f"toy top-8 favorites: {toy_fav}")
    print(f"overlap with transformer top-8: {len(set(toy_fav) & set(tf_fav))} "
          f"(chance 1.14)")


if __name__ == "__main__":
    main()
