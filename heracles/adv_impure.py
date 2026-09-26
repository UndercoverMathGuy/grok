"""adv_impure.py -- ADVERSARIAL attack on gap G2: can an IMPURE neuron (one carrying
several frequencies at once) beat the (n+1)-block?  "Can one neuron clean two
frequencies more cheaply than two blocks?"

Setting as in adv_net.py.  A neuron i is PURE at frequency k if all n of its operand
embeddings E[j, i, :] and its readout W[:, i] live in the two-dimensional Fourier
subspace span{cos(2 pi k x / p), sin(2 pi k x / p)}.  A general neuron is a sum over
frequencies; call PURIFICATION the map that replaces neuron i by the (p+1)/2 pure
neurons obtained by projecting all of its vectors onto each frequency subspace in turn
(k = 0 included, which contributes nothing to any logit gap).

Two exact facts this script tests and quantifies.

(P1) COST.  Write A_ik = sum_j ||Proj_k E[j,i]||^2 + ||Proj_k W[:,i]||^2, so that the
     neuron's squared norm is nrm2_i = sum_k A_ik (Parseval, the subspaces are
     orthogonal).  The penalty of the impure neuron is (sum_k A_ik)^{q/2}, that of its
     purification is sum_k A_ik^{q/2}.  For q >= 2 the map t -> t^{q/2} is convex with
     value 0 at 0, hence superadditive:

         sum_k A_ik^{q/2}  <=  ( sum_k A_ik )^{q/2} ,  equality for q = 2 exactly.

     *Plain reading: purifying a network never costs more, and at plain weight decay
     (q = 2) it costs exactly the same -- so impurity is COST-FREE at q = 2 and strictly
     WASTEFUL for q > 2.*

(P2) CONTENT.  The logit function of the purified net is the part of the original whose
     Fourier support lies on characters (k_1..k_n; k_c) with k_1 = ... = k_n = -k_c
     ("same frequency in every slot"); every removed cross term has zero mean over the
     p^n rows at every offset D.  So purification preserves the clean amplitudes
     g_k = sum_i prod_j Ehat[j,i,k] What[i,-k] EXACTLY, and removes all cross-frequency
     junk while keeping same-frequency junk.

What is NOT a theorem (and is what this script hunts for): that purification never
LOWERS the margin.  The margin is a min over rows, and a removed cross term could in
principle be positive exactly at the worst row.  If

     ratio(theta) > ratio(purified theta)      anywhere,

impurity buys something, which is the only way G2 can matter -- and if it ever pushes
ratio above 1, (T) is refuted.

Blocks
  V  verify (P1) and (P2) numerically on random nets (Parseval, g_k preserved, cross
     terms have zero row-mean).
  S  scan: random / trained-shape / deliberately impure nets, compare ratio(theta) with
     ratio(purified) -- the sign of the difference is the whole question.
  O  optimise the DIFFERENCE ratio(theta) - ratio(purified theta) directly (soft-min
     ascent on the impure net, exact evaluation of both), i.e. search for the best
     possible case for impurity.
  F  forced impurity: every neuron is constrained to carry two frequencies with
     comparable amplitude, then the ratio is maximised.  Does it ever exceed 1?

All float64, local CPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "FINAL_RUNS" / "ADV"

from adv_net import (Task, exact_margin, penalty, gamma_prog, ascend,  # noqa: E402
                     make_starts, ratio_of)


# ------------------------------------------------------------------ Fourier tools
def basis(p):
    """Orthogonal real Fourier basis of R^p, grouped by frequency. Returns list of
    (k, B_k) with B_k of shape (d_k, p), rows orthonormal."""
    x = np.arange(p)
    out = [(0, np.ones((1, p)) / np.sqrt(p))]
    for k in range(1, (p - 1) // 2 + 1):
        c = np.cos(2 * np.pi * k * x / p)
        s = np.sin(2 * np.pi * k * x / p)
        c = c / np.linalg.norm(c)
        s = s / np.linalg.norm(s)
        out.append((k, np.stack([c, s])))
    return out


def purify(E, W, p, n, drop_dc=False):
    """Replace every neuron by its per-frequency projections. Returns (E', W')."""
    m = E.shape[1]
    bs = basis(p)
    if drop_dc:
        bs = bs[1:]
    nk = len(bs)
    E2 = np.zeros((n, m * nk, p))
    W2 = np.zeros((p, m * nk))
    for t, (k, B) in enumerate(bs):
        P = B.T @ B                       # orthogonal projector onto frequency k
        for j in range(n):
            E2[j, t * m:(t + 1) * m] = E[j] @ P.T
        W2[:, t * m:(t + 1) * m] = P @ W
    return E2, W2


def clean_amps(E, W, p, n):
    """g_k = sum_i prod_j Ehat[j,i,k] * What[i,-k], with Ehat[j,i,k] = (1/p) sum_a E e^{-2pi i k a/p}."""
    x = np.arange(p)
    ks = np.arange(0, (p - 1) // 2 + 1)
    ph = np.exp(-2j * np.pi * np.outer(ks, x) / p) / p
    Eh = np.einsum("ka,jia->jik", ph, E)            # (n, m, nk)
    Wh = np.einsum("ka,ai->ik", ph, W)              # (m, nk)
    g = np.ones(Eh.shape[2], dtype=complex)
    tot = np.zeros(Eh.shape[2], dtype=complex)
    for i in range(E.shape[1]):
        t = np.ones(Eh.shape[2], dtype=complex)
        for j in range(n):
            t = t * Eh[j, i]
        tot = tot + t * np.conj(Wh[i])              # What[i,-k] = conj(What[i,k]) for real W
    return tot


# ------------------------------------------------------------------ blocks
def block_V(rng, log, trials=12):
    rows = []
    for (p, n, m, q) in [(7, 2, 5, 2.0), (11, 2, 6, 2.0), (11, 2, 6, 2.5),
                         (13, 2, 5, 2.9), (5, 3, 5, 2.0), (7, 3, 4, 3.0)]:
        tk = Task(p, n)
        for _ in range(trials):
            E = rng.normal(size=(n, m, p))
            W = rng.normal(size=(p, m))
            E2, W2 = purify(E, W, p, n)
            P1, _ = penalty(E, W, q)
            P2, _ = penalty(E2, W2, q)
            g1 = clean_amps(E, W, p, n)
            g2 = clean_amps(E2, W2, p, n)
            # row-mean of the gap function at each D must agree between the two nets
            _, z1 = _fwd(E, W, tk)
            _, z2 = _fwd(E2, W2, tk)
            mm = []
            for D in range(1, p):
                c = (tk.y - D) % p
                phi1 = z1[tk.ar, tk.y] - z1[tk.ar, c]
                phi2 = z2[tk.ar, tk.y] - z2[tk.ar, c]
                mm.append(abs(phi1.mean() - phi2.mean()))
            rows.append(dict(p=p, n=n, q=q, cost_ratio=P2 / P1,
                             gmax=float(np.abs(g1 - g2).max()),
                             meandiff=float(max(mm)),
                             m1=exact_margin(E, W, tk), m2=exact_margin(E2, W2, tk)))
    cr = np.array([r["cost_ratio"] for r in rows])
    q2 = np.array([r["cost_ratio"] for r in rows if r["q"] == 2.0])
    log(f"[V] purification cost ratio P_q(purified)/P_q(theta): max {cr.max():.12f} "
        f"(must be <= 1);  at q = 2 exactly: max |ratio - 1| = {np.abs(q2 - 1).max():.3e}")
    log(f"[V] clean amplitudes g_k preserved to {max(r['gmax'] for r in rows):.3e}; "
        f"row-mean of every gap function preserved to {max(r['meandiff'] for r in rows):.3e}")
    return rows


def _fwd(E, W, tk):
    H = np.ones((tk.R, E.shape[1]))
    for j in range(tk.n):
        H = H * E[j][:, tk.idx[:, j]].T
    return H, H @ W.T


def _ratio(E, W, tk, q, tgt):
    return exact_margin(E, W, tk) / penalty(E, W, q)[0] ** ((tk.n + 1) / q) / tgt


def block_S(rng, cells, nsamp, log):
    rows = []
    for (p, n, q, m) in cells:
        nu = n + 1
        Gs, S, b = gamma_prog(p, nu / q)
        tgt = (n + 1) ** (-(nu - q) / q) * 2 * (2 * p * (n + 1)) ** (-nu / 2) * Gs
        tk = Task(p, n)
        gains = []
        best = (-np.inf, None)
        kinds = ["rand", "cos1", "impure", "clustered", "junk"]
        for x in make_starts(rng, tk, m, nsamp, kinds):
            nE = n * m * p
            E = x[:nE].reshape(n, m, p)
            W = x[nE:].reshape(p, m)
            r1 = _ratio(E, W, tk, q, tgt)
            E2, W2 = purify(E, W, p, n)
            r2 = _ratio(E2, W2, tk, q, tgt)
            gains.append(r1 - r2)
            if r1 - r2 > best[0]:
                best = (r1 - r2, (r1, r2))
        g = np.array(gains)
        rows.append(dict(p=p, n=n, q=q, m=m, nsamp=nsamp,
                         frac_impurity_helps=float((g > 1e-12).mean()),
                         max_gain=float(g.max()), median_gain=float(np.median(g)),
                         best_pair=best[1]))
        log(f"[S] p={p} n={n} q={q:g} m={m}: over {nsamp} nets, "
            f"ratio(impure) - ratio(purified): median {np.median(g):+.4e}, "
            f"max {g.max():+.4e}, helps in {(g > 1e-12).mean() * 100:.1f}% of cases")
    return rows


def block_O(rng, cells, starts, log):
    """Maximise the impurity GAIN itself: soft-min ascent on the impure net, then compare."""
    betas = [8.0, 25.0, 80.0, 250.0, 800.0, 2500.0, 8000.0]
    rows = []
    for (p, n, q, m) in cells:
        nu = n + 1
        Gs, S, b = gamma_prog(p, nu / q)
        tgt = (n + 1) ** (-(nu - q) / q) * 2 * (2 * p * (n + 1)) ** (-nu / 2) * Gs
        tk = Task(p, n)
        bestgain, bestr1 = -np.inf, -np.inf
        t0 = time.time()
        for x0 in make_starts(rng, tk, m, starts, ["impure", "rand", "clustered"]):
            x = ascend(x0, tk, m, q, tgt, betas)
            nE = n * m * p
            E = x[:nE].reshape(n, m, p)
            W = x[nE:].reshape(p, m)
            r1 = _ratio(E, W, tk, q, tgt)
            E2, W2 = purify(E, W, p, n)
            r2 = _ratio(E2, W2, tk, q, tgt)
            bestgain = max(bestgain, r1 - r2)
            bestr1 = max(bestr1, r1)
        rows.append(dict(p=p, n=n, q=q, m=m, best_gain=bestgain, best_ratio=bestr1,
                         secs=time.time() - t0))
        log(f"[O] p={p} n={n} q={q:g} m={m}: best ratio {bestr1:.12f}; "
            f"best ratio(impure) - ratio(purified) = {bestgain:+.6e}  [{time.time()-t0:.0f}s]")
    return rows


def block_F(rng, cells, starts, log):
    """FORCED impurity: neuron i is w_i1 * (freq k1 part) + w_i2 * (freq k2 part) with both
    amplitudes pinned to comparable size; maximise the ratio within that class."""
    from scipy.optimize import minimize
    rows = []
    for (p, n, q, m) in cells:
        nu = n + 1
        Gs, S, b = gamma_prog(p, nu / q)
        tgt = (n + 1) ** (-(nu - q) / q) * 2 * (2 * p * (n + 1)) ** (-nu / 2) * Gs
        tk = Task(p, n)
        nf = (p - 1) // 2
        a = np.arange(p) * 2 * np.pi / p

        def build(z, ks):
            """z = per neuron: [amp1, amp2, n+1 phases for k1, n+1 phases for k2]."""
            E = np.zeros((n, m, p))
            W = np.zeros((p, m))
            w = z.reshape(m, 2 * (n + 1) + 2)
            for i in range(m):
                A1, A2 = w[i, 0], w[i, 1]
                ph1 = w[i, 2:3 + n]
                ph2 = w[i, 3 + n:4 + 2 * n]
                k1, k2 = ks[i]
                for j in range(n):
                    E[j, i] = A1 * np.cos(k1 * a + ph1[j]) + A2 * np.cos(k2 * a + ph2[j])
                W[:, i] = A1 * np.cos(k1 * a + ph1[n]) + A2 * np.cos(k2 * a + ph2[n])
            return E, W

        best = -np.inf
        for s in range(starts):
            ks = [tuple(rng.choice(np.arange(1, nf + 1), 2, replace=False)) for _ in range(m)]
            z0 = np.concatenate([np.concatenate([[1.0, rng.uniform(0.4, 1.0)],
                                                 rng.uniform(0, 2 * np.pi, 2 * (n + 1))])
                                 for _ in range(m)])

            def neg(z):
                E, W = build(z, ks)
                if (np.abs(z.reshape(m, -1)[:, :2]) < 1e-6).all():
                    return 0.0
                return -_ratio(E, W, tk, q, tgt)

            res = minimize(neg, z0, method="Powell",
                           options={"maxiter": 20000, "xtol": 1e-10, "ftol": 1e-12})
            best = max(best, -res.fun)
        rows.append(dict(p=p, n=n, q=q, m=m, best_ratio=best, starts=starts))
        log(f"[F] p={p} n={n} q={q:g} m={m}: best ratio over FORCED two-frequency "
            f"neurons = {best:.12f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="VSOF")
    ap.add_argument("--nsamp", type=int, default=400)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"impure{a.tag}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    rng = np.random.default_rng(a.seed)
    cells = [(11, 2, 2.0, 9), (11, 2, 2.9, 12), (13, 2, 2.0, 12), (7, 2, 2.0, 9),
             (7, 3, 2.0, 8), (5, 3, 3.0, 8)]
    out = {}
    if "V" in a.blocks:
        out["V"] = block_V(rng, log)
    if "S" in a.blocks:
        out["S"] = block_S(rng, cells, a.nsamp, log)
    if "O" in a.blocks:
        out["O"] = block_O(rng, cells[:4], a.starts, log)
    if "F" in a.blocks:
        out["F"] = block_F(rng, [(11, 2, 2.0, 6), (11, 2, 2.0, 9), (7, 2, 2.0, 6),
                                 (11, 2, 2.9, 10)], max(6, a.starts // 2), log)
    (OUT / f"impure{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {OUT / f'impure{a.tag}.json'}")


if __name__ == "__main__":
    main()
