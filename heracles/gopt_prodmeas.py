"""gopt_prodmeas.py -- LEMMA P (product test measures): the new handle on junk at a MULTI-frequency support.

For ANY gated-chain network (no purity, no junk-free, any width), any D != 0 and any PRODUCT probability
measure w = w_1 x ... x w_n on Z_p^n (one factor per operand):

    margin  <=  E_{a ~ w} [ Psi_D(a) ]  =  sum_{k=1}^{nf} 2 (1 - cos(2 pi k D / p)) Re X_k(w),
    X_k(w)  =  sum_i  what_i(k) prod_{j=1}^n S_ij(k),     S_ij(k) = E_{w_j}[ e^{2 pi i k a_j / p} e_ij(a_j) ],

where Psi_D(a) = (1/2)[Phi_D(a) + Phi_{-D}(a)] and Phi_D(a) = z(a, s(a)) - z(a, s(a) - D).

Plain reading: the classical clean bound is the case w_j = uniform, where S_ij(k) = conj(ehat_ij(k)) and
X_k is the clean amplitude.  A NON-uniform w_j tilts the average towards particular operand values, and
the tilted "embedding transform" S_ij(k) feels the actual values of the embedding, not just its k-th
Fourier coefficient -- which is how junk becomes visible.  Because the operands are independent
coordinates, every frequency k gets its own evaluation point (what_1(2k), ..., what_n(2k)) inside the
unit disc; for a pure single-frequency population this is exactly evaluation of the polynomial h of
gopt_atom.py at an interior point of the polydisc, and the extreme points of the reachable set are the
p^n grid points.

This script checks the identity and the inequality on random nets (uniform, tilted, and mixture measures).
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent


def rand_net(p, n, m, g):
    E = torch.randn(n, m, p, generator=g)
    W = torch.randn(p, m, generator=g)
    return E, W


def table(p, n):
    idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
    return idx, idx.sum(1) % p


def logits(E, W, p, n, idx):
    h = torch.ones(idx.shape[0], E.shape[1])
    for j in range(n):
        h = h * E[j][:, idx[:, j]].T
    return h @ W.T


def main(a):
    g = torch.Generator().manual_seed(a.seed)
    rows = []
    for (p, n, m) in [(7, 2, 5), (11, 2, 4), (13, 2, 3), (7, 3, 4), (11, 3, 3), (5, 4, 3)]:
        idx, y = table(p, n)
        maxerr = 0.0; minslack = 1e18
        for trial in range(a.trials):
            E, W = rand_net(p, n, m, g)
            z = logits(E, W, p, n, idx)
            zz = z.scatter(1, y[:, None], -float("inf"))
            margin = float((z.gather(1, y[:, None]).squeeze(1) - zz.max(1).values).min())
            D = 1 + int(torch.randint(p - 1, (1,), generator=g))
            # Psi_D on every row
            cP = (y - D) % p; cM = (y + D) % p
            Psi = 0.5 * ((z.gather(1, y[:, None]) - z.gather(1, cP[:, None]))
                         + (z.gather(1, y[:, None]) - z.gather(1, cM[:, None]))).squeeze(1)
            # three product measures: uniform, tilted, mixture
            for kind in ("uniform", "tilt", "mix"):
                if kind == "uniform":
                    w = torch.full((n, p), 1.0 / p)
                elif kind == "tilt":
                    w = torch.rand(n, p, generator=g) ** 3; w = w / w.sum(1, keepdim=True)
                else:
                    b = torch.randint(p, (n,), generator=g)
                    w = torch.full((n, p), 0.3 / p)
                    w[torch.arange(n), b] += 0.7
                    w = w / w.sum(1, keepdim=True)
                wrow = torch.ones(idx.shape[0])
                for j in range(n):
                    wrow = wrow * w[j][idx[:, j]]
                lhs = float((wrow * Psi).sum())
                # Fourier side
                kk = torch.arange(1, (p - 1) // 2 + 1)
                ang = 2 * math.pi * torch.outer(kk, torch.arange(p)) / p
                chi = torch.polar(torch.ones(len(kk), p), ang)          # (nf, p)
                S = torch.einsum("kb,jib->jik", chi, E.to(torch.cdouble) * w[:, None, :].to(torch.cdouble))
                What = (W.T.to(torch.cdouble) @ chi.conj().T) / p        # (m, nf) = what_i(k)
                X = (What * S.prod(0)).sum(0)                            # (nf,)
                coef = 2 * (1 - torch.cos(2 * math.pi * kk * D / p))
                rhs = float((coef * X.real).sum())
                maxerr = max(maxerr, abs(lhs - rhs) / max(1.0, abs(lhs)))
                minslack = min(minslack, lhs - margin)
        rows.append(dict(p=p, n=n, m=m, max_identity_relerr=maxerr, min_slack_E_minus_margin=minslack))
        print(f"p={p} n={n} m={m}: identity rel err <= {maxerr:.3e} | "
              f"min (E_w[Psi_D] - margin) = {minslack:.6f}  (must be >= 0)", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / "prodmeas.json").write_text(json.dumps(rows, indent=1))
    print("wrote", d / "prodmeas.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
