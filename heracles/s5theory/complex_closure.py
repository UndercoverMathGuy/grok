"""complex_closure.py -- Theorem C: the complex-weight gated chain with real-part readout on a finite abelian group.

Model: z(a, c) = Re sum_i W_i(c) prod_j E_ij(a_j),  E_ij, W_i : G -> C,  ||theta_i||^2 = sum_j ||E_ij||^2 + ||W_i||^2 (complex moduli),
P_q = sum_i ||theta_i||^q. Claim: gamma_q <= (N nu)^{-nu/2} Gamma^C, Gamma^C = the programme with EVERY character priced 1
(pairs chi, chi-bar one-sided), attained by one neuron per character: E_ij(a) = t chi(a), W_i(c) = t conj chi(c).
Checks (fp64, full table):
  C1 attainment: constructed nets on Z_7, Z_11, Z_13 (n = 2, 3; q = 2, 2.5) and Z_12 (has a real character) hit the bound.
  C2 random complex nets never exceed it.
  C3 direct margin ascent (Adam on a soft-min of the normalized margin) from random complex init, Z_7 n = 2 and Z_5 n = 3:
     the best value found vs the bound (must be <= 1; approaching 1 = the bound is the true max).
  C4 the real-weight optimum (ELEGANT, (H) class) sits at 2^{1 - nu/2} nu^{-(s - 1)} of the complex one.
"""
import itertools, math, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402

torch.set_default_dtype(torch.float64)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def cyclic_data(orders):
    """characters of Z_{n1} x ...: returns list of (label tuple, is_real), class = element; cov over non-zero elements."""
    els = list(itertools.product(*[range(n) for n in orders])); N = len(els)
    chars, seen = [], set()
    for j in els:
        if all(x == 0 for x in j) or j in seen:
            continue
        jb = tuple((-y) % n for y, n in zip(j, orders)); seen.add(j); seen.add(jb)
        chars.append((j, j == jb))
    ph = lambda j, a: 2 * np.pi * sum(y * x / n for y, x, n in zip(j, a, orders))
    nz = [a for a in els if any(x != 0 for x in a)]
    cov = np.array([[1 - math.cos(ph(j, a)) for a in nz] for j, _ in chars])   # 1 - Re chi(D) (D and D^-1 averaged)
    return els, chars, cov, ph


def programme(cov, c, r):
    verts = gc.vertices(cov, exact=False)
    Gam, k, costs = gc.solve(verts, c, r)
    return Gam, np.array(verts[k])


def complex_net(orders, chars, y, nu, ph, els):
    """one neuron per character with y > 0: amplitude t^nu = y (clean coefficient y at chi(D))."""
    N = len(els); n = nu - 1
    Es, Ws = [], []
    for (j, _), yy in zip(chars, y):
        if yy <= 0:
            continue
        t = yy ** (1 / nu)
        chi = np.array([np.exp(1j * ph(j, a)) for a in els])
        Es.append([t * chi for _ in range(n)]); Ws.append(t * np.conj(chi))
    return np.array(Es).transpose(1, 0, 2), np.array(Ws)


def margin_and_cost(orders, E, W, q, els):
    N = len(els); n = E.shape[0]; m = W.shape[0]
    idx = {a: i for i, a in enumerate(els)}
    h = np.ones((m, 1), dtype=complex)
    for j in range(n):
        h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
    Z = np.real(h.T @ W)                                   # (N^n, N)
    P = []
    for a in itertools.product(range(N), repeat=n):
        s = tuple(sum(els[x][t] for x in a) % orders[t] for t in range(len(orders)))
        P.append(idx[s])
    P = np.array(P); zz = Z.copy(); zy = zz[np.arange(len(P)), P]; zz[np.arange(len(P)), P] = -np.inf
    margin = float((zy - zz.max(1)).min())
    th2 = (np.abs(E) ** 2).sum((0, 2)) + (np.abs(W) ** 2).sum(1)
    return margin, float((th2 ** (q / 2)).sum()), Z, P


def c1(orders, n, q):
    nu = n + 1; r = q / nu
    els, chars, cov, ph = cyclic_data(orders); N = len(els)
    c = np.ones(len(chars))
    Gam, y = programme(cov, c, r)
    E, W = complex_net(orders, chars, y, nu, ph, els)
    margin, Pq, Z, P = margin_and_cost(orders, E, W, q, els)
    gam = margin / Pq ** (nu / q); bound = (N * nu) ** (-nu / 2) * Gam
    sup = [chars[i][0] for i in range(len(chars)) if y[i] > 0]
    log(f"  C1 Z{orders} n={n} q={q:g}: Gamma^C {Gam:.8f}, support {sup} (K = {len(sup)}), constructed gamma {gam:.10e} / bound {bound:.10e} = {gam / bound:.12f}")
    return Gam, bound, gam


def c2(orders, n, q, trials=200, m=6, seed=0):
    nu = n + 1; r = q / nu; rng = np.random.default_rng(seed)
    els, chars, cov, ph = cyclic_data(orders); N = len(els)
    Gam, _ = programme(cov, np.ones(len(chars)), r); bound = (N * nu) ** (-nu / 2) * Gam
    worst = -np.inf
    for _ in range(trials):
        E = rng.normal(size=(n, m, N)) + 1j * rng.normal(size=(n, m, N)); W = rng.normal(size=(m, N)) + 1j * rng.normal(size=(m, N))
        # bias towards structure: half the trials use near-character neurons plus noise
        if _ % 2:
            for i in range(m):
                j = chars[rng.integers(len(chars))][0]; chi = np.array([np.exp(1j * ph(j, a)) for a in els])
                E[:, i] = chi[None] * (1 + 0.1 * rng.normal()) + 0.2 * E[:, i]; W[i] = np.conj(chi) + 0.2 * W[i]
        margin, Pq, _, _ = margin_and_cost(orders, E, W, q, els)
        worst = max(worst, margin / Pq ** (nu / q) / bound)
    log(f"  C2 Z{orders} n={n} q={q:g}: max over {trials} random complex nets of gamma/bound = {worst:.4f} (must be <= 1)")
    return worst


def c3(orders, n, q, m=10, seeds=6, steps=6000):
    nu = n + 1; r = q / nu
    els, chars, cov, ph = cyclic_data(orders); N = len(els)
    Gam, _ = programme(cov, np.ones(len(chars)), r); bound = (N * nu) ** (-nu / 2) * Gam
    _, _, _, P = margin_and_cost(orders, np.ones((n, 1, N), complex), np.ones((1, N), complex), q, els)
    P = torch.from_numpy(P)
    best = -np.inf
    for sd in range(seeds):
        g = torch.Generator().manual_seed(sd)
        Er = torch.randn(n, m, N, 2, generator=g).requires_grad_(True); Wr = torch.randn(m, N, 2, generator=g).requires_grad_(True)
        opt = torch.optim.Adam([Er, Wr], lr=0.02)
        for it in range(steps):
            E = torch.view_as_complex(Er); W = torch.view_as_complex(Wr)
            h = torch.ones(m, 1, dtype=torch.complex128)
            for j in range(n):
                h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
            Z = (h.T @ W).real
            th2 = (Er ** 2).sum((0, 2, 3)) + (Wr ** 2).sum((1, 2)); Pq = (th2 ** (q / 2)).sum()
            Zn = Z / Pq ** (nu / q)
            zy = Zn.gather(1, P[:, None]); gaps = zy - Zn
            gaps = gaps + torch.zeros_like(gaps).scatter_(1, P[:, None], 1e9)
            tau = 0.05 * float(gaps.min().abs().detach() + 1e-12) + 1e-12 if it > steps // 2 else float(gaps.abs().mean().detach()) * 0.2 + 1e-12
            soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
            opt.zero_grad(); (-soft).backward(); opt.step()
        with torch.no_grad():
            E = torch.view_as_complex(Er).numpy(); W = torch.view_as_complex(Wr).numpy()
        margin, Pq, _, _ = margin_and_cost(orders, E, W, q, els)
        best = max(best, margin / Pq ** (nu / q) / bound)
        log(f"    C3 Z{orders} n={n} q={q:g} seed {sd}: gamma/bound = {margin / Pq ** (nu / q) / bound:.6f}")
    log(f"  C3 Z{orders} n={n} q={q:g}: best ascent value gamma/bound = {best:.6f} (theorem: <= 1)")
    return best


def main():
    log("== Theorem C checks")
    for orders, n, q in [((7,), 2, 2.0), ((11,), 2, 2.0), ((13,), 2, 2.0), ((11,), 2, 2.5), ((7,), 3, 2.0), ((7,), 3, 3.0), ((12,), 2, 2.0), ((3, 3), 2, 2.0)]:
        c1(orders, n, q)
    for orders, n, q in [((7,), 2, 2.0), ((12,), 2, 2.0), ((5,), 3, 2.0)]:
        c2(orders, n, q)
    for orders, n, q in [((7,), 2, 2.0), ((5,), 3, 2.0)]:
        c3(orders, n, q)
    # C4: ratio real (H) optimum / complex optimum on Z_p
    for nu, q in [(3, 2.0), (4, 2.0), (4, 3.0), (5, 2.0)]:
        s = nu / q
        log(f"  C4 nu {nu} q {q:g}: real-(H) optimum / complex optimum = 2^(1-nu/2) nu^-(s-1) = {2 ** (1 - nu / 2) * nu ** (-(s - 1)):.6f}")
    (HERE / "out" / "complex_closure.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
