"""coset_construct.py -- an explicit junk-carrying construction on D_n that beats the symmetric-class ceiling (Section 9).

D_n elements x = (k, e) (rotation index k, reflection bit e), product (k1,e1)(k2,e2) = (k1 + (-1)^e1 k2, e1 + e2). With
theta_x = 2 pi j k / n and s_x = (-1)^e one has theta_{xy} = theta_x + s_x theta_y EXACTLY, so for n operands
    theta_{P(a)} = theta_{a1} + s_{a1} theta_{a2} + s_{a1} s_{a2} theta_{a3} + ...
The coset feature at frequency j is  f_j(a, c) = cos(theta_{P(a)} - theta_c), expanded with s^2 = 1 into separable products:
    nu = 3: 4 atoms;  nu = 4: 8 atoms  (the symmetric schoolbook needs 2^nu = 8 / 16).
On sign-consistent triples (s_c = s_{P(a)}) f_j = chi_{rho_j}(P(a) c^{-1}) / 2 (a clean class function); on inconsistent triples it
is junk, which a sign neuron s_{a1} ... s_{an} s_c (one atom) must dominate. Amplitudes (alpha_j, beta) are optimised for the exact
full-table gamma_2 (Nelder-Mead on logs). Output: gamma / symmetric ceiling, compared with the unrestricted ascent optima.
"""
import itertools, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402
from junk_nu4 import ceil  # noqa: E402
from verify_fourier import row_products  # noqa: E402


def atoms(Gn, n, j):
    """list of (coef sign, [f_1..f_n, g]) separable products whose sum is cos(theta_P - theta_c)."""
    N = 2 * Gn; k = np.arange(N) % Gn; e = np.arange(N) // Gn
    th = 2 * np.pi * j * k / Gn; s = (-1.0) ** e; C, S = np.cos(th), np.sin(th)
    # represent exp(i theta_P) as a sum of terms coef * prod_j (function of a_j), complex coef in {1, i, -1, -i}
    terms = [(1 + 0j, [C + 0j], [0]), (1j, [S + 0j], [1])]            # e^{i th_a1} = cos + i sin ; track a "parity carry" list
    # generic expansion: e^{i(th1 + s1 th2 + s1 s2 th3 ...)}: multiply successively by (cos th_j + i S_prev sin th_j), S_prev = prod of earlier s
    terms = [(1 + 0j, [np.exp(1j * th)])]
    for jj in range(1, n):
        new = []
        for coef, fs in terms:
            # e^{i s_prev th} = cos th + i s_prev sin th, s_prev = s_{a1}...s_{a_jj}; distribute s_prev onto the earlier factors
            new.append((coef, fs + [C + 0j]))
            new.append((coef * 1j, [f * s for f in fs] + [S + 0j]))
        terms = new
    # multiply by e^{-i th_c} and take the real part: Re(coef prod F_j e^{-i th_c}); split every complex factor into real atoms
    out = []
    for coef, fs in terms:
        # expand prod over complex functions into real atoms: each factor f = Re f + i Im f
        parts = [[(1.0, np.real(f)), (1j, np.imag(f))] for f in fs] + [[(1.0, C), (-1j, S)]]
        for combo in itertools.product(*parts):
            ph = coef * np.prod([p for p, _ in combo])
            if abs(ph.imag) > 1e-12 or abs(ph.real) < 1e-12:
                continue
            funcs = [g for _, g in combo]
            if any(np.allclose(g, 0) for g in funcs):
                continue
            out.append((ph.real, funcs))
    return out


def build(Gn, n, alphas, beta):
    N = 2 * Gn; e = np.arange(N) // Gn; s = (-1.0) ** e
    Es, Ws = [], []
    for j, al in alphas.items():
        for c, funcs in atoms(Gn, n, j):
            norms = [np.linalg.norm(f) for f in funcs]; mu = abs(c * al) * np.prod(norms)
            t = mu ** (1 / (n + 1))
            fs = [t * f / nm for f, nm in zip(funcs, norms)]; fs[0] = fs[0] * np.sign(c * al)
            Es.append(fs[:-1]); Ws.append(fs[-1])
    if beta > 0:
        t = (beta * N ** ((n + 1) / 2)) ** (1 / (n + 1)); u = s / np.sqrt(N)
        Es.append([t * u] * n); Ws.append(t * u)
    return np.array(Es).transpose(1, 0, 2), np.array(Ws)


def gamma(G, n, E, W, y):
    m = W.shape[0]; h = np.ones((m, 1))
    for jj in range(n):
        h = (h[:, :, None] * E[jj][:, None, :]).reshape(m, -1)
    Z = h.T @ W; zz = Z.copy(); zy = zz[np.arange(len(y)), y]; zz[np.arange(len(y)), y] = -np.inf
    return float((zy - zz.max(1)).min()) / float((E ** 2).sum() + (W ** 2).sum()) ** ((n + 1) / 2)


def main():
    lines = []
    for Gn, n, js in [(3, 2, [1]), (7, 2, [1, 2, 3]), (3, 3, [1]), (5, 2, [1, 2]), (7, 3, [1, 2, 3])]:
        G = gc.get(f"d{Gn}"); y = row_products(G, n); ce = ceil(G, n + 1, 2.0); gs = ce["school"][0]
        # check the identity: sum of atoms = cos(theta_P - theta_c)
        k = np.arange(2 * Gn) % Gn; th = 2 * np.pi * js[0] * k / Gn
        rows = list(itertools.product(range(2 * Gn), repeat=n)); P = y
        E1, W1 = build(Gn, n, {js[0]: 1.0}, 0.0)
        h = np.ones((W1.shape[0], 1))
        for jj in range(n):
            h = (h[:, :, None] * E1[jj][:, None, :]).reshape(W1.shape[0], -1)
        Z = h.T @ W1
        # atoms were normalised; recover the sum with unit coefficient -> compare shape up to scale
        target = np.cos(th[P][:, None] - th[None, :])
        sc = (Z * target).sum() / (target * target).sum(); err = np.abs(Z - sc * target).max() / np.abs(sc)
        natoms = len(atoms(Gn, n, js[0]))
        def neg(x):
            al = {j: np.exp(v) for j, v in zip(js, x[:-1])}; E, W = build(Gn, n, al, np.exp(x[-1]))
            return -gamma(G, n, E, W, y)
        best = None
        for st in range(6):
            x0 = np.random.default_rng(st).normal(0, 0.5, len(js) + 1)
            r = minimize(neg, x0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-7, fatol=1e-14))
            if best is None or r.fun < best.fun: best = r
        g = -best.fun
        amps = np.exp(best.x)
        lines.append(f"D_{Gn} n={n} (nu {n + 1}) q=2: coset atoms per frequency {natoms} (identity err {err:.1e}); optimal gamma {g:.6e} = {g / gs:.5f} x symmetric ceiling "
                     f"(G1: {ce['unres'][0] / gs:.3f} x); amplitudes alpha {np.round(amps[:-1] / amps[:-1].max(), 4).tolist()} beta/alpha_max {amps[-1] / amps[:-1].max():.4f}")
        print(lines[-1], flush=True)
    (HERE / "out" / "coset_construct.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
