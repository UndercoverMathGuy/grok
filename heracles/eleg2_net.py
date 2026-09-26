"""eleg2_net.py -- the NETWORK corollary of Theorem A, verified against brute force.

A pure single-frequency (K = 1) chain network at frequency k on Z_p^n:

    e_lj(a) = r_lj cos(2 pi k a / p + alpha_lj) ,   w_l(c) = rho_l cos(2 pi k c / p + beta_l)
    z(a_1..a_n, c) = sum_l w_l(c) prod_j e_lj(a_j)

Atom dictionary (P3A SS2.1):  c_l = (prod_j r_lj) rho_l e^{i psi_l},  psi_l = sum_j alpha_lj - beta_l,
                              v_lj = e^{-2 i alpha_lj}.

Checks:
 (N1) the profile identity: 2^n * margin == min over a, D!=0 of Re[W_kD h(u(a))],  W_kD = 1 - omega^{-kD}
 (N2) the aiming step: 2^n * margin <= M_1 * m_A  with  M_1 = 1 - cos(2 pi/p), A = cos(pi/p),
      m_A = min over |t_j| = A of Re h(t)                       (this is the step P3C SS3.2 licenses)
 (N3) the budget: P_q >= ((n+1) p/2)^{q/2} sum_l |c_l|^r  after per-neuron rebalancing
 (N4) THE COROLLARY: gamma_q <= (1 - eps_p)^{-1} (n+1)^{-(nu-q)/q} B Gamma, and separately the
      unrelaxed gamma_q / target, which must stay <= 1.

Run: uv run --with scipy python eleg2_net.py --p 11 13 17 --n 2 --trials 60
"""
import argparse
import json
import os
import sys
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import min_re_h, h_at

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)


def build(p, k, n, r_arr, alpha, rho, beta):
    """Return the embeddings/readout arrays of the pure network."""
    a = np.arange(p)
    N = len(rho)
    E = np.zeros((N, n, p))
    for l in range(N):
        for j in range(n):
            E[l, j] = r_arr[l, j] * np.cos(2 * np.pi * k * a / p + alpha[l, j])
    W = np.zeros((N, p))
    for l in range(N):
        W[l] = rho[l] * np.cos(2 * np.pi * k * a / p + beta[l])
    return E, W


def margin_brute(E, W, p, n):
    """min over all p^n rows and wrong classes of (z_correct - z_wrong)."""
    N = E.shape[0]
    best = np.inf
    for row in product(range(p), repeat=n):
        hid = np.prod(np.array([[E[l, j, row[j]] for j in range(n)] for l in range(N)]),
                      axis=1)                                  # (N,)
        z = hid @ W                                            # (p,)
        s = sum(row) % p
        gap = z[s] - np.max(np.delete(z, s))
        best = min(best, gap)
    return best


def atoms(r_arr, alpha, rho, beta):
    N, n = r_arr.shape
    c = np.prod(r_arr, axis=1) * rho * np.exp(1j * (alpha.sum(axis=1) - beta))
    v = np.exp(-2j * alpha)
    return c, v


def profile_margin(c, v, p, k, n):
    """min over grid rows a and D != 0 of Re[(1 - omega^{-kD}) h(u(a))], divided by 2^n."""
    om = np.exp(2j * np.pi / p)
    best = np.inf
    Ws = [1 - om ** (-(k * D) % p) for D in range(1, p)]
    for row in product(range(p), repeat=n):
        u = np.array([om ** ((-2 * k * row[j]) % p) for j in range(n)])
        hv = h_at(c, v, u)
        for W in Ws:
            best = min(best, (W * hv).real)
    return best / 2 ** n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0, 2.5])
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    n = a.n
    nu = n + 1
    rng = np.random.default_rng(3)
    rows = []
    w = dict(N1=0.0, N2=-np.inf, N4=-np.inf, N4relaxed=-np.inf)
    for p in a.p:
        M1 = 1 - np.cos(2 * np.pi / p)
        Adep = np.cos(np.pi / p)
        for q in a.qs:
            r = q / nu
            kappa0 = nu ** ((nu - q) / q)
            eps_p = kappa0 * (2 ** n - (1 + Adep) ** n)
            B = 2 * (2 * p * nu) ** (-nu / 2)
            target = nu ** (-(nu - q) / q) * B * M1
            for t in range(a.trials):
                k = int(rng.integers(1, p))
                N = int(rng.integers(1, nu + 1))            # at most n+1 neurons
                if t % 4 == 0:
                    N = nu                                   # perturbed block
                    alpha = np.tile((np.pi * np.arange(N) / nu)[:, None], (1, n))
                    beta = n * np.pi * np.arange(N) / nu
                    alpha = alpha + 0.25 * rng.normal(size=alpha.shape) * (t > 0)
                    beta = beta + 0.25 * rng.normal(size=beta.shape) * (t > 0)
                    r_arr = np.ones((N, n)) + 0.2 * rng.normal(size=(N, n)) * (t > 0)
                    rho = np.ones(N) + 0.2 * rng.normal(size=N) * (t > 0)
                else:
                    alpha = 2 * np.pi * rng.random((N, n))
                    beta = 2 * np.pi * rng.random(N)
                    r_arr = np.exp(0.5 * rng.normal(size=(N, n)))
                    rho = np.exp(0.5 * rng.normal(size=N))
                E, W = build(p, k, n, r_arr, alpha, rho, beta)
                mg = margin_brute(E, W, p, n)
                c, v = atoms(r_arr, alpha, rho, beta)
                # (N1) profile identity
                pm = profile_margin(c, v, p, k, n)
                w["N1"] = max(w["N1"], abs(mg - pm) / max(abs(mg), 1e-12))
                if mg <= 0:
                    continue
                # (N2) the aiming step
                mA, _ = min_re_h(c, v, Adep, starts=24, rng=rng)
                w["N2"] = max(w["N2"], (2 ** n * mg) / (M1 * mA) if mA > 0 else -np.inf)
                # (N3)+(N4): cost and the target
                Pq = np.sum((np.sum(r_arr ** 2, axis=1) * (p / 2) + rho ** 2 * (p / 2)) ** (q / 2))
                cost = np.sum(np.abs(c) ** r)
                bud = ((nu * p / 2) ** (q / 2)) * cost
                gam = mg / Pq ** (nu / q)
                w["N4"] = max(w["N4"], gam / target)
                w["N4relaxed"] = max(w["N4relaxed"], gam / (target / (1 - eps_p)))
                rows.append(dict(p=p, q=q, k=k, N=N, margin=float(mg), gamma=float(gam),
                                 ratio=float(gam / target), eps_p=float(eps_p),
                                 budget_slack=float(Pq / bud)))
            print(f"p={p} q={q}: eps_p={eps_p:.4f}  worst gamma/target so far = {w['N4']:.10f}")
    print("\nWORST over everything")
    print(f"  (N1) profile identity rel. error      : {w['N1']:.3e}")
    print(f"  (N2) 2^n*margin / (M_1 m_A)   (<= 1)  : {w['N2']:.10f}")
    print(f"  (N4) gamma_q / target         (<= 1)  : {w['N4']:.10f}")
    print(f"  (N4) gamma_q / relaxed target (<= 1)  : {w['N4relaxed']:.10f}")
    with open(os.path.join(OUT, f"net_n{n}.json"), "w") as f:
        json.dump(dict(worst={k: float(x) for k, x in w.items()}, rows=rows), f, indent=1)
