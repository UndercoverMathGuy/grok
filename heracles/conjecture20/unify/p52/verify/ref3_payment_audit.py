"""Referee-3: independent audit of the rigorous payment F (pay_adapt.payment) and of LB_Smin.

(a) LB_Smin(Pbar, qlow) <= S_{2/3}(P, q) for random feasible (P,q) with P <= Pbar, q >= qlow, where S is computed from the
    eigenvalues of an explicit random zero-diagonal-plus-identity PSD 3x3 matrix (no cubic formulas).
(b) Random and adversarial atom systems (beta_i, xi_i, eta_i): own data A,B,C,D, Q = sum |beta_i|^{2/3}, P_1 = 3^{-1/2} Q^{3/2}.
    Normalise |A| = 1, put the data in a small box (and in a point box), call payment(); require F <= P_1 (s=3/2) and also
    F <= P_1(s) at s = 1, 1.1, 1.25 (the s-monotonicity step R8).
    Adversarial families: single atom, two aligned atoms, IF blocks (B=C=D=0), near-IF, atoms with all phases equal,
    many tiny atoms, systems minimising P_1 - F by local search (Nelder-Mead on atom parameters).
"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
from pay_rig import LB_Smin
from pay_adapt import payment
from scipy.optimize import minimize

rng = np.random.default_rng(777)
viol = 0

# ---------------- (a) LB_Smin brute force
t0 = time.time()
worst_a = np.inf
for it in range(20000):
    z = rng.normal(size=3) + 1j * rng.normal(size=3)
    z *= rng.uniform(0, 1.2) / max(np.abs(z).max(), 1e-12)
    H = np.eye(3, dtype=complex)
    H[1, 0] = z[0]; H[2, 1] = z[1]; H[2, 0] = z[2]
    H[0, 1] = np.conj(z[0]); H[1, 2] = np.conj(z[1]); H[0, 2] = np.conj(z[2])
    ev = np.linalg.eigvalsh(H)
    if ev.min() < 0:
        continue
    P = np.sum(np.abs(z) ** 2)
    Z = H - np.eye(3)
    q = np.real(np.linalg.det(Z))
    S = np.sum(np.maximum(ev, 0) ** (2 / 3))
    Pbar = P + rng.exponential(0.05) * (rng.random() < 0.7)
    qlow = q - rng.exponential(0.05) * (rng.random() < 0.7)
    lb = LB_Smin(np.array([Pbar]), np.array([qlow]))[0]
    worst_a = min(worst_a, S - lb)
    if lb > S + 1e-12:
        viol += 1; print("LB_Smin violation", P, q, Pbar, qlow, S, lb)
print(f"(a) LB_Smin: min(S - LB) = {worst_a:.3e}  violations={viol}  ({time.time()-t0:.0f}s)")


# ---------------- (b) atom systems
def own(beta, xi, eta):
    A = beta.sum(); B = (beta * xi).sum(); C = (beta * eta).sum(); D = (beta * xi * eta).sum()
    return A, B, C, D


def P1(beta, s):
    Q = np.sum(np.abs(beta) ** (1 / s))
    return 3 ** (1 - s) * Q ** s


def normalised(beta, xi, eta):
    A, B, C, D = own(beta, xi, eta)
    sc = abs(A)
    m = np.array([abs(B), abs(C), abs(D)]) / sc
    phi = np.angle(A * D * np.conj(B * C)) if abs(B) * abs(C) * abs(D) > 0 else 0.0
    return m, phi, sc


def check(beta, xi, eta, halfw=0.0, tag=""):
    global viol
    m, phi, sc = normalised(beta, xi, eta)
    if np.any(m > 1.4):
        return None
    lo = np.maximum(m - halfw * rng.random(3), 0.0); hi = m + halfw * rng.random(3)
    plo = phi - halfw * rng.random(); phi_hi = phi + halfw * rng.random()
    F = payment(lo, hi, max(plo, -np.pi), min(phi_hi, np.pi))
    out = []
    for s in (1.5, 1.25, 1.1, 1.0):
        P = P1(beta, s) / sc
        out.append(P - F)
        if F > P * (1 + 1e-12) + 1e-14:
            viol += 1; print("PAYMENT VIOLATION", tag, s, m, phi, F, P)
    return min(out), out[0], F


results = []
t0 = time.time()
# single atom
for k in range(20):
    b = np.array([rng.normal() + 1j * rng.normal()]); x = np.exp(1j * rng.uniform(0, 2 * np.pi, 1)); e = np.exp(1j * rng.uniform(0, 2 * np.pi, 1))
    r = check(b, x, e, 0.0, "single")
    results.append(("single", r))
# two aligned atoms, IF blocks, near-IF
for k in range(30):
    b = np.array([1.0, rng.uniform(0, 1)]) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    x = np.exp(1j * rng.uniform(0, 2 * np.pi, 2)); e = np.exp(1j * rng.uniform(0, 2 * np.pi, 2))
    results.append(("two", check(b, x, e, 0.0, "two")))
for k in range(20):
    # IF block: 4 equal atoms with (xi, eta) in {+-1}^2 -> B=C=D=0
    ph = rng.uniform(0, 2 * np.pi)
    b = np.full(4, np.exp(1j * ph)); x = np.array([1, -1, 1, -1.0]); e = np.array([1, 1, -1, -1.0])
    b = b * (1 + 0.3 * rng.normal(size=4) * (k > 5))
    x = x * np.exp(1j * 0.2 * rng.normal(size=4) * (k > 10)); e = e * np.exp(1j * 0.2 * rng.normal(size=4) * (k > 10))
    results.append(("IF/nearIF", check(b, x, e, 0.0, "IF")))
for k in range(100):
    K = rng.integers(1, 9)
    b = (rng.normal(size=K) + 1j * rng.normal(size=K)) * rng.exponential(1, K)
    x = np.exp(1j * rng.uniform(0, 2 * np.pi, K)); e = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
    results.append(("random", check(b, x, e, 0.02 * (k % 2), "rand")))
print(f"(b) random/adversarial atom systems done ({time.time()-t0:.0f}s), violations={viol}")
for tag in ("single", "two", "IF/nearIF", "random"):
    v = [r for t, r in results if t == tag and r is not None]
    if v:
        print(f"    {tag:10s}: n={len(v)}  min_s(P_1(s)-F) = {min(x[0] for x in v):.4e}   min(P_1(3/2)-F) = {min(x[1] for x in v):.4e}")

# (c) adversarial: minimise P_1 - F over atom parameters (K = 2, 3), F on point boxes
def obj(z, K):
    b = z[:K] * np.exp(1j * z[K:2 * K]); x = np.exp(1j * z[2 * K:3 * K]); e = np.exp(1j * z[3 * K:4 * K])
    m, phi, sc = normalised(b, x, e)
    if sc < 1e-9 or np.any(m > 1.4):
        return 1.0
    F = payment(m, m, phi, phi)
    return P1(b, 1.5) / sc - F


t0 = time.time()
best = np.inf
for trial in range(6):
    K = 2 + trial % 3
    z0 = np.concatenate([rng.uniform(0.2, 1, K), rng.uniform(0, 2 * np.pi, 3 * K)])
    res = minimize(obj, z0, args=(K,), method='Nelder-Mead', options=dict(maxiter=120, xatol=1e-6, fatol=1e-9))
    best = min(best, res.fun)
    if res.fun < -1e-12:
        viol += 1; print("ADVERSARIAL PAYMENT VIOLATION", K, res.fun, res.x)
print(f"(c) adversarial min (P_1 - F) over 6 Nelder-Mead runs = {best:.4e} ({time.time()-t0:.0f}s)")
print("TOTAL VIOLATIONS =", viol)
