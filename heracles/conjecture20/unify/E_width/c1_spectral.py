# Checks of the spectral (entropic) Lemma 7, all n:
#  (A) chain:   Q = sum_l |beta_l|^r >= nu^{-r} sum_k |lambda_k(H)|^r,  hence Q^s >= Lambda e^{(s-1)H_r(mu)} >= |A| e^{(s-1)H(mu)}
#  (B) twisted: Q^s >= L e^{(s-1) H_r(pi)},  pi = spec(K)/(nu L);  Schur-Horn: H_a(rho/L) >= H_a(pi)
#  (C) L2 form: |K_tt' - alpha^{d-1} beta_I| <= (d-1) sqrt(2L(L-|A|));  gain <= (|A|/L)(1 + 2 sum|K_off|^2/(nu L^2))^{s-1}
#  (D) IF systems: mu uniform (all eigenvalues = A), D = 0 (Lemma 7 recovered)
import numpy as np, itertools
from elib import *

rng = np.random.default_rng(1)
worst = {'A': -np.inf, 'A2': -np.inf, 'B': -np.inf, 'SH': -np.inf, 'C1': -np.inf, 'C2': -np.inf}


def interval_coeff(beta, xi, a, b):  # beta_{[a+1..b]} (slots a+1..b, 1-based) = sum beta_l prod xi
    return np.sum(beta * np.prod(xi[:, a:b], axis=1))


def check_system(beta, xi, s):
    n = xi.shape[1]; nu = n + 1; r = 1 / s
    rho = np.abs(beta); L = rho.sum(); A = beta.sum(); Q = (rho ** r).sum()
    lam = spec_chain(beta, xi)
    lam = np.where(np.abs(lam) < 1e-10 * np.abs(lam).max(), 0, lam)   # remove round-off eigenvalues of rank-deficient H
    lhs = Q; rhs = nu ** (-r) * np.sum(np.abs(lam) ** r)
    worst['A'] = max(worst['A'], rhs / lhs - 1)
    Lam = np.abs(lam).sum() / nu; mu = np.abs(lam) / np.abs(lam).sum()
    b2 = Lam * np.exp((s - 1) * renyi(mu, r))
    worst['A2'] = max(worst['A2'], b2 / Q ** s - 1, abs(A) * np.exp((s - 1) * shannon(mu)) / Q ** s - 1, abs(A) / Lam - 1)
    pis = spec_twisted(beta, xi); pis = np.where(pis < 1e-10 * pis.max(), 0, pis); pi = pis / pis.sum()
    worst['B'] = max(worst['B'], L * np.exp((s - 1) * renyi(pi, r)) / Q ** s - 1)
    for a in [0.3, r, 0.999, 2.0]:
        worst['SH'] = max(worst['SH'], renyi(pi, a) - renyi(rho / L, a))
    K = twisted_matrix(beta, xi)
    eps = np.sqrt(max(2 * L * (L - abs(A)), 0)); al = A / abs(A)
    off2 = 0.0; offb = 0.0
    for t in range(1, nu):
        for tp in range(t):
            d = t - tp
            bI = interval_coeff(beta, xi, tp, t)
            worst['C1'] = max(worst['C1'], abs(K[t, tp] - al ** (d - 1) * bI) - (d - 1) * eps - 1e-12 * L)
            off2 += abs(K[t, tp]) ** 2
    gain = abs(A) / (nu ** (1 - s) * Q ** s)
    bound = (abs(A) / L) * (1 + 2 * off2 / (nu * L ** 2)) ** (s - 1)
    worst['C2'] = max(worst['C2'], gain / bound - 1)


cnt = 0
for n in [1, 2, 3, 4, 5, 6]:
    nu = n + 1
    for trial in range(300):
        N = rng.integers(1, 2 * nu + 3)
        kind = trial % 3
        if kind == 0:   # generic complex atoms
            beta = rng.normal(size=N) + 1j * rng.normal(size=N)
        elif kind == 1:  # nearly phase-aligned atoms
            beta = rng.exponential(size=N) * np.exp(0.2j * rng.normal(size=N))
        else:            # heavy-tailed masses
            beta = rng.pareto(1.5, size=N) * np.exp(2j * np.pi * rng.random(N))
        xi = np.exp(2j * np.pi * rng.random((N, n)))
        for s in [1.0, 1.2, 1.7, nu / 2]:
            if s < 1: continue
            check_system(beta, xi, s); cnt += 1
# atoms from actual random networks (any inputs, DC, all frequencies)
for (p, n, m) in [(5, 2, 6), (7, 3, 5), (5, 4, 4), (11, 2, 7), (3, 3, 5)]:
    for trial in range(20):
        net = Net(rng.normal(size=(m, n, p)), rng.normal(size=(m, p)))
        beta, xi = net.atoms()
        for j in range(beta.shape[1]):
            for s in [1.0, 1.3, (n + 1) / 2]:
                check_system(beta[:, j], xi[:, :, j], s); cnt += 1
print(f"{cnt} systems checked; worst relative excesses (must be <= ~1e-12):")
for k, v in worst.items(): print(f"  {k}: {v:.3e}")

# (D) IF blocks (any n): chain-matrix spectrum = nu copies of A; twisted spectrum uniform
for n in range(1, 8):
    p = 2 * n + 7
    U, W = if_block(p, n, 1, 1.3, phase=0.4)
    net = Net(U, W) if n <= 4 else None
    nu = n + 1
    # atoms directly from factor transforms (no table needed)
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    beta = 2 * p ** (-nu) * np.conj(Wh[:, 1]) * np.prod(Uh[:, :, 1], axis=1)
    xi = np.conj(Uh[:, :, 1]) / Uh[:, :, 1]
    lam = spec_chain(beta, xi); pis = spec_twisted(beta, xi)
    A = beta.sum()
    print(f"IF block n={n}: max|lambda_k - A|/|A| = {np.max(np.abs(lam - A)) / abs(A):.1e}; twisted spectrum spread = {np.ptp(pis) / pis.mean():.1e}")

# Non-IF systems with a single eigenvalue ('spectrally IF'): H = A I + nilpotent with nonzero interval interference?
# search: n=2, 4 atoms; minimise |e_2(N)|^2+|e_3(N)|^2 where N = H - A I, keeping |B|+|C|+|D| large.
from scipy.optimize import minimize
def obj(x, n=2, N=4):
    beta = x[:N] + 1j * x[N:2 * N]; xi = np.exp(1j * x[2 * N:].reshape(N, n))
    H = chain_matrix(beta, xi); A = beta.sum(); Nm = H - A * np.eye(n + 1)
    c = np.poly(Nm)                     # char poly coefficients of N: z^3 + c1 z^2 + c2 z + c3
    pen = np.sum(np.abs(c[1:]) ** 2) / abs(A) ** 2
    intf = sum(abs(interval_coeff(beta, xi, a, b)) for a in range(n) for b in range(a + 1, n + 1)) / abs(A)
    return pen + 1e-3 * (abs(A) - 1) ** 2 - 1e-4 * min(intf, 1.0)
best = None
for t in range(30):
    x0 = rng.normal(size=4 * 2 + 4 * 2)
    res = minimize(obj, x0, method='BFGS', options={'maxiter': 3000})
    if best is None or res.fun < best.fun: best = res
x = best.x; N = 4; n = 2
beta = x[:N] + 1j * x[N:2 * N]; xi = np.exp(1j * x[2 * N:].reshape(N, n))
lam = spec_chain(beta, xi); A = beta.sum()
intf = [abs(interval_coeff(beta, xi, a, b)) / abs(A) for a in range(n) for b in range(a + 1, n + 1)]
print(f"spectrally-IF non-IF example (n=2): max|lambda-A|/|A| = {np.max(np.abs(lam - A)) / abs(A):.2e}; interval interference/|A| = {np.round(intf, 3)}")
r = 1 / 1.5
print(f"   its Q/(nu^(1-r)|A|^r) = {np.sum(np.abs(beta) ** r) / (3 ** (1 - r) * abs(A) ** r):.6f} (>= 1 required by spectral Lemma 7)")
