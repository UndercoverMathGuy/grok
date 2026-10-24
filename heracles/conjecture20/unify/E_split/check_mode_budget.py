"""Sanity check for R7 (mode budget off the endpoint).  For one neuron omega = (u_1..u_n, w) in (R^p)^nu:
  e = B_p ||omega||^nu,  beta_j = 2 p^-nu conj(what(j)) prod_t uhat_t(j)  (j in F),
  N = non-own mode mass = sum_{j in F} sum_{k in Z_p^n minus {+-j}^n} |2 p^-nu what(j) prod_t uhat_t(k_t)|^2.
Claims: (i) sum_j |beta_j|^2 + 2^-n N <= e^2 (every b);
        (ii) for b in [2, nu], r = b/nu:  sum_j |beta_j|^r + ((b-2)/(2 nu)) e^(r-2) 2^-n N <= e^r;
        (iii) at b = 2 no positive constant works: equal-split neurons are budget-tight with N > 0."""
import math
import numpy as np

rng = np.random.default_rng(5)
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def neuron_data(U, W):
    n, p = U.shape; nu = n + 1
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W)
    F = range(1, (p - 1) // 2 + 1)
    beta = np.array([2 * p ** (-nu) * np.conj(Wh[j]) * np.prod(Uh[:, j]) for j in F])
    tot_in = np.prod((np.abs(Uh) ** 2).sum(1))
    N = 0.0
    for j in F:
        own = np.prod(2 * np.abs(Uh[:, j]) ** 2)
        N += (4 / p ** (2 * nu)) * abs(Wh[j]) ** 2 * (tot_in - own)
    norm = math.sqrt((U ** 2).sum() + (W ** 2).sum())
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    return beta, N, Bp * norm ** nu


worst_i = -np.inf; worst_ii = -np.inf; ratios = []
for trial in range(20000):
    p = int(rng.choice([3, 5, 7, 11, 13])); n = int(rng.choice([2, 3, 4])); nu = n + 1
    kind = trial % 5
    if kind == 0:      # gaussian
        U = rng.standard_normal((n, p)); W = rng.standard_normal(p)
    elif kind == 1:    # sparse frequency content, random per factor
        a = np.arange(p); U = np.zeros((n, p)); W = np.zeros(p)
        for t in range(n + 1):
            v = sum(rng.standard_normal() * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 6.3)) for k in rng.choice(range(0, (p + 1) // 2), size=rng.integers(1, 3)))
            if t < n: U[t] = v
            else: W = v
    elif kind == 2:    # nearly pure frequency-1 neuron with small leakage
        a = np.arange(p); eps = 10 ** rng.uniform(-4, -0.5)
        U = np.array([np.cos(2 * np.pi * a / p + rng.uniform(0, 6.3)) + eps * rng.standard_normal(p) for _ in range(n)])
        W = np.cos(2 * np.pi * a / p + rng.uniform(0, 6.3)) + eps * rng.standard_normal(p)
    elif kind == 3:    # common spread distribution over frequencies (the b=2 tight family), balanced
        a = np.arange(p); K = rng.integers(1, (p - 1) // 2 + 1); ks = rng.choice(range(1, (p + 1) // 2), size=K, replace=False)
        amp = rng.uniform(0.2, 1, K)
        U = np.array([sum(am * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 6.3)) for am, k in zip(amp, ks)) for _ in range(n)])
        W = sum(am * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 6.3)) for am, k in zip(amp, ks))
        nr = np.linalg.norm(U, axis=1); U = U / nr[:, None]; W = W / np.linalg.norm(W)
    else:              # unbalanced scales + DC
        U = rng.standard_normal((n, p)) * rng.uniform(0.1, 3, (n, 1)) + rng.uniform(-1, 1, (n, 1)); W = rng.standard_normal(p) * rng.uniform(0.1, 3)
    beta, N, e = neuron_data(U, W)
    lhs_i = (np.abs(beta) ** 2).sum() + 2.0 ** (-n) * N
    worst_i = max(worst_i, lhs_i / e ** 2 - 1)
    for b in [2.0, 2.0 + 1e-3, 2.1, 2.5, nu]:
        r = b / nu
        lhs = (np.abs(beta) ** r).sum() + ((b - 2) / (2 * nu)) * e ** (r - 2) * 2.0 ** (-n) * N
        worst_ii = max(worst_ii, lhs / e ** r - 1)
        if b > 2 and N > 1e-12 * e ** 2:
            ratios.append((e ** r - (np.abs(beta) ** r).sum()) / (e ** (r - 2) * 2.0 ** (-n) * N) / ((b - 2) / (2 * nu)))
say(f"(i)  max [sum|beta|^2 + 2^-n N]/e^2 - 1 = {worst_i:.2e}  (<= 0 up to rounding)")
say(f"(ii) max [sum|beta|^r + (b-2)/(2nu) e^(r-2) 2^-n N]/e^r - 1 = {worst_ii:.2e}  (<= 0)")
say(f"     min observed (slack / (c_b * e^(r-2) 2^-n N)) = {min(ratios):.3f}  (>= 1 by the lemma)")
# (iii) equal split at b = 2: tight with N > 0
for (p, n, K) in [(7, 2, 2), (11, 2, 3), (13, 3, 4), (11, 4, 5)]:
    a = np.arange(p); nu = n + 1
    v = sum(np.cos(2 * np.pi * k * a / p) for k in range(1, K + 1)); v = v / np.linalg.norm(v)
    U = np.array([v] * n); W = v.copy()
    beta, N, e = neuron_data(U, W)
    say(f"(iii) p={p} n={n} equal split over {K} freqs: sum|beta|^(2/nu)/e^(2/nu) = {(np.abs(beta)**(2/nu)).sum()/e**(2/nu):.12f}, "
        f"2^-n N/e^2 = {2.0**(-n)*N/e**2:.4f} (= 1 - K^(1-nu) = {1-K**(1-nu):.4f})")
open("check_mode_budget.out", "w").write("\n".join(out) + "\n")

# network forms (iii)
def neuron_modes_l(U, W, l):
    """mass of all modes except the own modes of feature l, and the atom at l."""
    n, p = U.shape; nu = n + 1
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W)
    tot_in = np.prod((np.abs(Uh) ** 2).sum(1))
    tot = sum((4 / p ** (2 * nu)) * abs(Wh[j]) ** 2 * tot_in for j in range(1, (p - 1) // 2 + 1))
    own_l = (4 / p ** (2 * nu)) * abs(Wh[l]) ** 2 * np.prod(2 * np.abs(Uh[:, l]) ** 2)
    beta_l = 2 * p ** (-nu) * np.conj(Wh[l]) * np.prod(Uh[:, l])
    return tot - own_l, beta_l
w1 = -np.inf; w2 = -np.inf
for trial in range(3000):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); nu = n + 1
    m = int(rng.integers(1, 8)); b = float(rng.uniform(2, nu)); r = b / nu; s = 1 / r
    U = rng.standard_normal((m, n, p)) * rng.uniform(0.2, 2, (m, 1, 1)); W = rng.standard_normal((m, p))
    if trial % 2:
        U = U - U.mean(-1, keepdims=True)
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    P = sum(((U[i] ** 2).sum() + (W[i] ** 2).sum()) ** (b / 2) for i in range(m))
    # single own feature l = 1
    Nl = []; bl = []
    for i in range(m):
        a, c = neuron_modes_l(U[i], W[i], 1); Nl.append(a); bl.append(c)
    Q1 = sum(abs(c) ** r for c in bl); Rl = sum(math.sqrt(max(a, 0)) for a in Nl)
    w1 = max(w1, math.sqrt(Q1 ** (2 * s) + 2.0 ** (-n) * Rl ** 2) / (Bp * P ** s) - 1)
    # several features, b > 2
    Qs = 0.0; R = 0.0
    for i in range(m):
        beta, N, e = neuron_data(U[i], W[i]); Qs += (np.abs(beta) ** r).sum(); R += math.sqrt(N)
    rhs = Bp ** r * P * (1 - ((b - 2) / (2 * nu)) * 2.0 ** (-n) * (R / (Bp * P ** s)) ** 2)
    w2 = max(w2, Qs / (Bp ** r * P) - rhs / (Bp ** r * P))
say(f"(iii) single own feature: max sqrt(Q^2s + 2^-n R^2)/(B P^s) - 1 = {w1:.2e} (<= 0)")
say(f"(iii) several features: max [sum Q_j - B^r P (1 - c_b 2^-n (R/(B P^s))^2)]/(B^r P) = {w2:.2e} (<= 0)")
open("check_mode_budget.out", "w").write("\n".join(out) + "\n")
