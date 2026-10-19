import numpy as np, sys
from vlib import *
rng = np.random.default_rng(101)
# W1 on actual networks: gamma(theta;s) = gamma(theta;1) exp(-(s-1)H_{1/s}(m)), any h (multiplicative form)
err = 0
for (p, n, m) in [(5, 2, 7), (3, 4, 5), (7, 3, 3)]:
    for t in range(20):
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        h = margin(U, W); nu = n + 1; sq = sqnorm(U, W); a = sq ** (nu / 2)
        for s in [1.0, 1.3, nu / 2]:
            b = nu / s
            gs = h / np.sum(sq ** (b / 2)) ** s; g1 = h / a.sum()
            err = max(err, abs(gs - g1 * np.exp(-(s - 1) * renyi(a, 1 / s))) / abs(gs))
print(f"W1 on networks: max rel err {err:.1e}")
# W1 on p=3 certificate n=11
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec
for n in [11, 13]:
    d = ec.NETWORKS[n]; nu = n + 1
    u = np.array([[float(x) for x in r] for r in d['u']]); w = np.array([[float(x) for x in r] for r in d['w']])
    sq = n * (u ** 2).sum(1) + (w ** 2).sum(1); a = sq ** (nu / 2); s = nu / 2
    lhs = np.log(1 / np.sum(sq) ** s); rhs = np.log(1 / a.sum()) - (s - 1) * renyi(a, 1 / s)
    print(f"  p=3 n={n}: log(1/P_2^s) - [log(1/P_nu) - (s-1)H_(2/nu)] = {lhs - rhs:.2e}; H_(1/s)(m) = {renyi(a, 1/s):.4f}, log width = {np.log(len(a)):.4f}")

# W5: split exchange. b in [2, nu]; with and without DC. Check inequality and equality cases; and sign of entropy change.
def split(U, W, p):
    F = (p - 1) // 2
    def proj(v, k):
        vh = np.fft.fft(v, axis=-1); mask = np.zeros(p); mask[k] = 1; mask[(-k) % p] = 1
        return np.real(np.fft.ifft(vh * mask, axis=-1))
    return np.concatenate([proj(U, k) for k in range(1, F + 1)]), np.concatenate([proj(W, k) for k in range(1, F + 1)])
worst_ineq = -np.inf; worst_eq = 0; neg_examples = 0; tot = 0
for (p, n, m) in [(5, 2, 4), (7, 3, 3), (11, 2, 3), (5, 4, 3)]:
    nu = n + 1
    for trial in range(15):
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        dc = trial % 2 == 1
        if not dc:
            U -= U.mean(2, keepdims=True); W -= W.mean(1, keepdims=True)
        Us, Ws = split(U, W, p)
        a = sqnorm(U, W) ** (nu / 2); asp = sqnorm(Us, Ws) ** (nu / 2)
        # clean atoms equal?
        for s in np.linspace(1, nu / 2, 5):
            b = nu / s
            Pb = np.sum(sqnorm(U, W) ** (b / 2)); Pbs = np.sum(sqnorm(Us, Ws) ** (b / 2))
            L = np.log(a.sum()) + (s - 1) * renyi(a, 1 / s); R = np.log(asp.sum()) + (s - 1) * renyi(asp, 1 / s)
            worst_eq = max(worst_eq, abs(L - s * np.log(Pb)), abs(R - s * np.log(Pbs)))
            worst_ineq = max(worst_ineq, R - L)
            if s > 1:
                tot += 1
                if renyi(asp, 1 / s) < renyi(a, 1 / s): neg_examples += 1
        if not dc:
            s = nu / 2
            Pb = np.sum(sqnorm(U, W)); Pbs = np.sum(sqnorm(Us, Ws))
            worst_eq = max(worst_eq, abs(Pb - Pbs) / Pb)
print(f"W5: algebraic identities max err {worst_eq:.1e}; max [split side - theta side] = {worst_ineq:.2e} (<=0 required)")
print(f"W5 reading: cases with H_1/s(m_split) < H_1/s(m_theta) (sharing RAISES entropy): {neg_examples} of {tot}")
# targeted: DC-free b>2 example where split lowers entropy
p, n = 5, 2; nu = 3; a_ = np.arange(p)
U = np.zeros((2, n, p)); W = np.zeros((2, p))
eps = 0.05
for t in range(n):
    U[0, t] = eps * (np.cos(2 * np.pi * a_ / p) + np.cos(4 * np.pi * a_ / p)); U[1, t] = np.cos(2 * np.pi * a_ / p)
W[0] = eps * (np.cos(2 * np.pi * a_ / p) + np.cos(4 * np.pi * a_ / p)); W[1] = np.cos(2 * np.pi * a_ / p)
Us, Ws = split(U, W, p)
a = sqnorm(U, W) ** (nu / 2); asp = sqnorm(Us, Ws) ** (nu / 2); asp = asp[asp > 1e-14]
for s in [1.2, 1.5]:
    print(f"  shared-neuron example s={s}: H(m_theta)={renyi(a,1/s):.5f}, H(m_split)={renyi(asp,1/s):.5f}")
# W9: one neuron h <= 0 ; random search incl. large n small p
best = -np.inf
for (p, n) in [(3, 2), (3, 6), (5, 3), (7, 2)]:
    for t in range(3000):
        U = rng.normal(size=(1, n, p)); W = rng.normal(size=(1, p))
        best = max(best, margin(U, W))
print(f"W9: max single-neuron margin over random draws = {best:.3e} (<=0)")
