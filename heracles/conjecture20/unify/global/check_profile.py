# Sanity checks.
# (A) n=2 directional bound for SINGLE-FREQUENCY single features (inputs/outputs in span{1,cos,sin} at freq 1, p>=5):
#       (m(l)+m(-l))/2 <= kappa(l) (Re beta - M |sin(pi l/p)| )   [Riesz test, no partner]
#                      <= kappa(l) * 3^{1-s} Q^s                    [Theorem A: Re beta <= 3^{1-s}Q^s + M/2]
#       and the partner version for ARBITRARY inputs:  (m(l)+m(-l))/2 <= kappa(l)(Re beta - |cos(pi l/p)| M).
# (B) profile programme: union of relabelled copies of one single feature theta_0 with allocation y has margin
#       >= min_delta sum_j y_j m_0(j delta)  (normalised),  which is > c_0 Gamma when the profile is non-uniform.
import itertools, math, numpy as np
from lib import *

rng = np.random.default_rng(7)

def profile(net):
    """m(l) = min_x G(x,l) (unnormalised), l = 1..p-1"""
    return net.gaps().min(0)

print("(A) directional bounds, n=2")
worst = {}
for trial in range(300):
    p = [5, 7, 11, 13][trial % 4]; n = 2; b = rng.uniform(2, 3); s = 3 / b; r = 1 / s
    m = rng.integers(2, 6); a = np.arange(p)
    single_freq = trial % 2 == 0
    if single_freq:
        B = np.stack([np.ones(p), np.cos(2 * np.pi * a / p), np.sin(2 * np.pi * a / p)])
        U = rng.normal(size=(m, n, 3)) @ B
    else:
        U = rng.normal(size=(m, n, p))
    U0, W0 = if_block(p, n, 1); k = min(m, 3)
    eps = 10 ** rng.uniform(-3, 0.5)
    U = eps * U; U[:k] += 2.0 * U0[:k]
    W = eps * rng.normal(size=(m, 2)) @ np.stack([np.cos(2 * np.pi * a / p), np.sin(2 * np.pi * a / p)])
    W[:k] += 2.0 * W0[:k]
    net = Net(U, W, b); mp = profile(net)
    at = net.atoms()[:, 0]; beta = at.sum(); Q = (np.abs(at) ** r).sum()
    sf = net.signflip(); M = max(abs(sf[(1, T)]) for T in [(1, 0), (0, 1), (1, 1)])
    aIF = 3 ** (1 - s) * Q ** s
    for l in range(1, (p - 1) // 2 + 1):
        ka = 1 - math.cos(2 * math.pi * l / p); lhs = 0.5 * (mp[l - 1] + mp[p - l - 1])
        if single_freq:
            mid = ka * (beta.real - M * abs(math.sin(math.pi * l / p))) - 2 * 0  # Riesz, no partner (unnormalised, kappa*M/2 form below)
            mid = ka * beta.real - M * abs(math.sin(math.pi * l / p))
            rhs = ka * aIF
            key = 'single-freq'
        else:
            mid = ka * (beta.real - abs(math.cos(math.pi * l / p)) * M)
            rhs = None; key = 'general (partner form)'
        worst.setdefault(key, [-np.inf, -np.inf])
        worst[key][0] = max(worst[key][0], (lhs - mid) / max(abs(mid), 1e-12))
        if rhs is not None: worst[key][1] = max(worst[key][1], (mid - rhs) / max(abs(rhs), 1e-12))
for k, v in worst.items():
    print(f"  {k}: max relative violation of Riesz step {v[0]:.2e}" + (f";  of Theorem-A step {v[1]:.2e}" if v[1] > -np.inf else ""))

print("(B) profile-programme construction")
for (p, n, b) in [(7, 2, 2.0), (11, 2, 2.0), (7, 3, 2.0)]:
    s = (n + 1) / b; G, y, _ = programme(p, s, restarts=20); F = (p - 1) // 2
    # a non-IF single feature with positive cone margin: IF block plus interference-creating perturbation
    U0, W0 = if_block(p, n, 1)
    U = U0 + 0.02 * rng.normal(size=U0.shape); W = W0 + 0.02 * rng.normal(size=W0.shape)
    net0 = Net(U, W, b)
    G0 = net0.gaps(); kap = np.array([1 - math.cos(2 * math.pi * d / p) for d in range(1, p)])
    c0 = (G0 / kap[None, :]).min()
    if c0 <= 0: print("  skip (nonpositive cone margin)"); continue
    m0 = G0.min(0)                                             # profile
    nonunif = (m0 / kap / c0).max()
    # union of relabelled copies scaled by y_j (table factor), j in spt y
    Us, Ws = [], []
    for j in range(1, F + 1):
        if y[j - 1] <= 0: continue
        idx = (j * np.arange(p)) % p
        sc = y[j - 1] ** (1 / (n + 1))
        Us.append(net0.U[:, :, idx] * sc); Ws.append(net0.W[:, idx] * sc)
    U_ = np.concatenate(Us); W_ = np.concatenate(Ws); un = Net(U_, W_, b)
    hU = un.margin()
    pred = min(sum(y[j - 1] * m0[(j * d) % p - 1] for j in range(1, F + 1) if y[j - 1] > 0) for d in range(1, p))
    print(f"  p={p} n={n} b={b}: c0 Gamma = {c0*G:.6e}   min_d sum_j y_j m0(j d) = {pred:.6e}   union margin = {hU:.6e}   (max profile/(c0 kappa) = {nonunif:.3f})")
