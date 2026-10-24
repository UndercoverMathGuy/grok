# Referee checks of R0 (tax identities), R2(a) (weighted restricted bound, with cancelling non-features),
# R3 (temperature + tax derivative), single-neuron budget, R12 cost removal, R14(c) split identities at b=2.
import numpy as np, itertools, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
from vlib import *
rng = np.random.default_rng(12345)

def rand_net(m, n, p, dc=True):
    U = rng.normal(size=(m, n, p)) * rng.uniform(0.3, 2, size=(m, n, 1))
    W = rng.normal(size=(m, p)) * rng.uniform(0.3, 2, size=(m, 1))
    if not dc:
        U = U - U.mean(2, keepdims=True); W = W - W.mean(1, keepdims=True)
    return U, W

def wmargin(U, W, p, n, lam):
    f = table(U, W); F = (p - 1) // 2
    g = sum(lam[l - 1] * Gl(f, p, n, l) for l in range(1, F + 1))
    return g.min()

worst = {}
def upd(k, v):
    worst[k] = max(worst.get(k, -np.inf), v)

for trial in range(60):
    p = [3, 5, 7][trial % 3]; n = [2, 3][(trial // 3) % 2]
    if p == 7 and n == 3: n = 2
    nu = n + 1; F = (p - 1) // 2; m = rng.integers(1, 6)
    U, W = rand_net(m, n, p)
    f = table(U, W); g = gaps(f, p, n)
    # gap decomposition: g = sum_l G_l (DC cancels)
    Gsum = sum(Gl(f, p, n, l) for l in range(1, F + 1))
    upd('gap = sum_l G_l', np.abs(g - Gsum).max())
    # E_x Phi_l = conj(beta_l)
    be = beta(f, p, n)
    for l in range(1, F + 1):
        upd('E Phi_l - conj beta_l', abs(Phi(f, p, n, l).mean() - np.conj(be[l - 1])))
    # tax identity
    mu = rng.uniform(0.3, 3, F); lam = mu ** -0.5
    Lm = proj(p, 0) + sum(np.sqrt(mu[l - 1]) * proj(p, l) for l in range(1, F + 1))
    Wt = W @ Lm
    upd('h(theta) - h_lam(tau theta)', abs(g.min() - wmargin(U, Wt, p, n, lam)))
    b = rng.uniform(2, nu)
    upd('cost(tau) - cost_mu', abs(cost(U, Wt, b) - cost(U, W, b, mu)) / cost(U, W, b))
    bt = beta(table(U, Wt), p, n)
    upd('beta(tau) - sqrt(mu) beta', np.abs(bt - np.sqrt(mu) * be).max())
    # single neuron budget sum_j |beta_ij|^r <= C_b ||omega_i||^b, C_b^s = B_p
    s = nu / b; r = 1 / s; Cb = Bp(p, n) ** (1 / s)
    for i in range(m):
        bi = beta(table(U[i:i+1], W[i:i+1]), p, n)
        upd('budget ratio (<=1)', np.sum(np.abs(bi) ** r) / (Cb * neuron_sq(U[i:i+1], W[i:i+1])[0] ** (b / 2)))
    # temperature identity and tax derivative by central differences
    def lgam(sv, muv=None):
        bb = nu / sv; return np.log(abs(g.min())) - sv * np.log(cost(U, W, bb, muv))
    hstep = 1e-5
    dnum = (lgam(s + hstep) - lgam(s - hstep)) / (2 * hstep)
    upd('dlog gamma/ds + H', abs(dnum + H(shares(U, W, b))))
    E = Evec(U, W, b)
    for l in range(1, F + 1):
        e = np.zeros(F); e[l - 1] = hstep
        dmu = (lgam(s, 1 + e) - lgam(s, 1 - e)) / (2 * hstep)
        upd('dlog gamma_mu/dmu_l + nu/2 E_l', abs(dmu + nu / 2 * E[l - 1]))
    # R12 cost removal: P(theta^{-j}) <= (1 - E_j) P(theta), b >= 2
    for j in range(1, F + 1):
        Wm = W - W @ proj(p, j)
        upd('P(-j)/P - (1-E_j) (<=0)', cost(U, Wm, b) / cost(U, W, b) - (1 - E[j - 1]))

# R2(a): weighted restricted bound for N_S networks WITH cancellation on j notin S
ratios = []
for trial in range(240):
    p = [5, 7, 3][trial % 3]; n = 2 if (p == 7 or trial % 2) else 3
    nu = n + 1; F = (p - 1) // 2
    S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False).tolist())
    m = rng.integers(1, 4)
    U, W = rand_net(m, n, p)
    # build cancellation: duplicate neurons, negate frequency-j output (j notin S) on the copy
    flip = proj(p, 0).copy()
    for l in range(1, F + 1):
        flip += (1 if l in S else -1) * proj(p, l)
    U2 = np.concatenate([U, U]); W2 = np.concatenate([W, W @ flip])
    # optionally rescale a random subset to unbalance
    c0 = rng.uniform(0.5, 2); W2[:m] *= c0; U2[:m] /= c0 ** (1.0 / n)
    c1 = rng.uniform(0.7, 1.4); U2[m:] *= c1
    W2[m:] /= c1 ** n   # keep the rank-one term => cancellation persists
    f = table(U2, W2); be = beta(f, p, n)
    off = [abs(be[j - 1]) for j in range(1, F + 1) if j not in S]
    assert (max(off) if off else 0) < 1e-10, off
    lam = rng.uniform(0.5, 1.8, F)
    b = [2.0, nu, rng.uniform(2, nu)][trial % 3]; s = nu / b
    Gam, M = programme(p, s, lam=lam, S=S)
    h = sum(lam[l - 1] * Gl(f, p, n, l) for l in range(1, F + 1)).min()
    # interference present on j notin S?
    ratios.append(h / (Bp(p, n) * Gam * cost(U2, W2, b) ** s))
print('R2(a) weighted restricted bound with cancelling non-features: max ratio = %.6f over %d nets (must be <= 1)' % (max(ratios), len(ratios)))

# R14(c): b = 2 split identities
for trial in range(40):
    p = [5, 7][trial % 2]; n = 2; F = (p - 1) // 2; m = rng.integers(1, 5)
    U, W = rand_net(m, n, p, dc=False)
    Us = np.concatenate([U @ proj(p, k) for k in range(1, F + 1)]); Ws = np.concatenate([W @ proj(p, k) for k in range(1, F + 1)])
    upd('b=2 DC-free: |P - P_split|/P', abs(cost(U, W, 2) - cost(Us, Ws, 2)) / cost(U, W, 2))
    upd('b=2 DC-free: |E - E_split|', np.abs(Evec(U, W, 2) - Evec(Us, Ws, 2)).max())
    upd('b=2 DC-free: H - H_split (<=0)', H(shares(U, W, 2)) - H(shares(Us, Ws, 2)))
    upd('b=2: |beta - beta_split|', np.abs(beta(table(U, W), p, n) - beta(table(Us, Ws), p, n)).max())
    U, W = rand_net(m, n, p, dc=True)
    Us = np.concatenate([U @ proj(p, k) for k in range(1, F + 1)]); Ws = np.concatenate([W @ proj(p, k) for k in range(1, F + 1)])
    upd('b=2 with DC: |E - (Psplit/P) E_split|', np.abs(Evec(U, W, 2) - cost(Us, Ws, 2) / cost(U, W, 2) * Evec(Us, Ws, 2)).max())
for k, v in worst.items():
    print('%-42s %.3e' % (k, v))
