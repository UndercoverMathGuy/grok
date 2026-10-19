# K3(a): aligned class-C networks with arbitrary (multi-set) families, built from atoms.
# If skew(d_{+-d*}) <= 0 and C^2 sum_l kappa_l(d*) E_l / L_l^2 <= 1 then h <= sum_l at_l kappa_l(d*), at_l = nu^{1-s} Q_l^s.
import numpy as np, itertools, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint')
from cjlib import gamma_exact
rng = np.random.default_rng(7)
tested = 0; worst = -1e9; worst_bound = -1e9; skipped = 0
for trial in range(300):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); nu = n+1; F = (p-1)//2
    s = rng.uniform(1.0, nu/2); r = 1/s; C = 2*(s-1)/nu
    X = np.indices((p,)*n).reshape(n, -1).T
    S = rng.choice(np.arange(1, F+1), size=int(rng.integers(1, F+1)), replace=False)
    Phi = []; L = []; Q = []; E = []
    for l in S:
        m = int(rng.integers(nu, nu+3)); eps = rng.choice([0.02, 0.1, 0.3])
        rho_ = np.abs(1 + eps*rng.normal(size=m))*rng.exponential()
        xi = np.exp(1j*(2*np.pi*np.outer(np.arange(m), np.ones(n))*1.0/nu*np.arange(1, n+1)[None, :]*0 + 2*np.pi*np.arange(m)[:, None]/nu + eps*rng.normal(size=(m, n))))
        z = np.exp(2j*np.pi*2*l*X/p)                                   # (N, n)
        field = ((1 + z[:, None, :]*xi[None, :, :]).prod(2)) @ rho_       # aligned atoms (real positive)
        Phi.append(field); L.append(rho_.sum()); Q.append(np.sum(rho_**r)); E.append(np.mean(np.abs(field - rho_.sum())**2))
    Phi = np.array(Phi); L = np.array(L); Q = np.array(Q); E = np.array(E); at = nu**(1-s)*Q**s
    KAP = 1 - np.cos(2*np.pi*np.outer(np.arange(1, p), S)/p); CD = 1 - np.exp(-2j*np.pi*np.outer(np.arange(1, p), S)/p)
    G = np.real(CD @ Phi)                                                # (p-1, N) gaps g(x,d)
    h = G.min(); ds = int(np.argmin(KAP @ at)); dm = p-2-ds
    Dp = G[ds] - G[ds].mean(); Dm = G[dm] - G[dm].mean()
    cond = C**2*np.sum(KAP[ds]*E/L**2)
    sk = (np.mean(Dp**3) <= 1e-12*np.mean(Dp**2)**1.5) and (np.mean(Dm**3) <= 1e-12*np.mean(Dm**2)**1.5)
    # always-valid intermediate bound when skew ok: h <= sum Re A kappa - sigma
    if sk:
        sig = np.sqrt(KAP[ds] @ E); ReA = Phi.mean(1).real
        worst_bound = max(worst_bound, (h - (KAP[ds] @ ReA - sig))/abs(KAP[ds] @ ReA))
    if cond <= 1 and sk:
        tested += 1; worst = max(worst, h/(KAP[ds] @ at))
    elif cond <= 1: skipped += 1
print(f"networks meeting both hypotheses: {tested}; max h / sum at kappa(d*) = {worst:.5f} (<= 1); "
      f"max violation of h <= sum Re A kappa - sigma (skew<=0): {worst_bound:.2e} (<= 0); small-interference but skewed: {skipped}")
