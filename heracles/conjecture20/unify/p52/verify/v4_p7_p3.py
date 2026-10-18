"""Referee checks: R12 (delta-Fourier isolation) at p=7 n=2 and at p=3 n=11 (exact counterexample network),
R13 (cost ledger) at p=7, R11(ii), R17.  Own implementation."""
import numpy as np, itertools, sys
from fractions import Fraction as Fr
rng = np.random.default_rng(2026)
def hat(v, p):
    a = np.arange(p); return np.array([np.sum(v*np.exp(-2j*np.pi*k*a/p)) for k in range(p)])
def net_table(U, W, p, n):
    """U: m x n x p, W: m x p -> f[x..., c] via explicit sum"""
    X = np.array(list(itertools.product(range(p), repeat=n)))
    prod = np.ones((len(X), U.shape[0]))
    for t in range(n): prod *= U[:, t, :][:, X[:, t]].T
    return X, prod @ W          # (p^n, p)
def gaps(X, f, p):
    y = X.sum(1) % p
    G = np.stack([f[np.arange(len(X)), y] - f[np.arange(len(X)), (y-d) % p] for d in range(1, p)], axis=1)
    return y, G                 # G[:, d-1]
def fields(X, f, y, p):
    F = range(1, (p-1)//2+1)
    c = np.arange(p)
    return {l: (2/p)*np.exp(2j*np.pi*l*y/p)*(f @ np.exp(-2j*np.pi*l*c/p)) for l in F}
def coef(Phi, X, chi, p): return np.mean(Phi*np.exp(-2j*np.pi*(X @ np.array(chi))/p))
def kap(l, d, p): return 1 - np.cos(2*np.pi*l*d/p)
# ---------------------------------------------------------------- R12 at p = 7, n = 2: exact identity
p, n = 7, 2
worst_id = 0; worst_pos = np.inf; worst_bd = -np.inf
for t in range(120):
    m = rng.integers(1, 6)
    U = rng.normal(size=(m, n, p)) + rng.normal(size=(m, n, 1))*rng.uniform(0, 2)
    W = rng.normal(size=(m, p))
    X, f = net_table(U, W, p, n); y, G = gaps(X, f, p); h = G.min()
    Phi = fields(X, f, y, p)
    beta = {l: np.conj(coef(Phi[l], X, (0, 0), p)) for l in Phi}
    mu = rng.dirichlet(np.ones(3)); mu = np.concatenate([mu, mu[::-1]])/2     # symmetric on delta = 1..6
    K = {l: sum(mu[d-1]*kap(l, d, p) for d in range(1, p)) for l in Phi}
    for j in (1, 2, 3):
        for T in [(0,), (1,), (0, 1)]:
            chi = tuple((2*j if t_ in T else 0) % p for t_ in range(n))
            bjT = np.conj(coef(Phi[j], X, chi, p))
            for w in [rng.normal()+1j*rng.normal(), (mu.min()/2)*bjT/abs(bjT)]:
                q = (mu[None, :] + 2*np.real(w*np.exp(2j*np.pi*j*np.arange(1, p)/p)[None, :]*np.exp(-2j*np.pi*(X @ np.array(chi))/p)[:, None]))/p**n
                lhs = np.sum(q*G)
                rhs = sum(K[l]*beta[l].real for l in Phi) - p*np.real(w*np.conj(bjT))
                worst_id = max(worst_id, abs(lhs-rhs)/(1+abs(lhs)))
            # aligned test: q >= 0, h <= bound
            worst_pos = min(worst_pos, q.min())
            worst_bd = max(worst_bd, h - (sum(K[l]*beta[l].real for l in Phi) - (p/2)*mu.min()*abs(bjT)))
print(f"R12 p=7: max rel |sum q_w g - formula| = {worst_id:.2e};  min q (aligned) = {worst_pos:.2e};  max[h - bound] = {worst_bd:.3e} (<=0)")
# ---------------------------------------------------------------- R13 ledger at p = 7
B7 = 2*(42.0)**-1.5
worst13 = -np.inf
for t in range(400):
    m = rng.integers(1, 8)
    U = rng.normal(size=(m, n, p)) + rng.normal(size=(m, n, 1))*rng.uniform(0, 2)
    W = rng.normal(size=(m, p))
    if t % 3 == 0:   # IF-like single-frequency neurons
        a = np.arange(p)
        for i in range(m):
            jj = rng.integers(1, 4); ph = rng.uniform(0, 2*np.pi, 3)
            U[i, 0] = np.cos(2*np.pi*jj*a/p+ph[0]); U[i, 1] = np.cos(2*np.pi*jj*a/p+ph[1]); W[i] = np.cos(2*np.pi*jj*a/p+ph[2])
    Uh = np.array([[hat(U[i, tt], p) for tt in range(n)] for i in range(m)]); Wh = np.array([hat(W[i], p) for i in range(m)])
    nr = np.sum(U**2, axis=(1, 2)) + np.sum(W**2, 1)
    for s in (1.0, 1.1, 1.182, 1.5):
        r = 1/s; c1 = 1 - 3**(1-s); P = np.sum(nr**(1.5*r))
        terms = []
        for j in (1, 2, 3):
            bij = 2*p**-3*np.conj(Wh[:, j])*Uh[:, 0, j]*Uh[:, 1, j]
            Aj = abs(bij.sum())
            if Aj < 1e-12: continue
            Mj = max(abs((2*p**-3*np.conj(Wh[:, j])*Uh[:, 0, (-j if 0 in T else j) % p]*Uh[:, 1, (-j if 1 in T else j) % p]).sum()) for T in [(0,), (1,), (0, 1)])
            xj = Mj/Aj; Fj = max(1 - c1*xj, 3**(1-s)*max(1, xj))
            terms.append((Fj*Aj)**r)
        lhs = 3**(1-s)*B7*P**s; rhs = sum(terms)**s
        worst13 = max(worst13, (rhs-lhs)/lhs)
print(f"R13 p=7: max relative [ledger - 3^(1-s) B7 P^s] = {worst13:.3e} (<=0)")
# ---------------------------------------------------------------- R11(ii): single-feature networks at p = 7
worst11 = -np.inf
for t in range(300):
    m = rng.integers(1, 6); jj = rng.integers(1, 4)
    U = rng.normal(size=(m, n, p)) + rng.normal(size=(m, n, 1))
    W0 = rng.normal(size=(m, p))
    # outputs only on DC and frequency jj -> only feature jj (other beta_l = 0 because what(l) = 0)
    Wh = np.array([hat(W0[i], p) for i in range(m)]); keep = np.zeros(p, bool); keep[[0, jj, p-jj]] = True
    W = np.real(np.fft.ifft(np.where(keep[None, :], Wh, 0), axis=1))
    X, f = net_table(U, W, p, n); y, G = gaps(X, f, p); h = G.min()
    nr = np.sum(U**2, axis=(1, 2)) + np.sum(W**2, 1)
    for s in (1.0, 1.25, 1.5):
        P = np.sum(nr**(1.5/s)); gam = h/P**s; gIF = 3**(1-s)*B7*3.5*3**-s
        worst11 = max(worst11, gam/gIF)
print(f"R11(ii): max gamma/gamma_IF over random single-feature nets = {worst11:.4f} (claim <= 0.9682)")
# ---------------------------------------------------------------- R17
a = np.arange(p); ph = rng.uniform(0, 2*np.pi, 6)
mk = lambda k: np.cos(2*np.pi*1*a/p+ph[k]) + np.cos(2*np.pi*2*a/p+ph[k+3])
u1, u2, w = mk(0), mk(1), mk(2)
u1, u2, w = [v/np.linalg.norm(v) for v in (u1, u2, w)]
H1, H2, Hw = hat(u1, p), hat(u2, p), hat(w, p)
e = B7*3**1.5
b = {j: 2*p**-3*np.conj(Hw[j])*H1[j]*H2[j] for j in (1, 2, 3)}
tot = sum(abs(2*p**-3*Hw[j]*H1[k1]*H2[k2])**2 for j in (1, 2, 3) for k1 in range(p) for k2 in range(p))
own = sum(4*abs(b[j])**2 for j in (1, 2, 3))
print(f"R17: |b1|^(2/3)+|b2|^(2/3) = {abs(b[1])**(2/3)+abs(b[2])**(2/3):.6f}, e^(2/3) = {e**(2/3):.6f}; total mass/e^2 = {tot/e**2:.4f}; non-own/e^2 = {(tot-own)/e**2:.4f}")
# ---------------------------------------------------------------- R12 at p = 3, n = 11 (exact Conjecture-20 counterexample)
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec
net = ec.NETWORKS[11]
p, n = 3, 11
U = np.array([[float(v) for v in row] for row in net['u']]); W = np.array([[float(v) for v in row] for row in net['w']])
m = len(U)
X = np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int8)
cnt = np.stack([(X == a).sum(1) for a in range(3)], 1)
Z = np.prod(U[None, :, :]**cnt[:, None, :], axis=2)       # (3^n, m)
f = Z @ W
y = X.sum(1) % p
G = np.stack([f[np.arange(len(X)), y] - f[np.arange(len(X)), (y-d) % p] for d in (1, 2)], 1)
h = G.min()
Uh = np.array([hat(U[i], p) for i in range(m)]); Wh = np.array([hat(W[i], p) for i in range(m)])
beta1 = 2*p**-(n+1)*np.sum(np.conj(Wh[:, 1])*Uh[:, 1]**n)
P2 = np.sum(n*np.sum(U**2, 1) + np.sum(W**2, 1)); nu = n+1; s = nu/2
gIF = 3*nu*(6*nu**2)**(-nu/2)
print(f"p=3,n=11: h = {h:.6e}, gamma/gamma_IF = {h/P2**s/gIF:.4f}  (counterexample: > 1)")
worst = -np.inf
for k in range(1, n+1):
    bT = 2*p**-(n+1)*np.sum(np.conj(Wh[:, 1])*Uh[:, 1]**(n-k)*Uh[:, 2]**k)
    bound = 1.5*(beta1.real - 0.5*abs(bT))
    worst = max(worst, h - bound)
    if k in (1, 6, 11): print(f"   |T|={k}: h={h:.4e} <= bound {bound:.4e}  |beta_T|/|beta_1| = {abs(bT)/abs(beta1):.4f}")
print(f"R12 p=3 n=11: max over |T| of [h - bound] = {worst:.3e} (<=0)")
# identity check for T = {1,2,3}, random w (sum over all 3^11 x)
T = [0, 1, 2]; chi = np.zeros(n, int); chi[T] = 2
bT = 2*p**-(n+1)*np.sum(np.conj(Wh[:, 1])*Uh[:, 1]**(n-3)*Uh[:, 2]**3)
w = 0.1+0.2j; mu = np.array([0.5, 0.5])
ph = np.exp(-2j*np.pi*(X.astype(int) @ chi)/p)
q = (mu[None, :] + 2*np.real(w*np.exp(2j*np.pi*np.arange(1, 3)/p)[None, :]*ph[:, None]))/p**n
lhs = np.sum(q*G); rhs = 1.5*beta1.real - 3*np.real(w*np.conj(bT))
print(f"R12 p=3 identity: sum q g = {lhs:.10e}, formula = {rhs:.10e}")
