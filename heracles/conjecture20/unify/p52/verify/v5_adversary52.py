"""Hostile search at (5,2), b=2 (s=3/2): maximise h subject to cost sum||omega||^2 = 1 and beta_2 = 0 exactly
(SLSQP equality constraints), general real neurons (DC, all frequencies, outputs on all frequencies).
Reports max gamma/gamma_IF found; claim R9 says <= 0.995 (authors observed 0.7818)."""
import numpy as np, itertools
from scipy.optimize import minimize
rng = np.random.default_rng(99)
p = 5; rho = np.exp(2j*np.pi/p); B5 = 2*30**-1.5
gIF = 3**-0.5*B5*2.5*2**-1.5
a = np.arange(p)
X = np.array(list(itertools.product(range(p), repeat=2))); Y = X.sum(1) % p
Ek = np.exp(-2j*np.pi*np.outer(np.arange(p), a)/p)   # hat = Ek @ v
def unpack(v, m):
    U1 = v[:m*p].reshape(m, p); U2 = v[m*p:2*m*p].reshape(m, p); W = v[2*m*p:3*m*p].reshape(m, p); return U1, U2, W
def gaps_and_jac(v, m):
    U1, U2, W = unpack(v, m); t = v[-1]
    P1 = U1[:, X[:, 0]]; P2 = U2[:, X[:, 1]]                  # (m, 25)
    out = []; J = []
    for d in (1, 2, 3, 4):
        Wy = W[:, Y]; Wd = W[:, (Y-d) % p]                    # (m, 25)
        g = np.sum(P1*P2*(Wy-Wd), 0)
        out.append(g - t)
        # jacobian rows (25, nvar)
        Jd = np.zeros((25, len(v)))
        coef1 = P2*(Wy-Wd); coef2 = P1*(Wy-Wd); coefw = P1*P2
        for i in range(m):
            Jd[np.arange(25), i*p + X[:, 0]] += coef1[i]
            Jd[np.arange(25), m*p + i*p + X[:, 1]] += coef2[i]
            Jd[np.arange(25), 2*m*p + i*p + Y] += coefw[i]
            Jd[np.arange(25), 2*m*p + i*p + (Y-d) % p] -= coefw[i]
        Jd[:, -1] = -1
        J.append(Jd)
    return np.concatenate(out), np.vstack(J)
def beta2(v, m):
    U1, U2, W = unpack(v, m)
    h1 = U1 @ Ek[2]; h2 = U2 @ Ek[2]; hw = W @ Ek[2]
    val = 2/125*np.sum(np.conj(hw)*h1*h2)
    J = np.zeros(len(v), complex)
    for i in range(m):
        J[i*p:(i+1)*p] = 2/125*np.conj(hw[i])*h2[i]*Ek[2]
        J[m*p+i*p:m*p+(i+1)*p] = 2/125*np.conj(hw[i])*h1[i]*Ek[2]
        J[2*m*p+i*p:2*m*p+(i+1)*p] = 2/125*np.conj(Ek[2])*h1[i]*h2[i]
    return val, J
def beta1(v, m):
    U1, U2, W = unpack(v, m)
    return 2/125*np.sum(np.conj(W @ Ek[1])*(U1 @ Ek[1])*(U2 @ Ek[1]))
def solve(v0, m):
    cons = [dict(type='ineq', fun=lambda v: gaps_and_jac(v, m)[0], jac=lambda v: gaps_and_jac(v, m)[1]),
            dict(type='eq', fun=lambda v: np.array([np.sum(v[:-1]**2) - 1]), jac=lambda v: np.concatenate([2*v[:-1], [0]])[None, :]),
            dict(type='eq', fun=lambda v: np.array([beta2(v, m)[0].real, beta2(v, m)[0].imag]),
                 jac=lambda v: np.vstack([beta2(v, m)[1].real, beta2(v, m)[1].imag]))]
    res = minimize(lambda v: -v[-1], v0, jac=lambda v: np.concatenate([np.zeros(len(v)-1), [-1]]), constraints=cons,
                   method='SLSQP', options=dict(maxiter=600, ftol=1e-12))
    return res
def if_block(j, amp, m_extra, rng):
    """3-neuron IF block of feature j (phases 0, 2pi/3, 4pi/3 on slot 1, opposite on output) + random extra neurons"""
    U1 = []; U2 = []; W = []
    for k in range(3):
        ph = 2*np.pi*k/3
        U1.append(np.cos(2*np.pi*j*a/p + ph)); U2.append(np.cos(2*np.pi*j*a/p + ph)); W.append(np.cos(2*np.pi*j*a/p + 2*ph))
    U1 = np.array(U1); U2 = np.array(U2); W = np.array(W)
    if m_extra:
        U1 = np.vstack([U1, 0.3*rng.normal(size=(m_extra, p))]); U2 = np.vstack([U2, 0.3*rng.normal(size=(m_extra, p))]); W = np.vstack([W, 0.3*rng.normal(size=(m_extra, p))])
    v = np.concatenate([U1.ravel(), U2.ravel(), W.ravel()]); v = v/np.linalg.norm(v)
    return v
best = []
for trial in range(48):
    m = [3, 4, 5, 6, 8, 10][trial % 6]
    if trial % 3 == 0:
        v = if_block(1, 1.0, m-3, rng) + 0.05*rng.normal(size=3*m*p)
    elif trial % 3 == 1:
        v = rng.normal(size=3*m*p)
    else:   # two-feature IF network start (feature 2 must die)
        v = np.concatenate([if_block(1, 1, 0, rng).reshape(3, 3, p), if_block(2, 1, 0, rng).reshape(3, 3, p)], axis=1).ravel() if m == 6 else rng.normal(size=3*m*p)
        v = v + 0.05*rng.normal(size=3*m*p)
    v = v/np.linalg.norm(v)
    t0 = gaps_and_jac(np.append(v, 0), m)[0].min()
    res = solve(np.append(v, t0), m)
    vv = res.x
    vv_n = vv[:-1]/np.linalg.norm(vv[:-1])
    # true margin at the normalised point, and residual beta_2
    h = gaps_and_jac(np.append(vv_n, 0), m)[0].min()
    b2 = abs(beta2(np.append(vv_n, 0), m)[0]); b1 = abs(beta1(np.append(vv_n, 0), m))
    best.append((h/gIF, m, trial, b2/max(b1, 1e-30), res.status))
best.sort(reverse=True)
print("top results gamma/gamma_IF (m, trial, |beta2|/|beta1|, status):")
for r in best[:10]: print(f"  {r[0]:.5f}  m={r[1]} trial={r[2]} b2/b1={r[3]:.1e} st={r[4]}")
print(f"max gamma/gamma_IF found = {best[0][0]:.5f}  (kappa_a/Gamma_3/2 = {(1-np.cos(2*np.pi/5))/(2.5*2**-1.5):.5f})")
