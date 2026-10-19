"""Hostile test of R7 (twisted Rotfel'd payment + imported bounds) at s = 3/2:
is F(own data) <= P_1 = 3^{1-s} (sum |beta_i|^{1/s})^s for EVERY atom system?
Independent float implementation of the payment (grid over L, psi) + adversarial minimisation of P_1 - F over atoms;
worst cases re-checked with the rigorous pay_adapt.payment on point boxes."""
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '..')
rng = np.random.default_rng(7)
s = 1.5; r = 1/s
def S_of(P, q):
    P = np.minimum(P, 3.0); m = 2*np.sqrt(np.maximum(P, 0)/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.where(m > 1e-14, 4*q/np.maximum(m**3, 1e-300), 0.0)
    th = np.arccos(np.clip(arg, -1, 1))/3
    roots = [m*np.cos(th - 2*np.pi*k/3) for k in range(3)]
    return sum(np.maximum(1+x, 0)**r for x in roots)
def Smin(Pbar, qlow):
    P = np.clip(Pbar, 0, 3); dh = 2*(P/3)**1.5
    qs = np.maximum(qlow, np.maximum(P-1, -dh))
    out = S_of(P, qs)
    return np.where(qlow > dh + 1e-15, np.inf, out)
TG = np.concatenate([[0.0], np.geomspace(1e-6, 1.5, 70)])
PS = np.linspace(-np.pi, np.pi, 97)
def F_float(m1, m2, m12, phi):
    M = max(m1, m2, m12); Lmin = max(1.0, M)
    b1 = 1 - (1 - 3**(1-s))*M; b2 = 1 - np.sqrt(m1**2+m2**2+m12**2)/(2*np.sqrt(3)); b3 = max(1, M)*3**(1-s)
    L = Lmin*(1+TG)[:, None]; psi = PS[None, :]
    rho = np.sqrt(2*L*(L-1.0))
    best = -np.inf
    for (mX, mY, mZ) in [(m12, m1, m2), (m2, m1, m12), (m1, m2, m12)]:
        P = (mY**2+mZ**2+mX**2+rho**2+2*mX*rho*np.cos(psi-phi))/L**2
        q = 2*mY*mZ*(mX*np.cos(phi)+rho*np.cos(psi))/L**3
        v = L*(Smin(P, q)/3)**s
        best = max(best, min(v.min(), 3**(1-s)*Lmin*2.5))
    return max(b1, b2, b3, best), dict(c1=b1, thA=b2, triv=b3, tw=best)
def own(beta, xi, eta):
    A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum()
    return A, B, C, D
def P1(beta): return 3**(1-s)*np.sum(np.abs(beta)**r)**s
def unpack(v, k):
    rr = np.abs(v[:k]); pa = v[k:2*k]; xa = v[2*k:3*k]; ea = v[3*k:4*k]
    return rr*np.exp(1j*pa), np.exp(1j*xa), np.exp(1j*ea)
def gap(v, k):
    beta, xi, eta = unpack(v, k)
    A, B, C, D = own(beta, xi, eta)
    if abs(A) < 1e-9: return 10.0
    lam = abs(A); A, B, C, D = A/lam, B/lam, C/lam, D/lam
    phi = np.angle(A*D*np.conj(B*C)) if abs(B*C*D) > 0 else 0.0
    return P1(beta)/lam - F_float(abs(B), abs(C), abs(D), phi)[0]
# random sampling
worst = (np.inf, None)
for t in range(1500):
    k = rng.integers(1, 7)
    v = np.concatenate([rng.exponential(size=k), rng.uniform(-np.pi, np.pi, 3*k)])
    if t % 3 == 0:  # near-aligned atoms (A large relative to L) where the disc radius is small
        v[k:2*k] = rng.normal(scale=0.3, size=k)
    g = gap(v, k)
    if g < worst[0]: worst = (g, (v, k))
print(f"random: min[P1 - F_float] over 1500 atom systems = {worst[0]:.4e}", flush=True)
# adversarial local minimisation
results = []
for t in range(30):
    k = [2, 3, 3, 4, 5][t % 5]
    v0 = np.concatenate([rng.exponential(size=k), rng.normal(scale=[0.3, 1.0][t % 2], size=k), rng.uniform(-np.pi, np.pi, 2*k)])
    res = minimize(gap, v0, args=(k,), method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-9, fatol=1e-12))
    results.append((res.fun, k, res.x))
results.sort(key=lambda z: z[0])
import pickle; pickle.dump(results, open('v3_results.pkl','wb'))
print("adversarial: best 6 values of min[P1 - F_float]:", [round(z[0], 6) for z in results[:6]], flush=True)
from pay_adapt import payment
for (val, k, x) in results[:6]:
    beta, xi, eta = unpack(x, k)
    A, B, C, D = own(beta, xi, eta); lam = abs(A)
    m = np.array([abs(B), abs(C), abs(D)])/lam
    phi = float(np.angle(A*D*np.conj(B*C)))
    Fr = payment(m, m, phi, phi)
    Ff, parts = F_float(*m, phi)
    print(f"  k={k} m={np.round(m,4)} phi={phi:+.4f}: P1/|A|={P1(beta)/lam:.6f}  F_float={Ff:.6f} {dict((a, round(b, 4)) for a, b in parts.items())}  F_rig={Fr:.6f}  slack_rig={P1(beta)/lam-Fr:.2e}", flush=True)
