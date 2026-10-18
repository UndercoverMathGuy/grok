# Referee: random check of every link in the R6 chain, and of the (P,q)-monotonicity of Lambda on the valid domain.
import numpy as np
rng = np.random.default_rng(9)
def Lam_nu(nu): nu = np.clip(nu, 0, None); return sum(v*np.log(v) for v in nu if v > 0)/(3*np.log(3))
def Lam_Pq(P, q): return Lam_nu(1 + np.sort(np.roots([1, 0, -P, -q]).real))
worst = dict(rot=np.inf, jen=np.inf, mono=np.inf, gain=np.inf, J=np.inf, rgn=np.inf)
for t in range(20000):
    N = int(rng.integers(1, 7)); s = rng.uniform(1, 1.5); r = 1/s; x = 3**(1 - s); c = (1 - x)/np.sqrt(3)
    rho = rng.exponential(size=N)**rng.uniform(0.3, 3); p = np.exp(1j*rng.uniform(0, 2*np.pi, N))
    if t % 3 == 0: p = np.exp(1j*0.2*rng.normal(size=N))
    xi = np.exp(1j*rng.uniform(0, 2*np.pi, N)); eta = np.exp(1j*rng.uniform(0, 2*np.pi, N))
    if t % 5 == 1: w = np.exp(2j*np.pi*rng.integers(0, 3, N)/3); xi = w*np.exp(0.05j*rng.normal(size=N)); eta = w*np.exp(0.05j*rng.normal(size=N))
    beta = rho*p; L = rho.sum(); Q = (rho**r).sum()
    A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum()
    R = np.sqrt(abs(B)**2 + abs(C)**2 + abs(D)**2)
    U = np.stack([np.ones(N), p*xi, p**2*xi*eta], 1)
    H = (rho[:, None, None]*U[:, :, None]*U.conj()[:, None, :]).sum(0); Hh = H/L
    nu = np.linalg.eigvalsh(Hh)
    th = 3**(1 - s)
    Mr = np.mean(np.clip(nu, 0, None)**r)**(1/r)
    worst['rot'] = min(worst['rot'], (th*Q**s - L*Mr)/L)
    Lam = Lam_nu(nu)
    worst['jen'] = min(worst['jen'], Mr - x**Lam)
    b = abs(Hh[1, 0]); cc = abs(Hh[2, 1]); d = abs(Hh[2, 0]); bet = np.hypot(b, cc); P = bet**2 + d**2
    qs = max(P - 1, -bet**2*d)
    worst['mono'] = min(worst['mono'], Lam_Pq(P, qs) - Lam)
    worst['rgn'] = min(worst['rgn'], 1 + d - bet**2, 1 - d)
    lams = np.linspace(0, 1, 1001)
    G = np.sqrt(1 + c**2*lams**2 - 2*c*lams*d) - c*bet*np.sqrt(1 - lams**2); k = np.argmin(G)
    worst['gain'] = min(worst['gain'], G[k] - (abs(A) - c*R)/L)
    worst['J'] = min(worst['J'], x**Lam_Pq(P, qs) - G[k])
print("min slacks (all >= ~-1e-9 required):", {k: f"{v:.3e}" for k, v in worst.items()})
# monotonicity of Lambda(P,q) on the valid domain {q >= P-1, 27 q^2 <= 4 P^3, P <= 3}
bad = 0; tot = 0
for t in range(20000):
    P = rng.uniform(0, 3); qmax = 2*(P/3)**1.5; qmin = max(P - 1, -qmax)
    if qmin > qmax: continue
    q1, q2 = np.sort(rng.uniform(qmin, qmax, 2))
    tot += 1; bad += Lam_Pq(P, q2) > Lam_Pq(P, q1) + 1e-10
    P2 = rng.uniform(P, 3)
    if q1 >= P2 - 1 and abs(q1) <= 2*(P2/3)**1.5: bad += Lam_Pq(P2, q1) < Lam_Pq(P, q1) - 1e-10
print(f"Lambda monotonicity violations: {bad} of ~{2*tot}")
