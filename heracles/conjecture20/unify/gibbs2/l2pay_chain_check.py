# Sanity check of every step of the reduction for the sharp L2 payment (n = 2), on random atom systems:
#  (a) cost:      3^{1-s} Q^s >= L * M_{1/s}(nu),  nu = spec(Hhat), Hhat = twisted correlation matrix (entries B/L, C/L, D'/L)
#  (b) Jensen:    M_{1/s}(nu) >= x^Lambda,  x = 3^{1-s},  Lambda = sum (nu/3) log_3 nu
#  (c) gain:      |A| - c(x) R <= L * G_x(lam),  lam = (b, c', .)-aligned unit vector with any lam3
#  (d) worst q:   Lambda(P, q) <= Lambda(P, q*),  q* = max(P-1, -beta^2 delta)
#  (e) final:     |A| - c(x) R <= 3^{1-s} Q^s   (the theorem)
#  plus the divided-difference monotonicity claims: Lambda decreasing in q, increasing in P.
import numpy as np
rng = np.random.default_rng(5)
def atoms(N):
    rho = rng.exponential(size=N)*(rng.random(N) < 0.9) + 1e-9
    ph = rng.normal(size=N)*rng.choice([0.05, 0.5, 3.0])
    a1 = rng.uniform(0, 2*np.pi, N); a2 = rng.uniform(0, 2*np.pi, N)
    if rng.random() < 0.3: a1 = 2*np.pi*np.arange(N)/N + 0.1*rng.normal(size=N); a2 = a1 + 0.1*rng.normal(size=N)
    return rho*np.exp(1j*ph), np.exp(1j*a1), np.exp(1j*a2)
def Lam(nu):
    nu = np.clip(nu, 1e-300, None); return float(np.sum(nu/3*np.log(nu))/np.log(3))
def spec_Pq(P, q):
    return 1 + np.sort(np.roots([1, 0, -P, -q]).real)[::-1]
worst = dict(a=np.inf, b=np.inf, c=np.inf, d=np.inf, e=np.inf)
for t in range(20000):
    N = int(rng.integers(1, 7)); b_, xi, et = atoms(N); s = rng.uniform(1, 1.5); r = 1/s; x = 3**(1 - s); c = (1 - x)/np.sqrt(3)
    A = b_.sum(); B = (b_*xi).sum(); C = (b_*et).sum(); D = (b_*xi*et).sum()
    rho = np.abs(b_); p = b_/rho; L = rho.sum(); Q = (rho**r).sum()
    xx = p*xi; yy = p**2*xi*et
    U = np.stack([np.ones(N), xx, yy])                  # columns u_i
    H = (U*rho) @ U.conj().T; Hh = H/L
    nu = np.clip(np.linalg.eigvalsh(Hh), 0, None)
    Mr = (np.mean(nu**r))**s
    worst['a'] = min(worst['a'], x*Q**s - L*Mr)
    worst['b'] = min(worst['b'], Mr - x**Lam(nu))
    bb, cc, dd = abs(B)/L, abs(C)/L, abs(H[2, 0])/L
    beta = np.hypot(bb, cc)
    R = np.sqrt(abs(B)**2 + abs(C)**2 + abs(D)**2)
    for lam3 in [0, 0.3, 1/np.sqrt(3), 0.9, rng.random()]:
        G = np.sqrt(1 + c*c*lam3**2 - 2*c*lam3*dd) - c*beta*np.sqrt(1 - lam3**2)
        worst['c'] = min(worst['c'], L*G - (abs(A) - c*R))
    P = bb**2 + cc**2 + dd**2
    qmat = 2*np.real(Hh[1, 0]*Hh[2, 1]*np.conj(Hh[2, 0]))
    qstar = max(P - 1, -beta**2*dd)
    worst['d'] = min(worst['d'], Lam(spec_Pq(P, qstar)) - Lam(nu))
    worst['e'] = min(worst['e'], x*Q**s - (abs(A) - c*R))
print("min slack of each step over 20000 random atom systems (all must be >= -1e-9):")
for k, v in worst.items(): print(f"   ({k}) {v:.3e}")
# monotonicity of Lambda in q (decreasing) and P (increasing), and along det=0 (increasing in P)
bad = 0
for t in range(20000):
    P = rng.uniform(0.01, 3.0); qmax = 2*(P/3)**1.5; qmin = max(-qmax, P - 1)
    if qmin >= qmax: continue
    q1, q2 = np.sort(rng.uniform(qmin, qmax, 2))
    if Lam(spec_Pq(P, q2)) > Lam(spec_Pq(P, q1)) + 1e-12: bad += 1
    q = rng.uniform(qmin, qmax); P2 = P + rng.uniform(0, 0.5)
    if q >= P2 - 1 and Lam(spec_Pq(P2, q)) < Lam(spec_Pq(P, q)) - 1e-12: bad += 1
print("monotonicity violations (Lambda decreasing in q, increasing in P):", bad)
