# High-precision re-check of reduction step (d): Lambda(P, q_matrix) <= Lambda(P, q*) with q* = max(P-1, -beta^2 delta).
# Float run (l2pay_chain_check.py) showed a -1.2e-7 slack; here every case with float slack < 1e-6 is recomputed at 50 digits.
import numpy as np
from mpmath import mp, mpf, polyroots, log, re
mp.dps = 50
rng = np.random.default_rng(5)
def Lam_hp(P, q):
    t = polyroots([1, 0, -P, -q], maxsteps=200, extraprec=200)
    nu = [1 + re(x) for x in t]
    s = mpf(0)
    for v in nu:
        if v > 0: s += v*log(v)
    return s/(3*log(3))
worst_f = np.inf; worst_hp = mpf(10); n_close = 0
for trial in range(20000):
    N = int(rng.integers(1, 7))
    rho = rng.exponential(size=N)*(rng.random(N) < 0.9) + 1e-9
    ph = rng.normal(size=N)*rng.choice([0.05, 0.5, 3.0])
    a1 = rng.uniform(0, 2*np.pi, N); a2 = rng.uniform(0, 2*np.pi, N)
    if rng.random() < 0.3: a1 = 2*np.pi*np.arange(N)/N + 0.1*rng.normal(size=N); a2 = a1 + 0.1*rng.normal(size=N)
    p = np.exp(1j*ph); xi = np.exp(1j*a1); et = np.exp(1j*a2); L = rho.sum()
    x = (rho*p*xi).sum()/L; y = (rho*p*et).sum()/L; z = (rho*p**2*xi*et).sum()/L    # B/L, C/L, D'/L
    b, c, d = abs(x), abs(y), abs(z); P = b*b + c*c + d*d; beta = np.hypot(b, c)
    qm = 2*np.real(x*y*np.conj(z)); qs = max(P - 1, -beta**2*d)
    tf = np.roots([1, 0, -P, -qm]).real; nuf = np.clip(1 + tf, 1e-300, None)
    ts = np.roots([1, 0, -P, -qs]).real; nus = np.clip(1 + ts, 1e-300, None)
    df = np.sum(nus*np.log(nus))/np.log(3)/3 - np.sum(nuf*np.log(nuf))/np.log(3)/3
    worst_f = min(worst_f, df)
    if df < 1e-6:
        n_close += 1
        dh = Lam_hp(mpf(P), mpf(qs)) - Lam_hp(mpf(P), mpf(qm))
        worst_hp = min(worst_hp, dh)
        if qm < qs - 1e-15: print("q_matrix below q* ?!", qm, qs)
print(f"float min slack {worst_f:.3e};  {n_close} near-tight cases recomputed at 50 digits: min slack {mp.nstr(worst_hp, 5)}")
