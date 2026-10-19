# Re-evaluate the near-tight J and mono cases of v_r6_chain.py at 60 digits.
import numpy as np, mpmath as M
M.mp.dps = 60
rng = np.random.default_rng(9)
def Lam_hp(P, q):
    P = M.mpf(P); q = M.mpf(q)
    rts = M.polyroots([1, 0, -P, -q], maxsteps=400, extraprec=400)
    tot = M.mpf(0)
    for t in rts:
        v = 1 + M.re(t)
        if v > 0: tot += v*M.log(v)
    return tot/(3*M.log(3))
def Lam_nu(nu): nu = np.clip(nu, 0, None); return sum(v*np.log(v) for v in nu if v > 0)/(3*np.log(3))
def Lam_Pq(P, q): return Lam_nu(1 + np.sort(np.roots([1, 0, -P, -q]).real))
rec = []
for t in range(20000):
    N = int(rng.integers(1, 7)); s = rng.uniform(1, 1.5); r = 1/s; x = 3**(1 - s); c = (1 - x)/np.sqrt(3)
    rho = rng.exponential(size=N)**rng.uniform(0.3, 3); p = np.exp(1j*rng.uniform(0, 2*np.pi, N))
    if t % 3 == 0: p = np.exp(1j*0.2*rng.normal(size=N))
    xi = np.exp(1j*rng.uniform(0, 2*np.pi, N)); eta = np.exp(1j*rng.uniform(0, 2*np.pi, N))
    if t % 5 == 1: w = np.exp(2j*np.pi*rng.integers(0, 3, N)/3); xi = w*np.exp(0.05j*rng.normal(size=N)); eta = w*np.exp(0.05j*rng.normal(size=N))
    L = rho.sum()
    U = np.stack([np.ones(N), p*xi, p**2*xi*eta], 1)
    H = (rho[:, None, None]*U[:, :, None]*U.conj()[:, None, :]).sum(0); Hh = H/L
    b = abs(Hh[1, 0]); cc = abs(Hh[2, 1]); d = abs(Hh[2, 0]); bet = np.hypot(b, cc); P = bet**2 + d**2
    qs = max(P - 1, -bet**2*d)
    lams = np.linspace(0, 1, 1001)
    G = np.sqrt(1 + c**2*lams**2 - 2*c*lams*d) - c*bet*np.sqrt(1 - lams**2); k = np.argmin(G)
    J = x**Lam_Pq(P, qs) - G[k]
    if J < 1e-6: rec.append((J, P, qs, x, c, bet, d, lams[k], N))
rec.sort()
for (J, P, qs, x, c, bet, d, lam, N) in rec[:6]:
    # exact-ish: P and q* recomputed in mp from the float bet, d
    Pm = M.mpf(bet)**2 + M.mpf(d)**2; qm = max(Pm - 1, -M.mpf(bet)**2*M.mpf(d))
    Gm = M.sqrt(1 + M.mpf(c)**2*M.mpf(lam)**2 - 2*M.mpf(c)*M.mpf(lam)*M.mpf(d)) - M.mpf(c)*M.mpf(bet)*M.sqrt(1 - M.mpf(lam)**2)
    Jm = M.mpf(x)**Lam_hp(Pm, qm) - Gm
    print(f"float J={J:.3e}  hp J={M.nstr(Jm, 5)}  P={P:.12f} q*={qs:.12f} N={N} beta={bet:.6f} delta={d:.6f}")
