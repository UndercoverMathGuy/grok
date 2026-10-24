# Referee checks for R5/R7: closure table, the decoupling chain on random class-C nets (own code), and an
# adversarial maximisation of gamma within class C at small p (must not exceed the IF value 3^{1-s} B_p Gamma_s).
import numpy as np, sys, time
from scipy.optimize import minimize
from netlib import *
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts'); import prog
rng = np.random.default_rng(21)
# 1. closure table
def primes(N): return [q for q in range(3, N) if all(q % d for d in range(2, int(q**0.5) + 1))]
for b in [2.0, 2.1, 2.2, 2.25, 2.4, 2.5, 2.75]:
    s = 3/b; c = (1 - 3**(1 - s))/np.sqrt(3); ok = [q for q in primes(5000) if c**2*q/2 < (q - 1)/(q + 1)]
    last = max(ok); gap = [q for q in primes(5000) if q <= last and q not in ok]
    print(f"b={b}: c(s)={c:.5f} 1/c^2={1/c**2:.3f}; largest prime closed {last}; non-closed primes below it: {gap}")
print(f"Theorem A c=1/(2 sqrt3): largest prime closed {max(q for q in primes(200) if q/24 < (q-1)/(q+1))}")
# 2. chain on random class-C nets
def chain(U1, U2, W, fr, p, b):
    s = 3/b; r = 1/s; th = 3**(1 - s); c = (1 - th)/np.sqrt(3); Fn = (p - 1)//2
    f = table(U1, U2, W); g = gaps(f); h = g[:, :, 1:].min(); co = coeffs(f); at = atoms(U1, U2, W, fr, p)
    Q = np.array([(np.abs(at[fr == l])**r).sum() for l in range(1, Fn + 1)]); a = th*Q**s
    beta = np.array([co[(l, ())] for l in range(1, Fn + 1)])
    R = np.array([np.sqrt(sum(abs(co[(l, T)])**2 for T in [(1,), (2,), (1, 2)])) for l in range(1, Fn + 1)])
    K = np.array([[kap(l, d, p) for l in range(1, Fn + 1)] for d in range(1, p)])
    Wd = K @ a; ds = int(np.argmin(Wd)) + 1; k = K[ds - 1]
    cheap = np.abs(beta) > a + 1e-14
    mbar = 0.5*((g[:, :, ds].mean() - h) + (g[:, :, (-ds) % p].mean() - h))
    E = (R**2*k).sum()
    pay = (np.abs(beta) - a - c*R)[cheap].max() if cheap.any() else -1
    rhs = (a*k).sum() + c*(R*k)[cheap].sum() - mbar
    cond = c**2*k[cheap].sum() < (p - 1)/(p + 1)
    gamma = h/cost(U1, U2, W, b)**(3/b)
    return dict(h=h, W=Wd.min(), rhs=rhs, cond=cond, pay=pay, gamma=gamma, ncheap=int(cheap.sum()), E_ok=E <= (p+1)/(p-1)*mbar**2 + 1e-12)
for p in [11, 13, 17]:
    for b in [2.0, 2.5]:
        G = prog.solve(p, 3/b)[0]; gIF = 3**(1 - 3/b)*Bp(p)*G
        w = dict(chain=np.inf, dec=np.inf, pay=-np.inf, gam=-np.inf); nc = []; E_ok = True
        for t in range(30):
            fr_sel = [l for l in range(1, (p - 1)//2 + 1) if rng.random() < 0.6] or [1]
            U1, U2, W, fr = random_classC(p, rng, per=(1, 2), freqs=fr_sel)
            o = chain(U1, U2, W, fr, p, b)
            w['chain'] = min(w['chain'], o['rhs'] - o['h']); w['pay'] = max(w['pay'], o['pay']); E_ok &= o['E_ok']
            if o['cond']: w['dec'] = min(w['dec'], o['W'] - o['h'])
            w['gam'] = max(w['gam'], o['gamma']/gIF); nc.append(o['ncheap'])
        print(f"chain p={p} b={b}: min(rhs-h)={w['chain']:.3e}  min(W-h | cond)={w['dec']:.3e}  max payment excess={w['pay']:.3e}  E<=..mbar^2 all: {E_ok}  max gamma/gIF={w['gam']:.3f}  cheap counts {min(nc)}..{max(nc)}")
# 3. adversarial gamma maximisation within class C
def make(z, fr, p):
    m = len(fr); z = z.reshape(m, 6); a = np.arange(p)
    ang = 2*np.pi*np.outer(fr, a)/p
    U1 = z[:, [0]]*np.cos(ang + z[:, [1]]); U2 = z[:, [2]]*np.cos(ang + z[:, [3]]); W = z[:, [4]]*np.cos(ang + z[:, [5]])
    return U1, U2, W
def neg_gamma(z, fr, p, b, tau):
    U1, U2, W = make(z, fr, p); g = gaps(table(U1, U2, W))[:, :, 1:].ravel()
    nrm = (np.sum((p/2)*(z.reshape(-1, 6)[:, [0, 2, 4]]**2), 1)**(b/2)).sum()**(3/b)
    if tau is None: return -g.min()/nrm
    sc = np.abs(g).max() + 1e-300; gm = g.min()
    soft = gm - np.log(np.mean(np.exp(-tau*(g - gm)/sc)))*sc/tau
    return -soft/nrm
for (p, b, runs) in [(7, 2.0, 8), (11, 2.0, 8), (11, 2.5, 5)]:
    G = prog.solve(p, 3/b)[0]; gIF = 3**(1 - 3/b)*Bp(p)*G; best = -np.inf; t0 = time.time()
    for run in range(runs):
        Fn = (p - 1)//2
        if run % 2 == 0: fr = np.repeat(np.arange(1, Fn + 1), 1)      # one neuron per frequency: every feature cheap
        else: fr = np.repeat(np.arange(1, Fn + 1), 3)                  # room for IF blocks
        z = np.column_stack([rng.uniform(0.5, 1.5, len(fr)), rng.uniform(0, 2*np.pi, len(fr))] * 3).ravel()
        for tau in [20.0, 80.0, 300.0, 1000.0]:
            z = minimize(neg_gamma, z, args=(fr, p, b, tau), method='L-BFGS-B', options=dict(maxiter=300)).x
        val = -neg_gamma(z, fr, p, b, None)
        best = max(best, val)
        print(f"  adversarial p={p} b={b} run {run} (width {len(fr)}): gamma/gamma_IF = {val/gIF:.5f}   [{time.time()-t0:.0f}s]", flush=True)
    print(f"adversarial p={p} b={b}: best gamma/gamma_IF = {best/gIF:.5f} (must be <= 1)")
