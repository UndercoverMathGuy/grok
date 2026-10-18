# Referee: independent checks of R6's constant Xi.
# (a) high-precision (non-interval) evaluation of Xi_torus(s) via 2F1 and via quadrature, sup over t on a fine grid;
# (b) an independently coded interval certificate (mpmath.iv), different box grid and truncation;
# (c) iv.gamma enclosure sanity (Gamma(r+1/2) is NOT monotone on r in [2/3,1]);
# (d) brute-force Xi_p at finite p (p-point averages, sup over t and theta) vs the certified bound.
import mpmath as mp, numpy as np, sys
mp.mp.dps = 25
def Etor(t, r):  # E_U (1 + t cos U)^r, 0<=t<=1
    return mp.hyp2f1(-r/2, (1-r)/2, 1, t*t)
def xi_tor(s, nt=400):
    r = 1/s; c = 1-(mp.mpf(2)/3)**(s-1)
    best = (0, None)
    for k in range(nt+1):
        t = mp.mpf(k)/nt; val = Etor(t, r)/(1-c*t)**r
        if val > best[0]: best = (val, t)
    # refine
    t0 = best[1]; f = lambda t: -Etor(t, r)/(1-c*t)**r
    lo, hi = max(0, t0-mp.mpf(1)/nt), min(1, t0+mp.mpf(1)/nt)
    for _ in range(60):
        m1 = lo+(hi-lo)/3; m2 = hi-(hi-lo)/3
        if f(m1) < f(m2): hi = m2
        else: lo = m1
    tb = (lo+hi)/2; vb = -f(tb)
    tail = Etor(mp.mpf(1), r)/(1-c)**r
    return max(vb, tail), tb, tail
print("(a) torus Xi (2F1), cross-check with quad at s=1.5, t=0.825:",
      mp.nstr(Etor(mp.mpf('0.825'), mp.mpf(2)/3), 15), mp.nstr(mp.quad(lambda U: (1+mp.mpf('0.825')*mp.cos(U))**(mp.mpf(2)/3), [0, mp.pi])/mp.pi, 15))
worst = 0
for s in [1+mp.mpf(k)/40 for k in range(0, 21)]:
    X, tb, tail = xi_tor(s)
    worst = max(worst, X)
    if int(round((s-1)*40)) % 4 == 0: print(f"   s={mp.nstr(s,4)}: Xi_torus = {mp.nstr(X, 8)} (t*={mp.nstr(tb,5)}, t>=1 branch {mp.nstr(tail,6)}),  Xi^s = {mp.nstr(X**s, 7)}")
print("   max over s-grid:", mp.nstr(worst, 8))
# (c)
iv = mp.iv; iv.dps = 30
G = iv.gamma(iv.mpf([mp.mpf(2)/3+mp.mpf(1)/2, mp.mpf(3)/2]))
print("(c) iv.gamma([7/6, 3/2]) =", G, " true min Gamma = 0.8856031944 inside:", G.a <= mp.mpf('0.8856031944'))
# (b) independent interval certificate
def cert(ns=40, nt=1000, N=40):
    two3 = iv.mpf(2)/3; worst = mp.mpf(0)
    bc = [iv.mpf(mp.binomial(2*k, k))/iv.mpf(4)**k for k in range(N+1)]
    for i in range(ns):
        s1 = iv.mpf(1)+iv.mpf(i)/(2*ns); s2 = iv.mpf(1)+iv.mpf(i+1)/(2*ns)
        rmax = 1/s1
        # |binom(rmax, 2k)| as intervals
        co = []; bb = iv.mpf(1)
        for j in range(2*N):
            bb = bb*(rmax-j)/(j+1)
            if (j+1) % 2 == 0: co.append(abs(bb)*bc[(j+1)//2])
        c2 = 1-two3**(s2-1)
        for k in range(nt):
            t1 = iv.mpf(k)/nt; t2 = iv.mpf(k+1)/nt
            E = iv.mpf(1); tt = t1*t1; pw = tt
            for ck in co:
                E -= ck*pw; pw *= tt
            ub = (E/(1-c2*t2)**rmax).b
            if ub > worst: worst = ub
        # t >= 1 branch with r interval
        rI = iv.mpf([(1/s2).a, (1/s1).b])
        c0 = 2**rI*iv.gamma(rI+iv.mpf(1)/2)/(iv.sqrt(iv.pi)*iv.gamma(rI+1))
        tb = (c0*(iv.mpf(3)/2)**((s2-1)/s1)).b
        if tb > worst: worst = tb
    return worst
if len(sys.argv) > 1 and sys.argv[1] == 'cert':
    print("(b) independent interval certificate: sup Xi_torus over s in [1,1.5] <=", mp.nstr(cert(), 10))
# (d) finite p brute force
def xi_p(p, s, nt=241, nth=64):
    r = 1/s; c = 1-(2/3)**(s-1); k = np.arange(p)
    best = 0
    for t in np.concatenate([np.linspace(0, 1, nt), np.linspace(1, 3, 41)]):
        pi_t = 1-c*t if t <= 1 else t*(1-c)
        th = np.linspace(0, 2*np.pi/p, nth)
        vals = (np.maximum(1+t*np.cos(th[:, None]+2*np.pi*k[None, :]/p), 0)**r).mean(1)/pi_t**r
        best = max(best, vals.max())
    return best
for p in [5, 7, 11, 61, 67]:
    print(f"(d) p={p}: brute-force Xi_p(1.5) ~ {xi_p(p, 1.5):.5f}")
