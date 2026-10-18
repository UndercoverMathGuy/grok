# Core library for the relations case at n=2 (class C, all interference on T={1,2}).
# Notation (as in BRIEF.md / Note): p odd prime, F={1..(p-1)/2}, kappa_l(d)=1-cos(2 pi l d/p), s in [1,3/2], r=1/s,
# allocation a>=0 on F (IF prices), ||a|| = (sum a^r)^s, Gamma_s(p) programme value, m(a)=min_d (K a)_d.
# c = c12 = 1-(2/3)^{s-1} (P12),  tau' = (3/2)^{s-1} (two-atom cap: v_l <= L_l <= tau' a_l in the relations case),
# note 1 + c*tau' = tau'.
import numpy as np, json, os
from scipy.optimize import linprog

REL3 = '/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3'

def consts(s):
    r = 1.0/s; c = 1-(2/3)**(s-1); tp = 1.5**(s-1); return r, c, tp

def kmat(p):
    """K[d-1, l-1] = kappa_l(d) for d,l in F (kappa is symmetric in d -> -d)."""
    F = (p-1)//2; j = np.arange(1, F+1)
    return 1-np.cos(2*np.pi*np.outer(j, j)/p)

def norm_r(a, s):
    a = np.asarray(a, float); return np.sum(a[a > 0]**(1/s))**s

def cover(K, a):
    return K @ a

def prog(p, s, restarts=8, seed=0, K=None, iters=80):
    """Programme Gamma_s(p) by reweighted LP (concave min of sum a^r over {Ka>=1}); returns (Gamma, y normalised sum y^r=1)."""
    r = 1/s; K = kmat(p) if K is None else K; F = K.shape[1]; rng = np.random.default_rng(seed); best = (np.inf, None)
    for t in range(restarts):
        w = rng.uniform(0.3, 1.7, F) if t else np.ones(F)
        y = None
        for it in range(iters):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(K.shape[0]), bounds=(0, None), method='highs')
            if res.status != 0: break
            y = res.x; wn = (y+1e-9*max(1, y.max()))**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        if y is None: continue
        y = y*(y > 1e-12*y.max()); Q = np.sum(y[y > 0]**r)
        if Q < best[0]: best = (Q, y.copy())
    Q, y = best; yn = y/Q**s
    return Q**(-s), yn

def load_support(p):
    d = json.load(open(os.path.join(REL3, 'supports.json')))
    if str(p) in d:
        e = d[str(p)]; F = (p-1)//2; y = np.zeros(F); y[np.array(e['S'])-1] = e['y']; return e['Gamma'], y
    fn = os.path.join(REL3, f'large_{p}.json')
    if os.path.exists(fn):
        D = json.load(open(fn)); run = max(D['runs'], key=lambda x: x['Gamma'])
        F = (p-1)//2; y = np.zeros(F); y[np.array(run['S'])-1] = run['y']; return run['Gamma'], y
    return None

def gamma_lower(p, s, y=None):
    """Exact (float) value of an explicit allocation: a rigorous LOWER bound on Gamma_s(p) up to rounding."""
    K = kmat(p); a = y if y is not None else load_support(p)[1]
    return (K @ a).min()/norm_r(a, s)

def osc_detect(p, v, th, delta):
    """(1/2) osc_w D_delta(w), D_delta(w) = phi(w)-phi(w-delta), phi(w)=sum_l v_l cos(2 pi l w/p + th_l). Exact over Z_p."""
    F = (p-1)//2; l = np.arange(1, F+1)
    w = np.arange(p)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])).sum(0)
    D = phi - np.roll(phi, delta)       # phi(w) - phi(w - delta)
    return 0.5*(D.max()-D.min())

def label_row_cost(p, A, v, th, s):
    """C(w) = sum_l (A_l + v_l cos(2 pi l w/p + th_l))_+^r  for all w in Z_p."""
    r = 1/s; F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p)
    z = A[:, None] + v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])
    return (np.maximum(z, 0)**r).sum(0)

def waterfill_excess(a, kap, c):
    """E0(a,kappa) = max_{0<=u<=a} [c sum u kappa - sqrt(sum u^2 kappa)] (exact, via water-filling dual)."""
    a = np.asarray(a, float); kap = np.asarray(kap, float)
    m = (a > 0) & (kap > 0)
    a = a[m]; kap = kap[m]
    if c*c*kap.sum() <= 1: return 0.0
    # omega_l = min(c, lam a_l), find lam with sum min(c,lam a)^2 kappa = 1 (monotone in lam)
    lo, hi = 0.0, 1.0
    f = lambda lam: (np.minimum(c, lam*a)**2*kap).sum()-1
    while f(hi) < 0: hi *= 2
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    lam = hi
    return float((a*kap*np.maximum(c-lam*a, 0)).sum())

def tail_bound(a, kap, c):
    """c * sum_{l not in H} a_l kappa_l with H = greedy top weights within kappa-budget 1/c^2 (fractional last)."""
    a = np.asarray(a, float); kap = np.asarray(kap, float)
    idx = np.argsort(-a); budget = 1/c**2; tot = 0.0
    for i in idx:
        if a[i] <= 0: break
        k = kap[i]
        if k <= 0: continue
        if k <= budget: budget -= k                      # omega_i = c: fully detected
        elif budget > 0:                                 # omega_i = c sqrt(budget/k): partially detected
            tot += a[i]*k*(c - c*np.sqrt(budget/k)); budget = 0.0
        else: tot += c*a[i]*k                            # omega_i = 0: undetected
    return float(tot)
