# Independent referee re-certification of the reduced 2-D inequality (J) of R6 at s = 3/2:
#   for (beta, delta) in Rgn = {0<=delta<=1, beta^2 <= 1+delta}, exists lam in [0,1]:
#   G(beta,delta,lam) = sqrt(1 + c^2 lam^2 - 2 c lam delta) - c beta sqrt(1-lam^2) <= x0^{Lambda(P, q*)},
#   P = beta^2 + delta^2, q* = max(P-1, -beta^2 delta), c = (sqrt3-1)/3, x0 = 3^{-1/2}.
# Differences from the agent's script: decimal (non-dyadic) grid, P capped at 3 (valid spectra have P <= 3),
# trigonometric root enclosures (not sign changes), lambda from a fine grid search, own region-A/B grid checks.
import sys, time
from fractions import Fraction as Fr
import numpy as np
from mpmath import mp, iv, mpf
mp.dps = 50; iv.dps = 40
SQ3 = iv.sqrt(3); C0 = (SQ3 - 1)/3; LN3 = iv.log(3); cf = (3**0.5 - 1)/3
def I(fr): return iv.mpf(fr.numerator)/iv.mpf(fr.denominator)
def Lam_upper(P, q):            # P, q exact Fractions with real roots; returns rigorous upper bound of Lambda
    Pi = I(P); qi = I(q)
    a = 2*iv.sqrt(Pi/3)
    arg = (3*qi/(2*Pi))*iv.sqrt(3/Pi)
    lo = max(mpf(arg.a), mpf(-1)); hi = min(mpf(arg.b), mpf(1))
    if lo > hi: return None
    th = iv.mpf([mp.acos(hi) - mpf(10)**-45, mp.acos(lo) + mpf(10)**-45])
    tot = iv.mpf(0)
    for k in range(3):
        t = a*iv.cos(th/3 - 2*iv.pi*k/3)
        nlo = max(mpf((1 + t).a), mpf(0)); nhi = mpf((1 + t).b)
        if nhi <= 0: continue
        vals = []
        for v in (nlo, nhi):
            vals.append(mpf(0) if v <= 0 else mpf((iv.mpf(v)*iv.log(iv.mpf(v))).b))
        tot += iv.mpf(max(vals))
    return mpf((tot/(3*LN3)).b)
def G_upper(b, d, lam):
    b = I(b); d = I(d); lam = I(lam)
    return mpf((iv.sqrt(1 + C0**2*lam**2 - 2*C0*lam*d) - C0*b*iv.sqrt(1 - lam**2)).b)
LS = np.linspace(0, 1, 2001)
def lam_choice(b, d):
    g = np.sqrt(1 + cf**2*LS**2 - 2*cf*LS*d) - cf*b*np.sqrt(np.clip(1 - LS**2, 0, None))
    return Fr(int(np.argmin(g)), 2000)
PA = Fr(1, 25); PB = Fr(271, 100)
st = dict(ok=0, A=0, B=0, out=0, split=0, fail=0); mm = [mpf(9)]
def box(b0, b1, d0, d1, depth):
    if b0*b0 > 1 + d1: st['out'] += 1; return
    Phi = b1*b1 + d1*d1; Plo = b0*b0 + d0*d0
    if Phi <= PA: st['A'] += 1; return
    if Plo >= PB: st['B'] += 1; return
    Ph = min(Phi, Fr(3))
    qlo = max(Plo - 1, -b1*b1*d1); qq = max(qlo, Ph - 1)
    lam = lam_choice(float(b0), float(d0))
    Lu = Lam_upper(Ph, qq)
    if Lu is not None:
        marg = mpf(iv.exp(-(LN3/2)*iv.mpf(Lu)).a) - G_upper(b0, d0, lam)
        if marg > 0:
            st['ok'] += 1; mm[0] = min(mm[0], marg); return
    if depth >= 16: st['fail'] += 1; print('FAIL', float(b0), float(b1), float(d0), float(d1)); return
    st['split'] += 1; bm = (b0 + b1)/2; dm = (d0 + d1)/2
    for x in ((b0, bm), (bm, b1)):
        for y in ((d0, dm), (dm, d1)):
            box(x[0], x[1], y[0], y[1], depth + 1)
t0 = time.time()
for N in (10, 23):
    st.update(ok=0, A=0, B=0, out=0, split=0, fail=0); mm[0] = mpf(9)
    bs = [Fr(3, 2)*Fr(k, N) for k in range(N + 1)]; ds = [Fr(k, N) for k in range(N + 1)]
    for i in range(N):
        for j in range(N): box(bs[i], bs[i+1], ds[j], ds[j+1], 0)
    print(f"grid N={N}: {st}; min certified margin {float(mm[0]):.3e}; time {time.time()-t0:.0f}s", flush=True)
# Region A / B: dense float grid check of the true inequality with the analytic lambda choices (sanity of the hand proofs).
def Lam_float(P, q):
    t = np.roots([1, 0, -P, -q]).real; nu = np.clip(1 + t, 0, None)
    return sum(v*np.log(v) for v in nu if v > 0)/(3*np.log(3))
x0 = 3**-0.5; worstA = np.inf; worstB = np.inf
for beta in np.linspace(0, 0.2, 201):
    for delta in np.linspace(0, 0.2, 201):
        P = beta**2 + delta**2
        if P > 1/25 or P == 0: continue
        lam = delta/(cf + 2*beta)
        G = np.sqrt(1 + cf**2*lam**2 - 2*cf*lam*delta) - cf*beta*np.sqrt(1 - lam**2)
        worstA = min(worstA, (x0**Lam_float(P, max(P - 1, -beta**2*delta)) - G)/P)
for delta in np.linspace(0.85, 1, 301):
    for beta in np.linspace(1.3, np.sqrt(1 + delta), 301):
        P = beta**2 + delta**2
        if P < 2.71 or beta**2 > 1 + delta + 1e-15: continue
        lam = 1/np.sqrt(3)
        G = np.sqrt(1 + cf**2*lam**2 - 2*cf*lam*delta) - cf*beta*np.sqrt(1 - lam**2)
        worstB = min(worstB, x0**Lam_float(min(P, 3), min(P, 3) - 1) - G)
print(f"region A float grid: min (x0^Lam - G)/P = {worstA:.4e} (>=0);  region B float grid: min (x0^Lam - G) = {worstB:.4e} (>=0)")
