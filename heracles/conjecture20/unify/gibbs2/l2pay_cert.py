# Computer-assisted proof of the SHARP L2 PAYMENT at n = 2:
#     |A| - 3^{1-s} Q^s  <=  ((1 - 3^{1-s})/sqrt3) * sqrt(|B|^2 + |C|^2 + |D|^2)      for all atoms, all s in [1, 3/2].
# After the reductions of the report (twisted matrix + Rotfel'd, Jensen, gain step, monotonicity in (P,q), worst case
# b = c', concavity in x = 3^{1-s}), it suffices to prove, with c = (sqrt3-1)/3, x0 = 1/sqrt3:
#   (J)  for every (beta, delta) in R = {0 <= delta <= 1, 0 <= beta <= sqrt(1+delta)} there is lam in [0,1] with
#        G(beta,delta,lam) := sqrt(1 + c^2 lam^2 - 2 c lam delta) - c beta sqrt(1 - lam^2)  <=  x0^{Lambda(P, q*)},
#        P = beta^2 + delta^2, q* = max(P - 1, -beta^2 delta),
#        Lambda(P,q) = sum_k (nu_k/3) log_3 nu_k,  nu_k = 1 + t_k,  t_k the roots of t^3 - P t - q.
# Regions:  A = {P <= 1/25} (analytic, checked below),  B = {P >= 271/100} (analytic, 1-D in sigma, checked below),
#           C = rest: adaptive boxes; G is decreasing in beta and delta, so G(box) <= G(beta_lo, delta_lo, lam);
#           Lambda(box) <= Lambda(Phat, max(q_lo, Phat - 1)) with Phat >= P_hi, q_lo = max(P_lo - 1, -beta_hi^2 delta_hi)
#           (path argument in the report).  Roots are isolated with exact rational sign tests; logs/exps/sqrts use
#           mpmath interval arithmetic.
import sys, time
from fractions import Fraction as Fr
from mpmath import mp, iv, mpf
mp.dps = 40; iv.dps = 30
SQ3 = iv.sqrt(3); C0 = (SQ3 - 1)/3; LN3 = iv.log(3)
c_f = float((3**0.5 - 1)/3)
def LO(x): return mpf(x.a)
def mpf2fr(x):
    x = mpf(x)
    if x == 0: return Fr(0)
    sgn, man, e, bc = x._mpf_
    v = Fr(int(man)) * (Fr(2)**int(e))
    return -v if sgn else v
def HI(x): return mpf(x.b)

def G_upper(b, d, lam):
    b = iv.mpf(b); d = iv.mpf(d); lam = iv.mpf(lam)
    return HI(iv.sqrt(1 + C0**2*lam**2 - 2*C0*lam*d) - C0*b*iv.sqrt(1 - lam**2))

def best_lam(b, d):
    import numpy as np
    ls = np.linspace(0, 1, 401)
    g = np.sqrt(1 + c_f**2*ls**2 - 2*c_f*ls*d) - c_f*b*np.sqrt(np.clip(1 - ls**2, 0, None))
    k = int(np.argmin(g)); lo = ls[max(k - 1, 0)]; hi = ls[min(k + 1, 400)]
    for _ in range(40):
        m1 = lo + (hi - lo)/3; m2 = hi - (hi - lo)/3
        f1 = (1 + c_f**2*m1**2 - 2*c_f*m1*d)**0.5 - c_f*b*(1 - m1**2)**0.5
        f2 = (1 + c_f**2*m2**2 - 2*c_f*m2*d)**0.5 - c_f*b*(1 - m2**2)**0.5
        if f1 < f2: hi = m2
        else: lo = m1
    return Fr((lo + hi)/2).limit_denominator(2**20)

def cubic_sign(t, P, q):            # exact sign of t^3 - P t - q  (t, P, q Fractions)
    v = t*t*t - P*t - q
    return (v > 0) - (v < 0)

def root_enclosures(P, q):
    """rigorous enclosures of the three real roots of t^3 - P t - q: isolate the root farthest from the other two
    by an exact rational sign change, then the other two are (-t1 +- sqrt(4P - 3 t1^2))/2 (interval arithmetic)."""
    Pm = mpf(P.numerator)/P.denominator; qm = mpf(q.numerator)/q.denominator
    a = 2*mp.sqrt(Pm/3)
    arg = (3*qm/(2*Pm))*mp.sqrt(3/Pm)
    arg = max(min(arg, mpf(1)), mpf(-1))
    th = mp.acos(arg)/3
    rts = sorted([a*mp.cos(th - 2*mp.pi*k/3) for k in range(3)])
    gaps = [min(abs(rts[k] - rts[j]) for j in range(3) if j != k) for k in range(3)]
    k = max(range(3), key=lambda i: gaps[i])
    rf = mpf2fr(rts[k]).limit_denominator(10**18)
    enc = None
    if cubic_sign(rf, P, q) == 0: enc = (rf, rf)
    else:
        for eps in [Fr(1, 10**14), Fr(1, 10**11), Fr(1, 10**8)]:
            lo, hi = rf - eps, rf + eps
            if cubic_sign(lo, P, q)*cubic_sign(hi, P, q) < 0: enc = (lo, hi); break
    if enc is None: return None
    t1 = iv.mpf([mpf(enc[0].numerator)/enc[0].denominator, mpf(enc[1].numerator)/enc[1].denominator])
    disc = 4*iv.mpf(Pm) - 3*t1**2
    if HI(disc) < 0: return None
    disc = iv.mpf([max(LO(disc), mpf(0)), HI(disc)])
    sq = iv.sqrt(disc)
    u = (-t1 + sq)/2; v = (-t1 - sq)/2
    return [(LO(t1), HI(t1)), (LO(u), HI(u)), (LO(v), HI(v))]

def h_upper(lo, hi):                # upper bound of nu log nu on [lo, hi] (convex => endpoint max); 0 log 0 = 0
    vals = []
    for v in (lo, hi):
        if v <= 0: vals.append(mpf(0)); continue
        x = iv.mpf(v)
        vals.append(HI(x*iv.log(x)))
    return max(vals)

def Lambda_upper(P, q):
    enc = root_enclosures(P, q)
    if enc is None: return None
    s = mpf(0)
    for (lo, hi) in enc:
        s += h_upper(max(1 + lo, mpf(0)), 1 + hi)
    return HI(iv.mpf(s)/(3*LN3))

def rhs_lower(Lam_up):
    return LO(iv.exp(-(LN3/2)*iv.mpf(Lam_up)))

PA = Fr(1, 25); PB = Fr(271, 100)
stats = dict(ok=0, A=0, B=0, out=0, split=0, fail=0); minmargin = [mpf(10)]
def box(b0, b1, d0, d1, depth):
    if b0*b0 > 1 + d1: stats['out'] += 1; return
    Phi = b1*b1 + d1*d1; Plo = b0*b0 + d0*d0
    if Phi <= PA: stats['A'] += 1; return
    if Plo >= PB: stats['B'] += 1; return
    qlo = max(Plo - 1, -b1*b1*d1)
    qq = max(qlo, Phi - 1)
    lam = best_lam(float(b0), float(d0))
    Lu = Lambda_upper(Phi, qq)
    ok = False
    if Lu is not None:
        marg = rhs_lower(Lu) - G_upper(mpf(b0.numerator)/b0.denominator, mpf(d0.numerator)/d0.denominator,
                                       mpf(lam.numerator)/lam.denominator)
        if marg > 0:
            ok = True; stats['ok'] += 1; minmargin[0] = min(minmargin[0], marg)
            if stats['ok'] % 2000 == 0: print('progress', stats, float(minmargin[0]), flush=True)
    if not ok:
        if depth >= 14: stats['fail'] += 1; print("FAIL box", float(b0), float(b1), float(d0), float(d1)); return
        stats['split'] += 1
        bm = (b0 + b1)/2; dm = (d0 + d1)/2
        for (x0, x1) in ((b0, bm), (bm, b1)):
            for (y0, y1) in ((d0, dm), (dm, d1)):
                box(x0, x1, y0, y1, depth + 1)

def regionA_check():
    # analytic bound of the report on P <= 1/25:  margin >= beta*(c - beta/6 - beta*delta/12 - kA*beta)
    #   + delta^2*(c/(2(c+2beta)) - 1/6 - kA),  kA = PA/36 + tauA^3/(60(1-tauA)),  tauA = 2 sqrt(PA/3);  both brackets > 0.
    PAi = iv.mpf(1)/25; tau = 2*iv.sqrt(PAi/3); kA = PAi/36 + tau**3/(60*(1 - tau))
    bmax = iv.sqrt(PAi)
    br1 = C0 - bmax/6 - bmax*bmax/12 - kA*bmax        # beta <= 1/5, delta <= 1/5
    br2 = C0/(2*(C0 + 2*bmax)) - iv.mpf(1)/6 - kA     # c/(2(c+2beta)) decreasing in beta
    return LO(br1), LO(br2), HI(kA)

def regionB_check():
    # 1-D bound on sigma in (0, 1/10]:  margin >= sigma*[ (x0/6)(1 + ln(3/sigma_B) - sigma_B/3) - Gbar'_max ],
    # Gbar(sigma) = sqrt((1 - c/sqrt3)^2 + 2 c sigma/sqrt3) - c sqrt(2/3) sqrt(2 - 3 sigma + sigma^2),
    # Gbar'(sigma) = (c/sqrt3)/sqrt((1-c/sqrt3)^2 + 2c sigma/sqrt3) + c sqrt(2/3)(3 - 2 sigma)/(2 sqrt(2 - 3 sigma + sigma^2)).
    sB = iv.mpf(1)/10; X0 = 1/SQ3
    lead = (X0/6)*(1 + iv.log(3/sB) - sB/3)
    sig = iv.mpf([0, 0.1])
    t1 = (C0/SQ3)/iv.sqrt((1 - C0/SQ3)**2)                       # first term max at sigma = 0 (decreasing)
    t2 = C0*iv.sqrt(iv.mpf(2)/3)*(3 - 2*sig)/(2*iv.sqrt(2 - 3*sig + sig**2))   # interval enclosure on [0, 0.1]
    Gmax = HI(t1 + t2)
    G0 = (1 - C0/SQ3) - C0*iv.sqrt(iv.mpf(2)/3)*iv.sqrt(2)       # = 1 - sqrt3 c = x0
    return LO(lead), Gmax, (G0 - X0)

if __name__ == '__main__':
    t0 = time.time()
    a1, a2, kA = regionA_check()
    print(f"Region A (P <= 1/25): bracket1 >= {float(a1):.6f} (>0), bracket2 >= {float(a2):.6f} (>0), kA <= {float(kA):.6f}")
    lead, Gmax, g0 = regionB_check()
    print(f"Region B (P >= 2.71): (x0/6)(1+ln(3/sigma_B)-sigma_B/3) >= {float(lead):.6f}  >  Gbar'_max <= {float(Gmax):.6f}: {lead > Gmax};  Gbar(0) - x0 in {g0}")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    bs = [Fr(3, 2)*k/N for k in range(N + 1)]; ds = [Fr(k, N) for k in range(N + 1)]
    for i in range(N):
        for j in range(N):
            box(bs[i], bs[i + 1], ds[j], ds[j + 1], 0)
    print(f"Region C boxes: {stats};  min certified margin {float(minmargin[0]):.3e};  time {time.time() - t0:.0f}s")
    print("CERTIFIED" if stats['fail'] == 0 and a1 > 0 and a2 > 0 and lead > Gmax else "NOT CERTIFIED")
