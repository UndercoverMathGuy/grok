# Doubling lemma for compression of uniform progression blocks (b-independent statement, used here at s = 3/2).
#   P_n(x) = sum_{j<=n} (1 - cos j x) = n + 1/2 - sin((n+1/2)x)/(2 sin(x/2)),
#   P_{2q}(x) = 2 P_q(2x) + E_q(x),   E_q(x) = sum_{j<=q} [cos 2jx - cos (2j-1)x] = - sin(qx) sin((q+1/2)x) / cos(x/2).
# LEMMA (doubling).  q >= 84.  If B(x) = sum_l b_l (1 - cos l x) >= P_q(x) for all real x, then
#   B_2(x) := 2(1+eps) B(2x) + lam (1 - cos x),  eps = 1.01/q,  lam = 1.2 q,
# satisfies B_2 >= P_{2q} on R, and its cost obeys  rho_{2q}^r <= 2^{r-1} (1+eps)^r rho_q^r + lam^r/(2q)
# (rho_n := ||b||/n^s, ||b|| = (sum b^r)^s; frequency 1 is odd, the dilated frequencies 2l are even, so no merging is needed).
# Proof of B_2 >= P_{2q}: B_2 - P_{2q} >= 2 eps P_q(2x) + lam(1 - cos x) - E_q(x) =: R(x); show R >= 0 on [0, pi]  (*).
# This script: (1) checks the two closed forms; (2) checks (*) on fine grids (sanity check of the analytic proof);
# (3) builds B_2 explicitly from a certified base dominator and re-certifies it directly (independent cross-check);
# (4) runs the induction with interval arithmetic (outward rounding) for the base certificates q0 in {84, 110, 144};
# (5) prints the covered q-ranges (monotonicity: a dominator of P_q dominates P_q' for q' <= q, ratio rho_q (q/q')^s).
import numpy as np, json, sys
import mpmath as mp
from compress_certify3 import certify_fast
iv = mp.iv; iv.dps = 30

def P(n, x): return (1-np.cos(np.outer(x, np.arange(1, n+1)))).sum(1)
def Pc(n, x): return n + 0.5 - np.sin((n+0.5)*x)/(2*np.sin(x/2))
def Ec(q, x):
    # stable on [0, pi]: for x > pi/2 use E_q(pi - y) = sin(q y) cos((q+1/2) y) / sin(y/2) (removable singularity at x = pi)
    x = np.asarray(x, float); y = np.pi - x; out = np.empty_like(x); m = x <= np.pi/2
    out[m] = -np.sin(q*x[m])*np.sin((q+0.5)*x[m])/np.cos(x[m]/2)
    ym = np.maximum(y[~m], 1e-300); out[~m] = np.where(y[~m] > 1e-12, np.sin(q*ym)*np.cos((q+0.5)*ym)/np.sin(ym/2), 2.0*q)
    return out

rng = np.random.default_rng(0)
print("(1) closed forms:")
for q in [5, 84, 300]:
    x = rng.uniform(1e-3, np.pi-1e-3, 2000)
    E = np.array([np.sum(np.cos(2*np.arange(1, q+1)*t) - np.cos((2*np.arange(1, q+1)-1)*t)) for t in x])
    print(f"   q={q}: max|P_q - closed| = {np.abs(P(q, x)-Pc(q, x)).max():.1e}, max|P_2q - 2P_q(2x) - E_q| = "
          f"{np.abs(P(2*q, x) - 2*P(q, 2*x) - E).max():.1e}, max|E_q - closed| = {np.abs(E - Ec(q, x)).max():.1e}")
print("(2) R(x) = 2 eps P_q(2x) + lam(1-cos x) - E_q(x) >= 0 on fine grids (eps=1.01/q, lam=1.2q):")
for q in [84, 85, 110, 144, 168, 220, 288, 1000, 5000, 20000]:
    x = np.concatenate([np.linspace(1e-6, np.pi, 400001), np.geomspace(1e-7, 1e-1, 20001)])
    R = 2*(1.01/q)*Pc(q, 2*x) + 1.2*q*(1-np.cos(x)) - Ec(q, x)
    print(f"   q={q}: min R = {R.min():.4e} at x = {x[R.argmin()]:.4e}")

def load_cert(q0):
    for fn in ['compress_cert_big_s1.5.json', 'compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json']:
        try:
            d = json.load(open(fn))
        except FileNotFoundError:
            continue
        if str(q0) in d and d[str(q0)]['ok']:
            return d[str(q0)]['freqs'], [mp.mpf(v) for v in d[str(q0)]['bprime']]
    return None

s = mp.mpf(3)/2; r = 1/s; thr = (mp.mpf(2)/3)**(s-1)
print("(3) explicit doubled dominator, re-certified directly (base q0=84 -> 168):")
fr, bq = load_cert(84)
q0 = 84; eps = mp.mpf(1.01)/q0; lam = mp.mpf(1.2)*q0
L2 = max(2*max(fr), 1)
b2 = np.zeros(L2)
for l, bl in zip(fr, bq): b2[2*l-1] += float(2*(1+eps)*bl)
b2[0] += float(lam)
res, _ = certify_fast(2*q0, float(s), b2, 1e-9)
print(f"   direct certificate of B_2 for q=168: certified={res[0]} rho <= {res[1]:.4f}; doubling-lemma prediction "
      f"rho^r <= {mp.nstr(2**(r-1)*(1+eps)**r*(mp.mpf(res[1])*0 + (sum(b**r for b in bq))/q0) + lam**r/(2*q0), 6)} "
      f"i.e. rho <= {mp.nstr((2**(r-1)*(1+eps)**r*(sum(b**r for b in bq))/q0 + lam**r/(2*q0))**s, 6)}")

print("(4) induction along q_k = q0 2^k with outward rounding:")
ranges = []
for q0 in [84, 110, 144]:
    fr, bq = load_cert(q0)
    R = iv.mpf(0)
    for b in bq: R += iv.mpf(b)**(iv.mpf(2)/3)
    R = R/q0                                       # rho_{q0}^r
    rI = iv.mpf(2)/3; Rmax = R; qk = q0
    out = [(q0, R)]
    for k in range(1, 70):
        R = 2**(rI-1)*(1+iv.mpf(1.01)/qk)**rI*R + (iv.mpf(1.2)*qk)**rI/(2*qk)
        qk *= 2; out.append((qk, R))
        if R.b > Rmax.b: Rmax = R
    # tail: for k beyond, contraction: R_{k+1} <= a R_k + c with a <= 2^{r-1}(1+1.01/qk)^r < 1, c <= 0.5*1.2^r qk^{r-1}
    a = 2**(rI-1)*(1+iv.mpf(1.01)/qk)**rI; cst = (iv.mpf(1.2)*qk)**rI/(2*qk)
    tail_ok = (a*Rmax + cst).b <= Rmax.b
    rho_sup = (Rmax**s).b
    lo = q0*(rho_sup/thr)**(1/s)
    print(f"   q0={q0}: rho_q0 <= {mp.nstr((out[0][1]**s).b, 6)};  sup_k rho_(q0 2^k) <= {mp.nstr(rho_sup, 6)} (threshold {mp.nstr(thr, 6)});"
          f" tail invariant (R <= Rmax preserved for all later k): {tail_ok};  each q_k covers [{mp.nstr(lo/q0, 4)} q_k, q_k]")
    ranges.append((q0, float(rho_sup)))
print("(5) coverage of q >= 62 by the three chains (each q_k covers [q_k (rho_sup/thr)^{2/3}, q_k]):")
pts = sorted([(q0*2**k, q0, rho) for (q0, rho) in ranges for k in range(0, 40)])
gap = None; prev_hi = None
for (qk, q0, rho) in pts:
    lo = qk*(rho/float(thr))**(2/3)
    if prev_hi is not None and lo > prev_hi + 1: gap = (prev_hi, lo); break
    prev_hi = max(prev_hi or 0, qk)
lo0 = min(q0*(rho/float(thr))**(2/3) for (q0, rho) in ranges)
print(f"   union covers [{int(np.ceil(lo0))}, {pts[-1][0]:.3e}] without gaps: {gap is None}")
print("   (q in [17,144] is covered by the individual certificates in compress_certify_uniform.out / _warm.out / _big.out)")
Mq = [max(P(q, np.linspace(0, np.pi, 200001))) for q in [22, 23, 24]]
print(f"(6) Parseval single-block range at b=2: max_x P_q(x) = {Mq[0]:.4f} (q=22), {Mq[1]:.4f} (q=23), {Mq[2]:.4f} (q=24) vs 1/c^2 = {1/(1-(2/3)**0.5)**2:.4f}")
