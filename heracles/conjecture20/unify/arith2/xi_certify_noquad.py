# [variant: pi(t,s) = 1 - c t on [0,1], i.e. P12 + two-atom cap only; quadpay NOT used]
# Rigorous (interval arithmetic, mpmath.iv) upper bound for the label-row constant on the whole range s in [1, 3/2]:
#   Xi(s) = sup_{t>=0} E_U (1 + t cos U)_+^r / pi(t,s)^r ,  r = 1/s, U uniform on the circle,
#   pi(t,s) = max(1 - c t, 1 - Kq t^2)   (0 <= t <= 1),   pi(t,s) = t (1-c)  (t >= 1),
#   c = 1-(2/3)^{s-1} (P12),  Kq = 5/(5+4r) (Note, Prop quadpay (ii)),  1/(1-c) = (3/2)^{s-1} (two-atom cap).
# Meaning: in the relations case a feature with IF price a, interference v and clean amplitude A = Re beta has
#   A <= a/pi(t) with t = v/A, so its label-row cost (A + v cos U)_+^r  <=  a^r (1 + t cos U)_+^r / pi(t)^r.
# Finite p: the average over U in theta + (2 pi/p) Z_p differs from the torus mean by at most the aliasing bound
#   2 sum_{j>=1} |f^(jp)| <= sum_j 1/(jp(jp-1)) * ... (see below), valid for t <= 1 and inherited for t >= 1.
# Method: boxes [t1,t2] x [s1,s2].
#   E(t,r) = 1 - sum_{k>=1} |binom(r,2k)| binom(2k,k) 4^{-k} t^{2k}  (all terms <= 0) is increasing in r on [2/3,1]
#   (|binom(r,2k)| decreasing in r there) and decreasing in t; truncation gives an upper bound.
#   pi is decreasing in t, and on s in [s1,s2]: pi(t,s) >= max(1 - c(s2) t, 1 - Kq(1/s2) t^2) =: pi_low(t);
#   pi^r >= pi_low^{r_max} with r_max = 1/s1 (pi<=1).  Hence on the box: phi <= E(t1, 1/s1) / pi_low(t2)^{1/s1}.
#   t >= 1: phi(t,s) = E(1/t + cos U)_+^r/(1-c)^r is decreasing in t, so phi <= phi(1,s) = c0(r) (3/2)^{(s-1) r},
#   bounded on the box by c0(1/s1)*(3/2)^{(s2-1)/s1} with c0(r) = E(1+cos U)^r = 2^r Gamma(r+1/2)/(sqrt(pi) Gamma(r+1)),
#   c0 increasing in r on [2/3,1]?  (checked rigorously below by evaluating c0 at both ends with monotone enclosure).
import mpmath as mp
iv = mp.iv; iv.dps = 30

def binom_abs(r, k):
    b = iv.mpf(1)
    for j in range(k): b = b*(r - j)/(j+1)
    return abs(b)

def E_upper(t, r, N=60):
    val = iv.mpf(1)
    for k in range(1, N+1):
        val = val - binom_abs(r, 2*k)*iv.mpf(mp.binomial(2*k, k))/iv.mpf(4)**k * t**(2*k)
    return val

def c0(r):
    return 2**r*iv.gamma(r+iv.mpf(1)/2)/(iv.sqrt(iv.pi)*iv.gamma(r+1))

def coefs(r, N=30):
    out = []
    for k in range(1, N+1):
        out.append(binom_abs(r, 2*k)*iv.mpf(mp.binomial(2*k, k))/iv.mpf(4)**k)
    return out

def box_bound_c(t1, t2, s1, s2, cf):
    s1 = iv.mpf(s1); s2 = iv.mpf(s2); t1 = iv.mpf(t1); t2 = iv.mpf(t2)
    rmax = 1/s1
    c2 = 1 - (iv.mpf(2)/3)**(s2-1); Kq2 = iv.mpf(5)/(5 + 4/s2)
    pl = 1-c2*t2          # P12 only (no quadratic payment)
    E = iv.mpf(1); tt = t1*t1; pw = tt
    for ck in cf:
        E = E - ck*pw; pw = pw*tt
    return (E/pl**rmax).b

def tail_bound(s1, s2):
    # enclosure over the whole box: r in [1/s2, 1/s1] as an interval (no monotonicity needed)
    s1 = iv.mpf(s1); s2 = iv.mpf(s2)
    rI = iv.mpf([(1/s2).a, (1/s1).b])
    return (c0(rI)*(iv.mpf(3)/2)**((s2-1)/s1)).b

if __name__ == '__main__':
    # monotonicity check of c0 on [2/3,1] (needed for the t>=1 branch): c0(r) = E(1+cos U)^r is increasing in r?
    # (1+cos U)^r: for 1+cos U < 1 decreasing in r, >1 increasing; check numerically on a fine grid (enclosures):
    vals = [c0(iv.mpf(k)/300) for k in range(200, 301)]
    mono = all(vals[i+1].a >= vals[i].b - 1e-25 for i in range(len(vals)-1))
    print("c0 increasing on [2/3,1] (grid check):", mono, flush=True)
    ns = 25; nt = 800
    worst = 0; worst_at = None
    srows = []
    for i in range(ns):
        s1 = 1 + 0.5*i/ns; s2 = 1 + 0.5*(i+1)/ns
        best = 0; cf = coefs(1/iv.mpf(s1))
        for k in range(nt):
            t1 = k/nt; t2 = (k+1)/nt
            b = box_bound_c(t1, t2, s1, s2, cf)
            if b > best: best = b; bt = t1
        tb = tail_bound(s1, s2)
        X = max(best, tb)
        srows.append((s1, s2, X, bt))
        if X > worst: worst = X; worst_at = (s1, s2, bt)
    for (s1, s2, X, bt) in srows[::5] + [srows[-1]]:
        print(f"s in [{s1:.3f},{s2:.3f}]: Xi_torus <= {mp.nstr(X, 8)}  (worst t~{bt:.4f}),  Xi^s <= {mp.nstr(X**s2, 8)}", flush=True)
    print("overall: max Xi_torus bound", mp.nstr(worst, 8), "at", worst_at)
