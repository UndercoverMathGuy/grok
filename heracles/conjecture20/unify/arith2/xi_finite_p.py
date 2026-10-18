# Finite-p correction for the certified label-row constant Xi (relations case, n=2).
# For feature l the label-row average runs over U in theta_l + (2 pi/p) Z_p (w -> l w is a bijection of Z_p), so
#   avg_p f - fhat(0) = sum_{j != 0} fhat(j p) e^{i j p theta},   f(U) = (1 + t cos U)^r,  0 <= t <= 1.
# With t = 2q/(1+q^2), q in [0,1]:  f = (1+q^2)^{-r} (1+q e^{iU})^r (1+q e^{-iU})^r, hence
#   fhat(m) = (1+q^2)^{-r} sum_{b>=0} binom(r,b+m) binom(r,b) q^{2b+m},
#   |fhat(m)| <= q^m |binom(r,m)| sum_b |binom(r,b)| <= 2 |binom(r,m)|      (|binom(r,k)| decreasing in k; sum_{b>=1}|binom(r,b)| = 1).
# Gautschi (Gamma(m)/Gamma(m-r) > (m-1)^r) gives |binom(r,m)| = r Gamma(m-r)/(Gamma(1-r) m!) < r / (Gamma(1-r) m (m-1)^r).
# So |avg_p f - E f| <= alpha_p(r) := 4 sum_{j>=1} r/(Gamma(1-r) jp (jp-1)^r).  For t >= 1 the p-average of (1/t + cos U)_+^r is
# decreasing in t, so the t >= 1 branch reduces to t = 1 (q = 1) with the same alpha_p.  Denominators pi(t,s)^r >= (1-c)^r = (2/3)^{(s-1)/s}.
# Result: Xi_p(s) <= Xi_torus(s) + alpha_p(r) (3/2)^{(s-1)/s}, with Xi_torus(s) <= 1.0680 certified in xi_certify_all_s.py.
# Rigour: r/Gamma(1-r) is decreasing on [2/3,1) (checked below on a grid with interval enclosures + it tends to 0 at r=1), and
# (jp-1)^{-r} <= (jp-1)^{-2/3}; the j-sum is bounded by its first J terms plus the integral tail.
import mpmath as mp
iv = mp.iv; iv.dps = 30

def check_decreasing():
    vals = [iv.mpf(k)/300/iv.gamma(1-iv.mpf(k)/300) for k in range(200, 300)]
    return all(vals[i+1].b <= vals[i].a for i in range(len(vals)-1))

def alpha_upper(p, J=2000):
    C = (iv.mpf(2)/3)/iv.gamma(iv.mpf(1)/3)            # max of r/Gamma(1-r) on [2/3,1)
    S = iv.mpf(0)
    for j in range(1, J+1):
        S += 1/(iv.mpf(j*p)*(iv.mpf(j*p-1))**(iv.mpf(2)/3))
    # tail j > J: 1/(jp (jp-1)^{2/3}) <= 1/((jp-1)^{5/3}) integrated: int_{J}^{inf} dx/(xp-1)^{5/3} = (3/2)/(p (Jp-1)^{2/3})
    S += (iv.mpf(3)/2)/(iv.mpf(p)*(iv.mpf(J*p-1))**(iv.mpf(2)/3))
    up = (4*C*S).b
    return mp.mpf(str(up).strip('[]').split(',')[-1]) * (1 + mp.mpf('1e-20'))   # upper endpoint, rounded up

if __name__ == '__main__':
    print("r/Gamma(1-r) decreasing on [2/3,1) (interval grid check):", check_decreasing())
    XiT = mp.mpf('1.0680')
    for p in [5, 7, 11, 13, 23, 31, 61, 67, 101, 211, 1009, 10007]:
        a = alpha_upper(p)
        Xi = XiT + a*mp.mpf(1.5)**(mp.mpf(1)/3)
        print(f"p={p}: alpha_p <= {mp.nstr(a,6)}   Xi_p <= {mp.nstr(Xi,6)}   gain bound Xi_p^s (s<=3/2) <= {mp.nstr(Xi**1.5,6)}")
