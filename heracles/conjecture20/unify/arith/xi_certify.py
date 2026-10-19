# Rigorous upper bound for the label-row constant
#   Xi(s) = sup_{t>=0} E_U (1 + t cos U)_+^r / pi(t)^r,   pi(t) = max(1-c t, 1-Kq t^2) (t<=1),  t(1-c) (t>=1),
# c = 1-(2/3)^{s-1}, Kq = 5/(5+4r), r = 1/s.
# (i) t in [0,1]:  E(t) := E(1+t cos U)^r = 1 - sum_{k>=1} |binom(r,2k)| binom(2k,k) 4^{-k} t^{2k}   (all k>=1 terms are <= 0),
#     so any truncation is an UPPER bound for E(t); E is decreasing, pi is decreasing, hence on [t1,t2]:
#     phi(t) <= E_N(t1) / pi(t2)^r.   Computed in exact rational/interval-safe arithmetic with mpmath (outward rounding).
# (ii) t >= 1: phi(t) = E(1/t + cos U)_+^r / (1-c)^r is decreasing in t, so phi(t) <= phi(1) = c0(r) (3/2)^{(s-1) r}.
import mpmath as mp
mp.mp.dps = 40
iv = mp.iv; iv.dps = 40

def bound(s, n_grid=4000, N=60):
    r = iv.mpf(1)/iv.mpf(s); c = 1 - (iv.mpf(2)/3)**(iv.mpf(s)-1); Kq = iv.mpf(5)/(5+4*r)
    # coefficients |binom(r,2k)| binom(2k,k) 4^-k, k=1..N
    coef = []
    for k in range(1, N+1):
        b = iv.mpf(1)
        for j in range(2*k): b = b*(r - j)/(j+1)        # binom(r, 2k)
        coef.append(abs(b)*iv.mpf(mp.binomial(2*k, k))/iv.mpf(4)**k)
    def E_upper(t):          # rigorous upper bound of E(t) for t in [0,1]
        val = iv.mpf(1)
        for k, ck in enumerate(coef, start=1): val = val - ck*t**(2*k)
        return val
    def pi_lower(t):         # pi is decreasing on [0,1]; returns an interval containing pi(t)
        return iv.mpf([max((1-c*t).a, (1-Kq*t*t).a), max((1-c*t).b, (1-Kq*t*t).b)])
    best = iv.mpf(0)
    worst_t = None
    for i in range(n_grid):
        t1 = iv.mpf(i)/n_grid; t2 = iv.mpf(i+1)/n_grid
        ub = E_upper(t1)/pi_lower(t2)**r
        if ub.b > best.b: best = ub; worst_t = float(t1.a)
    c0 = 2**r*iv.gamma(r+iv.mpf(1)/2)/(iv.sqrt(iv.pi)*iv.gamma(r+1))
    tail = c0*(iv.mpf(3)/2)**((iv.mpf(s)-1)*r)
    return best.b, worst_t, tail.b

if __name__ == '__main__':
    for s in ['1.5', '1.25', '1.1']:
        ub, tw, tail = bound(mp.mpf(s))
        X = max(ub, tail)
        print(f"s={s}: rigorous sup_[0,1] phi <= {mp.nstr(ub, 8)} (worst cell t~{tw:.4f}); t>=1 branch <= {mp.nstr(tail, 8)};"
              f"  Xi <= {mp.nstr(X, 8)},  gain bound Xi^s <= {mp.nstr(X**mp.mpf(s), 8)}", flush=True)
