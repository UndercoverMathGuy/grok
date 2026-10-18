# b-dependent range of the Parseval-efficiency closure (relations case, n=2):
#   every relations-case network at prime p is closed (gamma <= IF, = only for IF) when c(s)^2 p/2 < 1, c(s) = 1-(2/3)^{s-1}, s = 3/b;
#   equivalently, at every prime p, for all b in (b_p, 3] with b_p = 3 / (1 + ln(1 - sqrt(2/p)) / ln(2/3)).
#   e_thr(p,s) = 1/(1 + tau'(c - sqrt(2/p))_+) >= (2/3)^{s-1}: allocations with e(a) <= e_thr are closed at p.
from mpmath import mp, mpf, log, sqrt
from sympy import prevprime, primerange
mp.dps = 30
for b in [mpf(2), mpf('2.1'), mpf('2.25'), mpf('2.4'), mpf('2.5'), mpf('2.75'), mpf('2.9')]:
    s = 3/b; c = 1-(mpf(2)/3)**(s-1); P = 2/c**2
    print(f"b={float(b):.2f} s={float(s):.4f} c12={float(c):.5f}  1/c^2={float(1/c**2):9.2f}  closed for all primes p < 2/c^2 = {float(P):10.2f}, largest prime {prevprime(int(P)+1) if P < 10**12 else 'large'}")
print()
for p in [61, 67, 101, 211, 1009, 10007, 10**6+3]:
    bp = 3/(1 + log(1-sqrt(mpf(2)/p))/log(mpf(2)/3))
    s = mpf(3)/2; c = 1-(mpf(2)/3)**(s-1); tp = (mpf(3)/2)**(s-1)
    eth = 1/(1+tp*max(c-sqrt(mpf(2)/p), 0))
    print(f"p={p}: closed for all b in ({float(bp):.4f}, 3];  e_thr(p, b=2) = {float(eth):.4f}  (p->inf limit (2/3)^(1/2) = {float((mpf(2)/3)**0.5):.4f})")
