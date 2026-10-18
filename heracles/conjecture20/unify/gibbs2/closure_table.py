# Closure ranges for class C at n = 2 from  c^2 * sum_{L'} kappa < (p-1)/(p+1)  with sum_F kappa = p/2.
#   c_A = 1/(2 sqrt3) (Theorem A, all s);  c(s) = (1 - 3^{1-s})/sqrt3 (sharp L2 payment, certified);  Note Thm loc: threshold 4.
import numpy as np
def primes(N):
    s = np.ones(N + 1, bool); s[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if s[i]: s[i*i::i] = False
    return [int(q) for q in np.nonzero(s)[0] if q >= 3]
P = primes(20000)
def pmax(c):
    ok = [p for p in P if c*c*p/2 < (p - 1)/(p + 1)]
    # closure holds for an initial segment; report the largest p such that all primes <= p close
    last = 3
    for p in P:
        if c*c*p/2 < (p - 1)/(p + 1): last = p
        else: break
    return last
print("Theorem A, all b in [2,3]: closes all primes p <=", pmax(1/(2*np.sqrt(3))))
for b in [2.0, 2.1, 2.2, 2.25, 2.4, 2.5, 2.6, 2.75, 2.9]:
    s = 3/b; c = (1 - 3**(1 - s))/np.sqrt(3)
    thr = lambda p: (p - 1)/((p + 1)*c*c)
    print(f"b={b:4.2f} s={s:.4f}: c(s) = {c:.5f}; closes all primes p <= {pmax(c)};  cheap-set threshold sum kappa < {1/c**2:.2f}(p-1)/(p+1)")
