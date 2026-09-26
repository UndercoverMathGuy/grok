"""Riesz product with relation errors (L3).

H = frequencies, modes m_k = 2k mod p.  For eps in {-1,0,1}^H let s(eps) = sum eps_k m_k mod p, |eps| = #nonzero.
Exact integer histograms:
  N0[s]    = #{eps != 0 : s(eps) = 0, |eps| = s}                  (mass errors)
  Nk[k][s] = #{eps != e_k : s(eps) = m_k, |eps| = s}              (moment errors at k)
Bounds (L3.md):
  one-sided  A*(H) >= max_rho min_k (rho - E_k(rho)) / (1 + E_0(rho)),          E(rho) = sum_s N[s] rho^s
  two-sided  A_s(H) >= max_rho min_k (rho - E_k^odd(rho)) / (1 + E_0^even(rho))
All counts are exact integers; evaluation at rational rho is exact (fractions)."""
import sys, numpy as np
from fractions import Fraction as Fr

def histograms(p, H):
    K = len(H); m = [(2 * k) % p for k in H]
    sums = np.zeros(1, np.int64); sup = np.zeros(1, np.int8)
    for mk in m:
        sums = np.concatenate([sums, (sums + mk) % p, (sums - mk) % p])
        sup = np.concatenate([sup, sup + 1, sup + 1])
    N0 = np.bincount(sup[(sums == 0)], minlength=K + 1).astype(object); N0[0] -= 1       # drop eps = 0
    Nk = []
    for mk in m:
        c = np.bincount(sup[(sums == mk)], minlength=K + 1).astype(object); c[1] -= 1     # drop eps = e_k
        Nk.append(c)
    return N0, Nk

def E(N, rho, parity=None):
    return sum(N[s] * rho ** s for s in range(len(N)) if N[s] and (parity is None or s % 2 == parity))

def bounds(p, H, rhos=None):
    N0, Nk = histograms(p, H)
    rhos = rhos or [Fr(i, 200) for i in range(1, 101)]
    one = max((min(rho - E(n, rho) for n in Nk) / (1 + E(N0, rho)), rho) for rho in rhos)
    two = max((min(rho - E(n, rho, 1) for n in Nk) / (1 + E(N0, rho, 0)), rho) for rho in rhos)
    return one, two, N0, Nk

if __name__ == '__main__':
    p = int(sys.argv[1]); H = [int(x) for x in sys.argv[2].split(',')]
    one, two, N0, Nk = bounds(p, H)
    print(f"p={p} H={H}: one-sided >= {float(one[0]):.4f} (rho={one[1]}), two-sided >= {float(two[0]):.4f} (rho={two[1]})")
    print("  mass-error counts by support:", list(N0))
