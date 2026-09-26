# Depth of the flip-enlarged aim body: can one weighting (plain, slot-1 flipped, slot-2 flipped, or mixtures) reach every
# point of the l2 polyball of radius A in the 3K junk coordinates?  depth = min_z max_sigma h_sigma(z) / sum_k |z_k|_2.
# Search gives an UPPER estimate (an explicit bad z). Compare plain body vs enlarged body; the proven cost side needs 0.272.
import sys, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(9)
def make(p, S):
    x = np.arange(p); ze = np.exp(-2j * np.pi * 2 * np.outer(S, x) / p); ab = (x[:, None] + x[None, :]) % p
    def h(zc, flips):
        best = -np.inf
        for s1, s2 in flips:     # sigma = (sign on slot 1 patterns, sign on slot 2 patterns); {1,2} pattern gets s1*s2
            f1 = s1 * np.real(zc[0] @ ze); f2 = s2 * np.real(zc[1] @ ze); f3 = s1 * s2 * np.real(zc[2] @ ze)
            best = max(best, (f1[:, None] + f2[None, :] + f3[ab]).max())
        return best
    return h
def depth(p, S, flips, starts=30):
    K = len(S); h = make(p, np.array(S))
    obj = lambda v: h((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K), flips) / np.linalg.norm((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K), axis=0).sum()
    return min(minimize(obj, rng.normal(size=6 * K), method='Powell', options=dict(maxiter=40000, xtol=1e-8, ftol=1e-10)).fun for _ in range(starts))
PLAIN = [(1, 1)]; FLIP = [(1, 1), (-1, 1), (1, -1)]
if 'v2' not in sys.argv:
  for p, S in [(11, [1, 2, 3, 4, 5]), (13, [1, 2, 3, 4, 5, 6]), (59, [1, 6, 14, 16, 19]), (59, [1, 4, 12, 18, 25, 27])]:
    print(f"p={p} S={S}: plain body depth <= {depth(p, S, PLAIN):.3f}   flip-enlarged <= {depth(p, S, FLIP):.3f}   (need 0.272)", flush=True)

# --- better search: structured starts (one pattern only, equal / alternating / random phases) + random, then polish ---
def depth2(p, S, flips, extra=20):
    K = len(S); h = make(p, np.array(S)); I = np.arange(K)
    obj = lambda v: h((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K), flips) / np.linalg.norm((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K), axis=0).sum()
    starts = []
    for T in range(3):
        for ph in (np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), rng.uniform(0, 2 * np.pi, K), np.pi * I ** 2 / K):
            zc = np.zeros((3, K), complex); zc[T] = np.exp(1j * ph)
            starts.append(np.r_[zc.real.ravel(), zc.imag.ravel()])
    starts += [rng.normal(size=6 * K) for _ in range(extra)]
    best = 9
    for s in starts:
        best = min(best, obj(s), minimize(obj, s + 1e-3 * rng.normal(size=6 * K), method='Powell',
                                          options=dict(maxiter=40000, xtol=1e-8, ftol=1e-10)).fun)
    return best
if __name__ == '__main__' and 'v2' in sys.argv:
    for p, S in [(11, [1, 2, 3, 4, 5]), (13, [1, 2, 3, 4, 5, 6]), (59, [1, 6, 14, 16, 19]), (59, [1, 4, 12, 18, 25, 27])]:
        print(f"p={p} S={S}: plain <= {depth2(p, S, PLAIN):.3f}   flip-enlarged <= {depth2(p, S, FLIP):.3f}   (need 0.272)", flush=True)
