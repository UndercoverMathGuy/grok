# Theorem G wording check: the exceptional set {lambda : |M^lambda| >= 2} near 1 is not just {1} (or the ray c*1):
# it contains tie hypersurfaces through 1. Find lambda = (1, 1+t, 1+u), t != 0, with two programme maximisers.
import numpy as np, sys
from scipy.optimize import brentq
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
from vlib import *
p, s = 7, 1.8
def vals(lam):
    G, M = programme(p, s, lam=lam)
    out = {}
    for y in M: out[tuple(np.nonzero(y)[0] + 1)] = G
    return G, M
def restricted(lam, S): return programme(p, s, lam=lam, S=S)[0]
for t in [0.01, -0.02]:
    # tie between supports (1,2) and (1,3)? solve restricted values equal in u
    fn = lambda u: np.log(restricted(np.array([1, 1 + t, 1 + u]), [1, 2])) - np.log(restricted(np.array([1, 1 + t, 1 + u]), [1, 3]))
    u = brentq(fn, -0.2, 0.2)
    lam = np.array([1, 1 + t, 1 + u]); G, M = programme(p, s, lam=lam)
    print('lam=%s: #programme maximisers=%d, supports=%s' % (np.round(lam, 6), len(M), [tuple(np.nonzero(y)[0] + 1) for y in M]))
