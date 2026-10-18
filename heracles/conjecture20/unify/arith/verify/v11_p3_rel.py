# Referee: value sanity at p=3 and p=5 (n=2, relations case, class C): optimise gamma/IF over atom parameters with B=C=0
# enforced by construction; R2(b) predicts gamma <= IF (p < 59), with equality only for interference-free networks.
import numpy as np, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/arith/verify'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/arith2')
import v2_network as V, lib
from scipy.optimize import minimize
rng = np.random.default_rng(3)
def feats_from(theta, p, k, s):
    # per feature: k atoms; b-phases from zero-sum complex z (k complex), c = b + g (or -b+g), a-phases free
    F = (p-1)//2; out = []; i = 0
    for l in range(1, F+1):
        z = theta[i:i+k]+1j*theta[i+k:i+2*k]; z = z-z.mean(); g = theta[i+2*k]; aph = theta[i+2*k+1:i+3*k+1]; i += 3*k+1
        atoms = [(abs(zj)+1e-9, aph[j], np.angle(zj), np.angle(zj)+g) for j, zj in enumerate(z)]
        out.append((l, atoms))
    return out
for p, s in [(3, 1.5), (5, 1.5), (5, 1.25), (7, 1.5)]:
    Gam, y = lib.prog(p, s, restarts=3); best = -np.inf; k = 3
    F = (p-1)//2; dim = F*(3*k+1)
    def obj(th):
        o = V.analyse(p, s, V.build(p, feats_from(th, p, k, s)), Gam)
        return -o['gamma/IF']
    for st in range(6):
        th0 = rng.normal(size=dim)
        res = minimize(obj, th0, method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-6, fatol=1e-9))
        best = max(best, -res.fun)
    print(f"p={p} s={s}: best gamma/IF found in relations case = {best:.6f}  (must be <= 1)", flush=True)
