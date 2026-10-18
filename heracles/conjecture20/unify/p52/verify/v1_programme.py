"""Referee check of R1(i) and R11(i): Gamma_s(p) for p=5,7, s in [1,1.5].
Constraint: y>=0, sum y_j^r <= 1, r = 1/s.  Parametrise boundary y_j = z_j^s, z in simplex.
Also test the claimed step 'y_1+y_2 <= 2^{1-s}' (power mean) directly."""
import numpy as np
def kap(p, j, d): return 1 - np.cos(2*np.pi*j*d/p)
def f_val(p, y):
    F = range(1, (p-1)//2+1)
    return min(sum(kap(p, j, d)*y[j-1] for j in F) for d in range(1, p))
print("== power-mean step test (claimed y1+y2 <= 2^{1-s} under y1^r+y2^r<=1) ==")
for s in [1.0, 1.25, 1.5]:
    y = np.array([1.0, 0.0]); print(f" s={s}: y=(1,0): sum y^r = 1, y1+y2 = 1.0 vs 2^(1-s) = {2**(1-s):.4f}  -> step {'OK' if 1.0 <= 2**(1-s)+1e-12 else 'FALSE'}")
    y = np.array([0.9**s, 0.1**s]); print(f"        y=(0.9^s,0.1^s): y1+y2={y.sum():.4f}  vs {2**(1-s):.4f}")
print("== Gamma_s(5): grid over z in [0,1] ==")
zs = np.linspace(0, 1, 200001)
for s in np.linspace(1, 1.5, 11):
    vals = np.array([0.0]*len(zs))
    y1 = zs**s; y2 = (1-zs)**s
    v = np.minimum(kap(5,1,1)*y1 + kap(5,2,1)*y2, kap(5,1,2)*y1 + kap(5,2,2)*y2)
    i = np.argmax(v)
    # second best local maximum away from z=1/2
    mask = np.abs(zs-0.5) > 0.05
    print(f" s={s:.2f}: max={v[i]:.6f} at z={zs[i]:.5f}; claim 2.5*2^-s={2.5*2**-s:.6f}; best away from 1/2: {v[mask].max():.6f}")
print("== Gamma_s(7): grid over the 2-simplex in z ==")
N = 1500
a = np.linspace(0, 1, N+1)
Z1, Z2 = np.meshgrid(a, a, indexing='ij'); Z3 = 1 - Z1 - Z2
m = Z3 >= -1e-15; Z1 = Z1[m]; Z2 = Z2[m]; Z3 = np.maximum(Z3[m], 0)
for s in [1.0, 1.1, 1.182, 1.25, 1.4, 1.5]:
    Y = [Z1**s, Z2**s, Z3**s]
    v = np.min(np.stack([sum(kap(7, j, d)*Y[j-1] for j in (1,2,3)) for d in (1,2,3)]), axis=0)
    i = np.argmax(v)
    far = (np.abs(Z1-1/3)+np.abs(Z2-1/3)+np.abs(Z3-1/3)) > 0.05
    print(f" s={s}: max={v[i]:.6f} at z=({Z1[i]:.4f},{Z2[i]:.4f},{Z3[i]:.4f}); claim 3.5*3^-s={3.5*3**-s:.6f}; best far from uniform {v[far].max():.6f}")
# local refinement of the best 2-support value at p=7
from scipy.optimize import minimize_scalar
print("== p=7 best two-feature and one-feature values ==")
for s in [1.0, 1.25, 1.5]:
    best = 0
    for (j1, j2) in [(1,2),(2,3),(1,3)]:
        g = lambda z: -min(kap(7,j1,d)*z**s + kap(7,j2,d)*(1-z)**s for d in (1,2,3))
        zz = np.linspace(0,1,100001); vv = [-g(z) for z in zz[::100]]
        best = max(best, max(vv))
    print(f" s={s}: two-feature best ~{best:.5f}, one feature {kap(7,1,1):.5f}, uniform {3.5*3**-s:.5f}")
