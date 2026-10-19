# Sharp constant: maximise E F^2 over laws on [-1, X] with E F = 0, E F^3 <= 0 (so that -min F <= 1).
import numpy as np
from scipy.optimize import linprog
for X in [3, 10, 30]:
    g = np.concatenate([np.linspace(-1, 0, 2001), np.linspace(0, X, 20001)[1:]])
    res = linprog(-g**2, A_eq=np.vstack([np.ones_like(g), g]), b_eq=[1, 0], A_ub=(g**3)[None, :], b_ub=[0], bounds=(0, None), method='highs')
    q = res.x; sup = g[q > 1e-9]
    print(f"X={X}: max E F^2 = {-res.fun:.6f} -> c* = {1/np.sqrt(-res.fun):.6f}; atoms {np.round(sup,4)} weights {np.round(q[q>1e-9],4)}")
# closed form guess: atoms -1, 0?, b ; check two/three atom optimum analytically
from scipy.optimize import minimize_scalar
best = 0
for a in np.linspace(-1, 0, 401):          # atoms -1 (w0), a (w1), b (w2) with b>0
    def negE2(b):
        M = np.array([[1, 1, 1], [-1, a, b], [-1, a**3, b**3]]); 
        try: w = np.linalg.solve(M, [1, 0, 0])
        except: return 0
        if np.any(w < -1e-12): return 0
        return -(w @ np.array([1, a*a, b*b]))
    r = minimize_scalar(negE2, bounds=(1e-6, 10), method='bounded')
    best = max(best, -r.fun)
print("3-atom search (E F^3 = 0 active): max E F^2 =", best, " c* =", 1/np.sqrt(best))
