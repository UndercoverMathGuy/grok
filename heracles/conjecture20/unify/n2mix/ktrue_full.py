"""For the worst phase-aware configuration found at b = 2 (out/fullsocp72c_best.npy): compare the K-plan constant of the
optimal full test with the numerical neuron-level supremum K_true (exact denominator 1 - (sum_l (x y1 y2)^(1/3))^3),
and report the bound with K_true.  Exploration only (K_true is a numerical sup, not certified)."""
import numpy as np
from scipy.optimize import minimize
from fullsocp72 import FullBound, feature_from_atoms, FAMS
from fullsocp72c import decode
from socp72 import ROWS
from kplan import make_plan, eval_up

s = 1.5; r = 1 / s
v = np.load('out/fullsocp72c_best.npy')
k1, k2 = int(v[0]), int(v[1]); v = v[2:]
(b1, x1), (b2, x2) = decode(v, k1, k2)
A1, bt1, P1 = feature_from_atoms(b1, x1, s); A2, bt2, P2 = feature_from_atoms(b2, x2, s)
F = (P1 ** r + P2 ** r) ** s
fb = FullBound(s, 'pat')
val = fb([A1, A2], [bt1, bt2], F)
q = fb.q.value
ent = {}
for f, mem in FAMS.items():
    ent[f] = sum(2 ** (z / 2) * abs(ROWS[key] @ q) for (key, z) in mem)
Kplan = eval_up(make_plan(ent.keys()), ent)
a_over_F = None
rng = np.random.default_rng(0)
def neg(vv):
    xw = np.abs(vv[0:3]); y1 = np.abs(vv[3:7]); y2 = np.abs(vv[7:11])
    xw = xw / max(np.sum(xw), 1e-300); y1 = y1 / np.sum(y1); y2 = y2 / np.sum(y2)
    N = sum(c * np.sqrt(xw[l - 1] * y1[a] * y2[b]) for (l, a, b), c in ent.items())
    bL3 = sum((xw[l - 1] * y1[l] * y2[l]) ** (1 / 3) for l in (1, 2)) ** 3
    return -N / np.sqrt(max(1 - bL3, 1e-300))
Kt = 0
for k in range(60):
    res = minimize(neg, rng.uniform(0, 1, 11), method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
    Kt = max(Kt, -res.fun)
Kval = fb.K.value
t_val = np.sqrt(max((val * 3.5 * 3 ** (-s)) ** 2 - (3 ** (s - 1) * Kval) ** 2, 0))   # a/F recovered from objective
print(f"full test bound (K-plan) {val:.5f}; K_plan(socp) {Kval:.4f} K_plan(eval) {Kplan:.4f} K_true {Kt:.4f}; "
      f"bound with K_true and the same test {np.sqrt(t_val ** 2 + (3 ** (s - 1) * Kt) ** 2) / (3.5 * 3 ** (-s)):.5f}")
