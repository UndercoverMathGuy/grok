# Depth side of the progression race, with PROVED inputs only:
#   one-sided K-mode depth A* >= 1/(2L), two-sided A_s >= 1/sqrt(2L) (X_CHARGE / VERIFY general floors, any L-set);
#   flip-enhanced transfer constant c(A_s, A*) certified exactly (xcharge/cert_transfer.py, rational vertex enumeration);
#   PROVED discount curve (max of the two proved cost inequalities, see APPROG.md sec 2):
#       delta_prov(a) = min(sqrt3, max(1, sqrt3 (1 - a (1 - 3^{-1/2}) / 0.272))).
import sys, math
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../xcharge')
from transfer_flip import depth
from cert_transfer import run
def rat_down(x, den=10 ** 6):
    return Fr(math.floor(x * den), den)
def delta_prov(a):
    return min(math.sqrt(3), max(1.0, math.sqrt(3) * (1 - a * (1 - 3 ** -0.5) / 0.272)))
out = {}
for L in range(1, 21):
    Ao, As = rat_down(1 / (2 * L)), rat_down(1 / math.sqrt(2 * L))
    c_grid, lam = depth(float(As), float(Ao), n=200)
    lamr = [Fr(int(round(l * 10 ** 6)), 10 ** 6) for l in lam]
    s = sum(lamr); lamr = [l / s for l in lamr]
    c = rat_down(c_grid * 0.999, 10 ** 5)                  # certify slightly below the grid optimum
    while True:
        ok, nv, _ = run(Ao, As, lamr, c)
        if ok: break
        c = c * Fr(99, 100)
    out[L] = float(c)
    print(f"L={L:2d}: A*>={float(Ao):.4f} A_s>={float(As):.4f}  certified polyball depth a_cert={float(c):.4f} "
          f"(grid {c_grid:.4f})  -> proved discount delta_prov={delta_prov(float(c)):.4f}", flush=True)
