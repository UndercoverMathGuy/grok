# Partial defect, idealized model: frequencies in A get the full discount (c = 1, junk hidden for free), all others the
# block rate (c = 3^{1/3}). Does the discount on A pay, i.e. Gamma_c > Gamma*/sqrt3? Compare with A's own coverage defect.
import sys, numpy as np
from l5lib import *
sys.path.insert(0, '..')
from common import cover_matrix
rng = np.random.default_rng(5)
c3 = 3 ** (1 / 3)
for p in (13, 17, 19, 23):
    V = vertices(p); h = V.shape[1]; Acov = cover_matrix(p)
    base = gamma_c(V, np.full(h, c3))[0]                      # = Gamma*/sqrt3
    rows = []
    def test(Aset, label):
        c = np.full(h, c3); c[np.array(Aset) - 1] = 1.0
        g, v = gamma_c(V, c)
        vA = v[np.array(Aset) - 1]; shareA = (vA ** (2 / 3)).sum() / (c * v ** (2 / 3)).sum()
        rel_defect = (Acov[:, np.array(Aset) - 1].mean(1)).min()            # min over offsets of A's mean coverage (1 = average)
        bohr = int((Acov[:, np.array(Aset) - 1].mean(1) <= 0.25).sum())     # offsets (up to sign) where A covers <= 0.25
        rows.append((label, g / base, shareA, rel_defect, bohr))
    for L in (2, 3, 4, 6, h // 2 + 1, h):
        test(list(range(1, L + 1)), f"AP {{1..{L}}}")
    for K in (2, 3, 4):
        S = sorted(rng.choice(np.arange(1, h + 1), K, replace=False).tolist()); test(S, f"random {S}")
    print(f"p={p} (h={h}, Gamma*/sqrt3={base:.5f})")
    for label, r, sA, d, b in rows:
        print(f"   {label:22s} Gamma_c/(Gamma*/sqrt3) = {r:.4f}  share on A at optimum {sA:.3f}  A's min mean coverage {d:.3f}  #offsets with A-cov<=0.25: {b}")
