# Critical discount: largest amplitude-per-cost^{3/2} advantage delta on the set A (others at block rate) for which the
# discounted programme does NOT exceed Gamma*/sqrt3. delta = 1: block rate; delta = sqrt3: full discount (one hidden atom).
# budget weight on A: c = 3^{1/3} delta^{-2/3}. Also the depth a_A the aim must reach on A if the discount at depth a is
# sqrt3 (1 - sqrt3 a) (lone-atom curve, CHAIN d.3; conjectured sharp): a_A = (1 - delta_crit/sqrt3)/sqrt3.
import itertools, sys, numpy as np
from l5lib import *
c3 = 3 ** (1 / 3)
def delta_crit(V, h, Aset, base):
    lo, hi = 1.0, np.sqrt(3)
    def pays(d):
        c = np.full(h, c3); c[np.array(Aset) - 1] = c3 * d ** (-2 / 3)
        return gamma_c(V, c)[0] > base * (1 + 1e-12)
    if not pays(1 + 1e-9): pass
    if not pays(hi): return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        if pays(mid): hi = mid
        else: lo = mid
    return lo
for p in (13, 17, 19, 23):
    V = vertices(p); h = V.shape[1]; base = gamma_c(V, np.full(h, c3))[0]
    optsup = {tuple(np.nonzero(v > 1e-9)[0] + 1) for v in V if abs((c3 * v ** (2 / 3)).sum() ** -1.5 - base) < 1e-9}
    res = []
    for K in (1, 2, 3):
        for Aset in itertools.combinations(range(1, h + 1), K):
            res.append((delta_crit(V, h, Aset, base), Aset))
    res.sort()
    harmless = [(d, A) for d, A in res if d > 1 + 1e-6]
    print(f"p={p}: optimal support(s) {sorted(optsup)}; sets of size 1-3 tested {len(res)}; with delta_crit > 1: {len(harmless)}")
    for d, A in harmless[-6:][::-1]:
        print(f"    A={A}: delta_crit = {d:.4f}  -> depth needed on A <= {(1 - d / np.sqrt(3)) / np.sqrt(3):.4f} (vs 0.244 generic)")
