# Corollary (Parseval closure at every n): explicit single-T networks (Prop. emb) with |L'| <= 3 cheap features, or with
# p < 2/c_n(s)^2, never beat the IF value.  Random and locally optimised configurations; exact Gamma (p <= 23).
import numpy as np
from cjlib import TL, gamma_exact, tl_optimise
rng = np.random.default_rng(31)
worst = -1e9; cases = 0
SG = {}
for trial in range(150):
    p = [7, 11, 13, 17, 19][trial % 5]; n = [2, 3, 4, 5, 7][(trial//5) % 5]; nu = n+1
    s = round(rng.choice(np.linspace(1.1, nu/2, 4)), 6); c = 1-(n/nu)**(s-1); F = (p-1)//2
    if (p, s) not in SG: SG[(p, s)] = gamma_exact(p, s)
    Gam, ys = SG[(p, s)]
    tl = TL(p, n, s, list(range(1, F+1)))
    small_p = p < 2/c**2
    k = F if small_p else min(3, F)
    cheap = rng.choice(F, size=k, replace=False)
    al = np.maximum(ys[0], 0) + 1e-9; be = np.zeros(F); be[cheap] = rng.uniform(0.05, 1.5, k)*np.maximum(ys[0][cheap], 0.2*ys[0].max())
    ph = rng.uniform(-np.pi, np.pi, F)
    if trial % 2 == 1:   # optimise amplitudes and phases (beta only on the cheap set) via the soft-min optimiser
        x0 = np.concatenate([np.log(al), np.log(be + 1e-12), ph])
        R, (a1, b1, p1) = tl_optimise(tl, Gam, x0, taus=(1e-2, 3e-3, 1e-3))
        mask = np.ones(F, bool); mask[cheap] = False; b1[mask] = 0.0
        al, be, ph = a1, b1, p1
    R = tl.ratio(al, be, ph, Gam); worst = max(worst, R); cases += 1
print(f"{cases} single-T networks (|L'|<=3 or p<2/c_n^2): max gamma/gamma_IF = {worst:.6f} (<= 1 expected)")
