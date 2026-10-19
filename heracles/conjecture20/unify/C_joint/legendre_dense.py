# Legendre single-T network on all of F: gamma/gamma_IF = tau_n(s) (1 - 2/sqrt p) e(s,p), e = (p/2)F^{-s}/Gamma_s.
# Gamma_s from reweighted LP (lower bound) -> this ratio is an UPPER bound for the true ratio.
import sys, numpy as np
from cjlib import gamma_lp
for p in [101, 211, 401, 1009]:
    F = (p-1)//2
    for s in [1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
        G, y = gamma_lp(p, s, restarts=8)
        U = (p/2)*F**(-s); e = U/G
        line = f"p={p} s={s:.2f} Gamma_lp={G:.6f} uniform={U:.6f} e<={e:.5f} K*={np.sum(y>1e-9)}"
        for n in [2, 3, 5]:
            if s <= (n+1)/2:
                tau = (1+1/n)**(s-1); line += f" | n={n}: tau={tau:.4f} R<={tau*(1-2/np.sqrt(p))*e:.5f}"
        print(line, flush=True)
