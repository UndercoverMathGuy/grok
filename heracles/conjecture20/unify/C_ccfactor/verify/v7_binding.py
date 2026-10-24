# Referee check v7 (CC15): programme maximisers by exact vertex enumeration (vlib.programme), binding sets,
# antipodal-feature property, individual / joint detectability by an entrywise-nonnegative Q (own LP).
import sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V


def Psi(p, l, offs):
    c = 1 - np.exp(-2j * np.pi * l * np.array(offs) / p)
    return np.real(np.outer(c, np.conj(c)))


rows = []
for p in [5, 7, 11, 13, 17, 19]:
    for s in [1.05, 1.5, 2.0, 3.0, 4.0]:
        G, y = V.programme(p, s)
        S = list(np.nonzero(y > 1e-12)[0] + 1)
        cov = V.kap(p) @ y
        Bd = list(np.nonzero(cov <= G * (1 + 1e-9))[0] + 1)
        Bd_half = [d for d in Bd if d <= (p - 1) // 2]
        anti = all(any(np.cos(2 * np.pi * l * d / p) <= 0 for l in S) for d in Bd_half)
        offs = sorted(Bd)
        k = len(offs); pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if len(S) >= 2:
            A = np.array([[2 * Psi(p, l, offs)[i, j] for (i, j) in pairs] for l in S])
            joint = linprog(np.zeros(len(pairs)), A_ub=A, b_ub=-np.ones(len(S)), bounds=(0, None), method='highs').status == 0
            indiv = all((A[i] < -1e-12).any() for i in range(len(S)))
        else:
            joint = indiv = None
        rows.append((p, s, len(S), len(Bd_half), anti, indiv, joint))
        print(f'p={p:2d} s={s:4.2f}: K*={len(S)} support={S} binding(1..F)={Bd_half} antipodal-at-every-binding={anti}'
              f' indiv={indiv} joint={joint}', flush=True)
