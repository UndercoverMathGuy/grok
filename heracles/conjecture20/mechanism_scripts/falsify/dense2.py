# Near b=3: programme value from several solvers (rlp from uniform/cosets/random; prog2 random-support restarts; prog3 local search)
# and the realisable full-F Legendre all-pair network ratio relative to the best value found.
import numpy as np, sys, math, json, os
sys.path.insert(0, '..')
from sympy import legendre_symbol
from nearthree import programme, coset_sets
from prog2 import solve as solve2
from prog3 import solve as solve3
from relmodel import Model
s = float(sys.argv[1]); CS = 1 - (2 / 3) ** (s - 1); PAIR = 1.5 ** (s - 1)
out = {}
for p in [int(a) for a in sys.argv[2].split(',')]:
    F = (p - 1) // 2
    sup = [sorted(idx.keys()) for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20] for co, idx in coset_sets(p, k)[:3]]
    G1, y1 = programme(p, s, sup, starts=4)
    G2, y2, _ = solve2(p, s, restarts=30, seed=p)
    G3, y3, _ = solve3(p, s, starts=int(sys.argv[3]) if len(sys.argv) > 3 else 2, seed=p)
    Gs = [(G1, y1, 'rlp'), (G2, y2, 'prog2'), (G3, y3, 'prog3')]
    G, y, src = max(Gs, key=lambda z: z[0])
    valF = (p / 2) / F ** s
    M = Model(p, s, CS, G)
    chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
    th = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
    a = np.ones(F); R = M.ratio(a, PAIR * a, th, CS)
    print(f's={s} p={p} (p%4={p % 4}) Gamma={G:.5f} [{src}; rlp {G1:.5f} prog2 {G2:.5f} prog3 {G3:.5f}] K*={int(np.sum(y > 0))}/{F} '
          f'value_F/Gamma={valF / G:.4f}  full-F Legendre pair network ratio={R:.4f}  [(3/2)^(s-1)(1-2/sqrt p)={PAIR * (1 - 2 / math.sqrt(p)):.4f}]', flush=True)
    out[p] = dict(G=G, K=int(np.sum(y > 0)), valF=valF, R=R)
json.dump(out, open(f'dense2_{s}_{sys.argv[2].replace(",", "_")}.json', 'w'))
