# Near b=3: is the programme still dense where the full-F Legendre all-pair network would beat IF?
import numpy as np, sys, math
sys.path.insert(0, '..')
from sympy import legendre_symbol, isprime
from nearthree import programme, coset_sets
from relmodel import Model
for s in [float(a) for a in sys.argv[1].split(',')]:
    CS = 1 - (2 / 3) ** (s - 1); PAIR = 1.5 ** (s - 1)
    for p in [int(a) for a in sys.argv[2].split(',')]:
        F = (p - 1) // 2
        sup = [sorted(idx.keys()) for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20] for co, idx in coset_sets(p, k)[:3]]
        G, y = programme(p, s, sup, starts=4); K = int(np.sum(y > 0))
        valF = (p / 2) / F ** s
        M = Model(p, s, CS, G)
        chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
        th = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
        a = np.ones(F); R = M.ratio(a, PAIR * a, th, CS)
        print(f's={s} p={p} (p%4={p % 4}) Gamma={G:.5f} K*={K}/{F} value_F/Gamma={valF / G:.4f}  full-F Legendre pair network ratio={R:.4f}  '
              f'[needs s-1 > 4.93/sqrt(p)={4.93 / math.sqrt(p):.3f}]', flush=True)
