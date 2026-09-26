# Confirm f.3 findings: (i) explicit 2-atom population invisible to slot-1 aiming; (ii) boundary points with a finer grid.
import json, glob, numpy as np
import f3_threshold as F
F.E = np.exp(1j * np.linspace(0, 2 * np.pi, 16385)[:-1])
# (i) two equal atoms, opposite slot-1 phases, same slot-2 phase: J1 = J12 = 0, J2 = G
c = np.array([1.0, 1.0]); v1 = np.array([1.0, -1.0]); v2 = np.array([1.0, 1.0])
G = c.sum(); J1 = (c * v1).sum(); J2 = (c * v2).sum(); J12 = (c * v1 * v2).sum()
cost = (np.abs(c) ** (2 / 3)).sum()
for A1 in (0.5, 1.0, 1.9):
    m = min(np.real(G + J2 * 0.0) - A1 * abs(J1 + J12 * 0.0), np.inf)
    print(f"2-atom, A2=0, A1={A1}: J1={J1:.0f} J12={J12:.0f} J2={J2:.0f}; cost {cost:.4f} vs 3^(1/3) m^(2/3) = {3 ** (1 / 3) * m ** (2 / 3):.4f} -> {'FAILS' if cost < 3 ** (1 / 3) * m ** (2 / 3) else 'holds'}")
# (ii) re-evaluate the search argmax at each A2 on a 16384-point grid
for fn in sorted(glob.glob('f3_thr_*.log')):
    d = json.loads(open(fn).read().strip().splitlines()[-1])
    par = np.array(d['argmax_par']); N = d['argmax_N']
    print(f"A2={d['A2']}: search {d['threshold_A1']:.5f}, refined {F.a1star(par, N, d['A2']):.5f}, N={N}")
