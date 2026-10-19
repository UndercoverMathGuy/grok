# c* search at general s (b = 3/s): seeds from seeds.py, gradient crit, polish best 3.
import numpy as np, sys, time, math, torch
from gget import gamma_table
from relmodel import Model, optimise, crit_polish
from seeds import seeds
torch.set_num_threads(1)
s = float(sys.argv[1]); p = int(sys.argv[2])
CS = 1 - (2 / 3) ** (s - 1); CA = 1 / (2 * 3 ** 0.5)
T = gamma_table(s)[p]; G = T['Gamma']; M = Model(p, s, CS, G)
rng = np.random.default_rng(p)
cands = []; t0 = time.time()
for name, S, A, TH in seeds(p, rng, (T['S'], np.array(T['y'])), nrand=1):
    a, v, th = optimise(M, np.array(S), A + 1e-12, M.cap * A * 0.9 + 1e-12, TH, mode='crit')
    c1 = M.cstar(a, v, th)[0]; cands.append((c1, name, a, v, th))
    print(f'{name:14s} c*={c1:.4f}  [{time.time()-t0:.0f}s]', flush=True)
# guard against an underestimated Gamma: re-solve the programme on every candidate support
import sys as _s; _s.path.insert(0,'..')
from prog3 import restricted
for c1, name, a, v, th in cands:
    S = np.nonzero(a > 1e-9 * a.max())[0]
    Q, yy = restricted(M.Knp, 1 / s, list(S), np.random.default_rng(0), tries=4)
    if Q < np.inf and Q ** (-s) > M.G * (1 + 1e-9):
        print(f'  Gamma raised by support of {name}: {M.G:.6f} -> {Q ** (-s):.6f}', flush=True); M.G = Q ** (-s)
cands = [(M.cstar(a, v, th)[0], name, a, v, th) for c1, name, a, v, th in cands]
cands = [z for z in cands if np.isfinite(z[0])]
cands.sort(key=lambda z: z[0]); best = cands[0][:2]
for c1, name, a, v, th in cands[:3]:
    c2, a2, v2 = crit_polish(M, a, v, th)
    if c2 < best[0]: best = (c2, name)
print(f'== s={s} p={p} Gamma={M.G:.5f} K*={len(T["S"])}: best c*={best[0]:.4f} ({best[1]})  sharp c={CS:.4f}  ratio c*/c_sharp={best[0]/CS:.2f}  c*/0.289={best[0]/CA:.2f}', flush=True)
