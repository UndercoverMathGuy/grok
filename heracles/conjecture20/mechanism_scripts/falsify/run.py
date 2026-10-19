# For each prime: (i) minimise the critical payment constant c* over configurations (seeds below);
# (ii) maximise the ratio with every support feature forced cheap: v = cap*a (cap=3^{s-1}) and v = (3/2)^{s-1} a
#      (the two-aligned-neuron level), at c = 0.1835 (sharp, P12) and c = 0.2887 (Theorem A);
# (iii) free maximisation of the ratio at c=0.1835 / 0.2887 (expected to return to IF, ratio 1, v=0).
import numpy as np, sys, time, json, os, torch
from gget import gamma_table
from relmodel import Model, optimise, lp_polish
from seeds import seeds
torch.set_num_threads(1)
s = 1.5; CS = 1 - (2 / 3) ** (s - 1); CA = 1 / (2 * np.sqrt(3)); PAIR = 1.5 ** (s - 1)
p = int(sys.argv[1]); nrand = int(sys.argv[2]) if len(sys.argv) > 2 else 2
T = gamma_table()[p]; G = T['Gamma']
M = Model(p, s, CS, G)
rng = np.random.default_rng(p)
SD = seeds(p, rng, (T['S'], np.array(T['y'])), nrand=nrand)
t0 = time.time(); res = {}
def upd(key, val, cfg, name, larger=True):
    if key not in res or (val > res[key][0] if larger else val < res[key][0]):
        res[key] = (val, name, cfg)
for name, S, A, TH in SD:
    S = np.array(S)
    # (i) critical constant
    V = M.cap * A * 0.9
    a, v, th = optimise(M, S, A + 1e-9, V + 1e-12, TH, mode='crit')
    cs, d = M.cstar(a, v, th)
    upd('cstar', cs, (a, v, th), name, larger=False)
    line = f'{name:14s} |S|={len(S):3d}  c*={cs:.4f}'
    # (ii) forced cheap
    for lvl, ln in [(M.cap, 'cap'), (PAIR, 'pair')]:
        for c, cn in [(CS, 'sh'), (CA, 'A')]:
            a, v, th = optimise(M, S, A + 1e-9, lvl * A + 1e-12, TH, mode='ratio', fixv=lvl, c=c)
            R = M.ratio(a, v, th, c)
            R2, a2, v2 = lp_polish(M, a, v, th, fixv=lvl, c=c, rounds=6)
            if R2 > R: R, a, v = R2, a2, v2
            upd(f'forced_{ln}_{cn}', R, (a, v, th), name)
            line += f'  {ln}/{cn}={R:.4f}'
    # (iii) free
    for c, cn in [(CS, 'sh'), (CA, 'A')]:
        a, v, th = optimise(M, S, A + 1e-9, 0.5 * M.cap * A + 1e-12, TH, mode='ratio', c=c)
        R = M.ratio(a, v, th, c)
        upd(f'free_{cn}', R, (a, v, th), name)
        line += f'  free/{cn}={R:.4f}(cheap {int(np.sum(v > 1e-3 * a.max()))})'
    print(line + f'  [{time.time() - t0:.0f}s]', flush=True)
print(f'== p={p} Gamma={G:.6f}  SUMMARY', flush=True)
for k, (val, name, (a, v, th)) in sorted(res.items()):
    inf = M.info(a, v, th, CA if k.endswith('_A') else CS)
    print(f'  {k:16s} {val:.4f}  from {name:14s} {inf}', flush=True)
os.makedirs('res', exist_ok=True)
np.save(f'res/run_p{p}.npy', {k: (val, name, [x.tolist() for x in cfg]) for k, (val, name, cfg) in res.items()}, allow_pickle=True)
