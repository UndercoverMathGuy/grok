# Large-p version: reduced seed set, working-set optimiser.  Reports best c* and best all-cheap (forced) ratios.
import numpy as np, sys, time, math, torch, os
from sympy import legendre_symbol
from gget import gamma_table
from relmodel import Model, optimise_ws, lp_polish, crit_polish
from seeds import fold
torch.set_num_threads(1)
s = 1.5; CS = 1 - (2 / 3) ** 0.5; CA = 1 / (2 * 3 ** 0.5); PAIR = 1.5 ** 0.5
p = int(sys.argv[1])
T = gamma_table()[p]; G = T['Gamma']; M = Model(p, s, CS, G); F = M.F; L = math.log(p)
rng = np.random.default_rng(p + 99)
Sy = np.array(T['S']) - 1; y = np.array(T['y'])
chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
thleg = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
def goodrand(K, tries=20):
    best = None
    for q in range(tries):
        S = np.sort(rng.choice(F, K, replace=False)); cv = M.Knp[:, S].sum(1).min()
        if best is None or cv > best[0]: best = (cv, S)
    return best[1]
SD = [('full-legendre', np.arange(F), np.ones(F), thleg), ('full-rand', np.arange(F), np.ones(F), rng.uniform(0, 2 * np.pi, F))]
for m in [4, 8, 16]:
    K = min(F, int(round(m * L))); S = goodrand(K); A = np.zeros(F); A[S] = 1
    SD.append((f'rand{m}L', S, A, rng.uniform(0, 2 * np.pi, F)))
for m in [2, 6]:
    rest = np.setdiff1d(np.arange(F), Sy); ex = rng.choice(rest, min(len(rest), int(round(m * L))), replace=False)
    S = np.sort(np.concatenate([Sy, ex])); A = np.zeros(F); A[Sy] = y; A[ex] = y.mean() * 0.7
    SD.append((f'ystar+{m}L', S, A, rng.uniform(0, 2 * np.pi, F)))
K = int(round(8 * L)); d = int(rng.integers(1, p)); l0 = int(rng.integers(1, p))
im = fold([(l0 + k * d) % p for k in range(K)], p); S = np.array(sorted(im)); A = np.zeros(F); A[S] = 1
th = np.zeros(F)
for k, (i, sg) in enumerate(im.items()): th[i] = sg * math.pi * k * k / K
SD.append(('AP8L-chirp', S, A, th))
t0 = time.time(); res = {}; cands = []
def upd(k, val, cfg, name, larger):
    if k not in res or (val > res[k][0] if larger else val < res[k][0]): res[k] = (val, name, cfg)
for name, S, A, TH in SD:
    a, v, th = optimise_ws(M, S, A + 1e-12, M.cap * A * 0.9 + 1e-12, TH, mode='crit', stages=(0.03, 0.01, 0.003, 0.001), iters=200, nws=40)
    c1 = M.cstar(a, v, th)[0]; cands.append((c1, name, a, v, th))
    upd('cstar', c1, (a, v, th), name, False)
    line = f'{name:14s} |S|={len(S):4d} c*(grad)={c1:.4f} (supp {np.sum(a > 1e-6 * a.max())}, cheap {np.sum(v > 1e-6 * a.max())})'
    for lvl, ln in [(PAIR, 'pair'), (M.cap, 'cap')]:
        a, v, th = optimise_ws(M, S, A + 1e-12, lvl * A + 1e-12, TH, mode='ratio', fixv=lvl, c=CS, stages=(0.03, 0.01, 0.003, 0.001), iters=200, nws=40)
        R = M.ratio(a, v, th, CS)
        upd(f'forced_{ln}', R, (a, v, th), name, True)
        line += f'  {ln}/sharp={R:.4f} ({M.ratio(a, v, th, CA):.4f} at c=0.289)'
    print(line + f'  [{time.time() - t0:.0f}s]', flush=True)
# LP polish of the best candidates
cands.sort(key=lambda z: z[0])
for c1, name, a, v, th in cands[:3]:
    try:
        c2, a2, v2 = crit_polish(M, a, v, th, tol=4e-3, rounds=4)
        print(f'polish {name}: {c1:.4f} -> {c2:.4f}  [{time.time() - t0:.0f}s]', flush=True)
        upd('cstar', c2, (a2, v2, th), name, False)
    except Exception as e: print('polish fail', e)
for ln, lvl in [('pair', PAIR), ('cap', M.cap)]:
    R, name, (a, v, th) = res[f'forced_{ln}']
    try:
        R2, a2, v2 = lp_polish(M, a, v, th, fixv=lvl, c=CS, rounds=4)
        if R2 > R: res[f'forced_{ln}'] = (R2, name, (a2, v2, th))
    except Exception: pass
print(f'== p={p} Gamma={G:.6f}', flush=True)
for k, (val, name, (a, v, th)) in sorted(res.items()):
    print(f'  {k:12s} {val:.4f} from {name}: {M.info(a, v, th, CS)}  ratio@0.289={M.ratio(a, v, th, CA):.4f}', flush=True)
os.makedirs('res', exist_ok=True)
np.save(f'res/run3_p{p}.npy', {k: (val, name, [np.asarray(x).tolist() for x in cfg]) for k, (val, name, cfg) in res.items()}, allow_pickle=True)
