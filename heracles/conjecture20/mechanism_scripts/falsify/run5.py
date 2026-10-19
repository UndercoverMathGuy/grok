# Focused c* search in the winning family: programme allocation y* (IF) + m*ln(p) extra frequencies, several draws.
import numpy as np, sys, time, math, torch, os
from gget import gamma_table
from relmodel import Model, optimise_ws, crit_polish
torch.set_num_threads(1)
CS = 1 - (2 / 3) ** 0.5
p = int(sys.argv[1]); draws = int(sys.argv[2])
T = gamma_table()[p]; G = T['Gamma']; M = Model(p, 1.5, CS, G); F = M.F; L = math.log(p)
Sy = np.array(T['S']) - 1; y = np.array(T['y']); rest = np.setdiff1d(np.arange(F), Sy)
rng = np.random.default_rng(p * 3 + 1); cands = []; t0 = time.time()
for m in [3, 4, 5, 6, 8, 10, 12]:
    k = min(len(rest), int(round(m * L)))
    for dr in range(draws):
        for wext in [0.5, 1.0]:
            ex = rng.choice(rest, k, replace=False); S = np.sort(np.concatenate([Sy, ex]))
            A = np.zeros(F); A[Sy] = y; A[ex] = y.mean() * wext
            a, v, th = optimise_ws(M, S, A + 1e-12, M.cap * A * 0.9 + 1e-12, rng.uniform(0, 2 * np.pi, F), mode='crit',
                                   stages=(0.03, 0.01, 0.003, 0.001), iters=200, nws=40)
            c1 = M.cstar(a, v, th)[0]; cands.append((c1, f'ystar+{m}L/d{dr}/w{wext}', a, v, th))
            print(f'm={m} draw={dr} w={wext}: c*={c1:.4f} supp={np.sum(a > 1e-6 * a.max())} cheap={np.sum(v > 1e-6 * a.max())} [{time.time() - t0:.0f}s]', flush=True)
cands.sort(key=lambda z: z[0]); best = cands[0]
for c1, name, a, v, th in cands[:3]:
    c2, a2, v2 = crit_polish(M, a, v, th, tol=3e-3, rounds=4)
    print(f'polish {name}: {c1:.4f} -> {c2:.4f}', flush=True)
    if c2 < best[0]: best = (c2, name, a2, v2, th)
c, name, a, v, th = best
print(f'== p={p} best c* in y*+extras family: {c:.4f} ({name})', flush=True)
os.makedirs('res', exist_ok=True)
np.save(f'res/run2_p{p}_y.npy', dict(c=c, name=name, a=a, v=v, th=th), allow_pickle=True)
