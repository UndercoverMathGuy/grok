# s near 1: best realisable all-pair network ratio (v=(3/2)^{s-1} a on the support, sharp c) and c*, structured seeds.
import numpy as np, sys, time, math, torch
from gget import gamma_table
from relmodel import Model, optimise, lp_polish, crit_polish
from seeds import seeds
from nearthree import programme, coset_sets
torch.set_num_threads(1)
s = float(sys.argv[1]); CS = 1 - (2 / 3) ** (s - 1); PAIR = 1.5 ** (s - 1)
for p in [int(a) for a in sys.argv[2].split(',')]:
    F = (p - 1) // 2
    sup = [sorted(idx.keys()) for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20] for co, idx in coset_sets(p, k)[:3]]
    G1, y1 = programme(p, s, sup, starts=6)
    T = gamma_table(s).get(p); G = max(G1, T['Gamma'] if T else 0)
    ys = (np.nonzero(y1)[0] + 1, y1[y1 > 0]) if G1 >= G else (T['S'], np.array(T['y']))
    M = Model(p, s, CS, G); rng = np.random.default_rng(p); t0 = time.time()
    SD = seeds(p, rng, ys, nrand=1)
    for k in [3, 4, 5, 6, 8]:
        for co, idx in coset_sets(p, k)[:1]:
            S = np.array(sorted(idx.keys())); d = (p - 1) // k
            for m in [1, d // 2]:
                th = np.zeros(F)
                for i, (j, sg) in idx.items(): th[i] = sg * 2 * np.pi * m * j / d
                A = np.zeros(F); A[S] = 1; SD.append((f'coset{k}-chi{m}', S, A, th))
    bestR = (-9, None); bestC = (9, None)
    for name, S, A, TH in SD:
        S = np.array(S)
        a, v, th = optimise(M, S, A + 1e-12, PAIR * A + 1e-12, TH, mode='ratio', fixv=PAIR, c=CS, stages=(0.02, 0.005, 0.001))
        R = M.ratio(a, v, th, CS)
        if R > bestR[0]: bestR = (R, name, a, v, th)
        a, v, th = optimise(M, S, A + 1e-12, M.cap * A * 0.9 + 1e-12, TH, mode='crit', stages=(0.02, 0.005, 0.001))
        c1 = M.cstar(a, v, th)[0]
        if np.isfinite(c1) and c1 < bestC[0]: bestC = (c1, name)
    R, name, a, v, th = bestR
    R2, a2, v2 = lp_polish(M, a, v, th, fixv=PAIR, c=CS, rounds=6)
    print(f'== s={s} p={p} Gamma={G:.5f} (K*={len(ys[0])}/{F})  best all-pair network ratio {max(R, R2):.4f} [{name}]   best c*={bestC[0]:.4f} [{bestC[1]}] vs c_sharp={CS:.4f} (x{bestC[0] / CS:.2f})  [{time.time() - t0:.0f}s]', flush=True)
