# Task 3: full-network check of relations-model configurations at p=31..61 (b=2).
import numpy as np, sys
from scipy.optimize import minimize
from gget import gamma_table
from relmodel import Model, optimise, lp_polish
from netcheck import build, gamma, gamma_if, margin
CS = 1 - (2 / 3) ** 0.5; PAIR = 1.5 ** 0.5
for p in [int(x) for x in sys.argv[1:]]:
    T = gamma_table()[p]; G = T['Gamma']; M = Model(p, 1.5, CS, G); F = M.F
    gif = gamma_if(p, G)
    # (0) IF network on the programme support
    y = np.zeros(F); y[np.array(T['S']) - 1] = T['y']
    N = build(p, [('IF', l + 1, y[l], 0.0, 0.0) for l in np.nonzero(y)[0]])
    h, P, g = gamma(p, N); print(f'p={p} IF network on y*: gamma/gamma_IF = {g / gif:.6f}', flush=True)
    # configurations: (A) best all-pair config from run.py; (B) y* with one feature switched to a pair (each choice)
    cfgs = []
    try:
        R = np.load(f'res/run_p{p}.npy', allow_pickle=True).item()
        val, name, (a, v, th) = R['forced_pair_sh']; cfgs.append((f'all-pair ({name})', np.array(a), np.array(v), np.array(th)))
    except Exception as e: print('no run file', e)
    rng = np.random.default_rng(0)
    for j in np.nonzero(y)[0]:
        # re-optimise the model with feature j forced to be a pair, others IF (fixv only on j): use LP over (a) with v_j = PAIR a_j
        best = None
        for t in range(6):
            th = rng.uniform(0, 2 * np.pi, F)
            a0 = y.copy(); v0 = np.zeros(F); v0[j] = PAIR * a0[j]
            # model ratio for this structure: optimise allocation a (support of y*) by LP with v_j tied, phase irrelevant (single cheap)
            S = np.nonzero(y)[0]
            from relmodel import linprog
            R0 = M.ratio(a0, v0, th)
            if best is None or R0 > best[0]: best = (R0, a0, v0, th)
        cfgs.append((f'y* with feature {j + 1} as pair', best[1], best[2], best[3]))
    for name, a, v, th in cfgs:
        Rm = M.ratio(a, v, th)
        feats = []
        for l in np.nonzero(a > 1e-9 * a.max())[0]:
            if v[l] > 1e-9 * a.max(): feats.append(['pair', l + 1, v[l], th[l], 0.0])
            else: feats.append(['IF', l + 1, a[l], 0.0, 0.0])
        h, P, g = gamma(p, build(p, [tuple(f) for f in feats]))
        r0 = g / gif
        # optimise clean-phase rotations eps (allowed by the network, approximates the osc/2 centring)
        idx = [i for i, f in enumerate(feats)]
        def negr(e):
            ff = [tuple(f[:4] + [e[i]]) for i, f in enumerate(feats)]
            return -gamma(p, build(p, ff))[2] / gif
        res = minimize(negr, np.zeros(len(feats)), method='Nelder-Mead', options={'maxiter': 400 if p <= 31 else 200, 'xatol': 1e-5, 'fatol': 1e-7})
        print(f'  {name:32s} model ratio (osc/2) = {Rm:.4f} | network exact: eps=0 {r0:.4f}, eps optimised {-res.fun:.4f}  '
              f'[{len(feats)} features, {sum(f[0] == "pair" for f in feats)} pairs]', flush=True)
