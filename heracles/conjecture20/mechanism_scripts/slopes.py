# Outer search over slope sets (CMA-ES), inner exact LP for coefficients; true l_r efficiency as objective.
import math, sys, time, numpy as np, torch, cma, warnings
sys.path.insert(0, '.')
from elim import lp, true_eff
from elim2 import path
torch.set_num_threads(1); warnings.filterwarnings("ignore")

def value(lam, p, n, b, rew=1):
    w = np.ones(len(lam)); best = (-1e9, None)
    for it in range(rew + 1):
        res = lp(np.sort(lam), p, n, w)
        if res is None or res[0] <= 0: return best
        c, Vt, lz = res; a = np.abs(Vt); keep = a > 1e-9 * a.max()
        e, x = true_eff(np.sort(lam), Vt, lz, p, n, b, keep)
        if e > best[0]: best = (e, x)
        w = (a / a.max() + 1e-3) ** (b / (n + 1) - 1)
    return best

def cma_slopes(lam0, p, n, b, sigma=0.15, iters=60, pop=None, seed=0):
    m = len(lam0)
    es = cma.CMAEvolutionStrategy(np.sort(lam0), sigma, {'seed': seed + 1, 'verbose': -9, 'popsize': pop or (8 + 2 * m), 'maxiter': iters})
    best = value(lam0, p, n, b)
    while not es.stop():
        X = es.ask(); F = []
        for xx in X:
            v = value(np.array(xx), p, n, b, rew=0)
            F.append(-v[0])
            if v[0] > best[0]: best = v
        es.tell(X, F)
    return best

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); b = float(sys.argv[3]); R = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(seed); t0 = time.time()
    th = math.pi / p; L = (p - 1) * n
    mD = int(math.ceil(1 + (th * L - 2 * th) / math.pi - 1e-9))
    starts = {}
    for k in range(R):
        J = int(rng.integers(60, 260))
        tg = np.sort(rng.uniform(0.002, p - 1 - 0.002, J)) if k % 2 else np.linspace(0.002, p - 1 - 0.002, J)
        for m, e, x in path(p, n, b, tg, rng, max(mD - 1, 2), drop_rand=0.3 if k % 3 == 2 else 0.0):
            if e > 0 and m <= min(n + 1, mD + 3):
                starts.setdefault(m, []).append((e, x[:m].copy()))
    out = {}
    for m in sorted(starts):
        cand = sorted(starts[m], key=lambda t: -t[0])[:3]
        bestm = (-1e9, None)
        for e0, lam0 in cand:
            v = cma_slopes(lam0, p, n, b, seed=seed)
            if v[0] > bestm[0]: bestm = v
        out[m] = bestm
        print("   p=%d n=%d m=%d: elim %.4f -> CMA %.5f (%.0fs)" % (p, n, m, cand[0][0], bestm[0], time.time() - t0), flush=True)
        np.save("sl_p%d_n%d_m%d.npy" % (p, n, m), bestm[1])
    mb = max(out, key=lambda k: out[k][0])
    print("p=%d n=%d nu=%d mD=%d  BEST eff %.5f at m=%d   all: %s" % (p, n, n + 1, mD, out[mb][0], mb, sorted([(m, round(v[0], 4)) for m, v in out.items()])), flush=True)
