# Minimal width by elimination from feasible starts; general slot-symmetric inputs; LP over outputs.
import math, sys, time, numpy as np, cma, warnings
sys.path.insert(0, '.')
from genwidth import lp_margin
warnings.filterwarnings("ignore")

def local(U, p, n, freqs, sigma=0.15, iters=60, seed=0):
    m = len(U); best = (lp_margin(U, p, n, freqs), U)
    es = cma.CMAEvolutionStrategy(U.ravel(), sigma, {'seed': seed + 1, 'verbose': -9, 'popsize': 14 + m, 'maxiter': iters})
    while not es.stop():
        X = es.ask(); F = [lp_margin(np.array(x).reshape(m, p), p, n, freqs) for x in X]
        es.tell(X, [-f for f in F]); k = int(np.argmax(F))
        if F[k] > best[0]: best = (F[k], np.array(X[k]).reshape(m, p))
    return best

def starts(p, n, rng):
    nu = n + 1; a = np.arange(p); S = []
    S.append(np.array([np.cos(2 * math.pi * a / p + math.pi * l / nu) for l in range(nu)]))     # IF block
    eps = np.array([1.0 if x < p / 2 else -1.0 for x in a]); e = np.array([2 * x if x < p / 2 else 2 * x - p for x in a], float)
    for spread in (1.0, 1.6):
        lam = np.linspace(-spread, spread, nu)
        S.append(np.array([eps * np.exp(l * e) for l in lam]))                                    # tilted sign vectors
    return S

def eliminate(U, p, n, freqs, rng):
    path = []
    cur = local(U, p, n, freqs, sigma=0.05, iters=30)
    while True:
        path.append((len(cur[1]), cur[0]))
        if cur[0] <= 1e-7 or len(cur[1]) <= 2: break
        cands = []
        for i in range(len(cur[1])):
            V = np.delete(cur[1], i, axis=0); cands.append((lp_margin(V, p, n, freqs), V))
        cands.sort(key=lambda t: -t[0])
        best = (-9, None)
        for v, V in cands[:2]:
            r = local(V, p, n, freqs, sigma=0.15, iters=50, seed=len(V))
            if r[0] > best[0]: best = r
        cur = best
    return path

if __name__ == "__main__":
    p = int(sys.argv[1]); t0 = time.time()
    for n in [int(v) for v in sys.argv[2].split(",")]:
        rng = np.random.default_rng(n); nu = n + 1
        for freqs in [(1,), tuple(range(1, (p - 1) // 2 + 1))]:
            mins = []
            for S in starts(p, n, rng):
                path = eliminate(S, p, n, freqs, rng)
                feas = [m for m, v in path if v > 1e-7]
                mins.append(min(feas) if feas else None)
            print("p=%d n=%d nu=%d freqs=%s: minimal feasible width per start %s   (bound nu-floor(nu/p)=%d) (%.0fs)" % (p, n, nu, freqs, mins, nu - nu // p, time.time() - t0), flush=True)
