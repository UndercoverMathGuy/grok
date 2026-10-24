# Global search seeded by IF blocks on every frequency subset S (nu neurons per frequency), plus perturbations.
import math, sys, time, itertools, numpy as np, cma, warnings
sys.path.insert(0, '.')
from globaleff import Prob
warnings.filterwarnings("ignore")

def if_inputs(p, n, S):
    nu = n + 1; a = np.arange(p)
    return np.array([np.cos(2 * math.pi * j * a / p + math.pi * l / nu) for j in S for l in range(nu)])

def report(pr, best, tag, t0):
    U, W, keep = best[1]
    A = pr.amplitudes(U[keep], W[keep]); sh = A ** (2 / (pr.n + 1)); sh = sh / sh.sum()
    print("  %-30s eff %.5f  width %2d  shares %s  (%.0fs)" % (tag, best[0], keep.sum(), np.round(sh, 4), time.time() - t0), flush=True)

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    pr = Prob(p, n); F = list(range(1, (p - 1) // 2 + 1)); rng = np.random.default_rng(0)
    overall = (-1, None, '')
    for k in range(1, len(F) + 1):
        for S in itertools.combinations(F, k):
            U0 = if_inputs(p, n, S)
            b0 = pr.solve(U0, rew=4)
            if b0[1] is not None: report(pr, b0, "IF block S=%s" % (S,), t0)
            best = b0; m = len(U0)
            for trial in range(2):
                x0 = (U0 + 0.15 * rng.standard_normal(U0.shape)).ravel()
                es = cma.CMAEvolutionStrategy(x0, 0.15, {'seed': trial + 1, 'verbose': -9, 'popsize': 12 + m, 'maxiter': 50})
                while not es.stop():
                    X = es.ask(); Fv = []
                    for x in X:
                        v = pr.solve(np.array(x).reshape(m, p), rew=1); Fv.append(v[0] if v[0] > -np.inf else -1.0)
                        if v[0] > best[0]: best = v
                    es.tell(X, [-f for f in Fv])
            if best[1] is not None:
                v = pr.solve(best[1][0], rew=6); best = v if v[0] > best[0] else best
                report(pr, best, "after CMA from S=%s" % (S,), t0)
                if best[0] > overall[0]: overall = (best[0], best, str(S))
    print("p=%d n=%d BEST overall eff %.5f (seeded from S=%s)" % (p, n, overall[0], overall[2]), flush=True)
