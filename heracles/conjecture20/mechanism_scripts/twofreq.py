# p>=5: networks of tilted sign-vector neurons (type A: sine pattern, type B: all positive), outputs on chosen
# frequencies.  Exact LP (joint.lp_joint) over all count classes and all shifts; CMA over slopes.
import math, sys, time, numpy as np, cma, warnings
sys.path.insert(0, '.')
from joint import lp_joint
warnings.filterwarnings("ignore")

def inputs(lamA, lamB, p):
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
    U = [eps * np.exp(l * e) for l in lamA] + [np.exp(l * e) for l in lamB]
    U = np.array(U); return U / np.linalg.norm(U, axis=1, keepdims=True)

def val(x, mA, p, n, freqs):
    r = lp_joint(inputs(x[:mA], x[mA:], p), p, n, freqs=freqs)
    return -1.0 if r is None else r[0]

def search(mA, mB, p, n, freqs, R=4, iters=40, seed=0):
    rng = np.random.default_rng(seed); best = (-1.0, None)
    for r in range(R):
        x0 = np.concatenate([np.sort(rng.uniform(-2, 2, mA)), np.sort(rng.uniform(-2, 2, mB))])
        es = cma.CMAEvolutionStrategy(x0, 0.5, {'seed': seed + r + 1, 'verbose': -9, 'popsize': 12 + 2 * (mA + mB), 'maxiter': iters})
        while not es.stop():
            X = es.ask(); F = [val(np.array(xx), mA, p, n, freqs) for xx in X]; es.tell(X, [-f for f in F])
            k = int(np.argmax(F))
            if F[k] > best[0]: best = (F[k], np.array(X[k]))
    return best

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    for m in [int(v) for v in sys.argv[3].split(",")]:
        for freqs in [(1,), (1, 2)]:
            out = []
            for mB in range(0, m + 1) if len(freqs) == 2 else [0, m // 2]:
                v = search(m - mB, mB, p, n, freqs, R=3, iters=35)
                out.append((round(v[0], 4), m - mB, mB))
            out.sort(reverse=True)
            print("p=%d n=%d m=%d freqs=%s : best (eff, mA, mB) %s  (%.0fs)" % (p, n, m, freqs, out[:3], time.time() - t0), flush=True)
