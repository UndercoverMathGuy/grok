import math, sys, time, numpy as np, cma, warnings
sys.path.insert(0, '.')
from joint import inputs, lp_joint
warnings.filterwarnings("ignore")
p = 5

def val(lam, n, freqs):
    r = lp_joint(inputs(np.sort(lam), p, 1), p, n, freqs=freqs)
    return (-1.0, 0, None) if r is None else (r[0], r[2], r[3])

if __name__ == "__main__":
    n = int(sys.argv[1]); f = sys.argv[2]; iters = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    x = np.load(f); m = len(x) // 3; lam0 = np.sort(x[:m]); t0 = time.time()
    v1 = val(lam0, n, (1,)); v2 = val(lam0, n, (1, 2))
    print("n=%d m=%d start: single-freq %.5f  joint %.5f" % (n, m, v1[0], v2[0]), flush=True)
    best = (v2[0], lam0, v2[1])
    es = cma.CMAEvolutionStrategy(lam0, 0.1, {'seed': 3, 'verbose': -9, 'popsize': 8 + 2 * m, 'maxiter': iters})
    while not es.stop():
        X = es.ask(); F = []
        for xx in X:
            v = val(np.array(xx), n, (1, 2)); F.append(-v[0])
            if v[0] > best[0]: best = (v[0], np.sort(np.array(xx)), v[1])
        es.tell(X, F)
        print("   iter %d best joint %.5f (freq2 frac %.3f) (%.0fs)" % (es.countiter, best[0], best[2], time.time() - t0), flush=True)
    vs = val(best[1], n, (1,))
    print("n=%d m=%d FINAL joint %.5f  (single-freq at same slopes %.5f)  (%.0fs)" % (n, m, best[0], vs[0], time.time() - t0), flush=True)
    np.save("joint_p5_n%d_m%d.npy" % (n, m), best[1])
