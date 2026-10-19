# Minimal width for positive margin: slot-symmetric networks, inputs U (m x p) free, outputs on given frequencies.
# Inner LP: max h s.t. gap(N,delta) >= h * scale(N) for all count classes N and shifts delta, box |z| <= 1.
import math, sys, time, numpy as np, cma, warnings
from scipy.optimize import linprog
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import counts
warnings.filterwarnings("ignore")
_cache = {}
def classes(n, p):
    if (n, p) not in _cache:
        N, y, _ = counts(n, p); _cache[(n, p)] = (N.numpy(), y.numpy())
    return _cache[(n, p)]

def basis(p, freqs):
    cs = np.arange(p); Bs = []
    for l in freqs:
        for f in (np.cos, np.sin):
            v = f(2 * math.pi * l * cs / p); Bs.append(v / np.linalg.norm(v))
    return np.array(Bs).T

def lp_margin(U, p, n, freqs):
    N, y = classes(n, p); m = len(U)
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    with np.errstate(divide='ignore'):
        logP = N @ np.log(np.abs(Un) + 1e-300).T
    sgn = (-1.0) ** (N @ (Un < 0).T)
    rowmax = logP.max(1); Pm = sgn * np.exp(logP - rowmax[:, None])
    B = basis(p, freqs); d = B.shape[1]
    rows = []
    for delta in range(1, p):
        D = B[y] - B[(y - delta) % p]
        rows.append((Pm[:, :, None] * D[:, None, :]).reshape(len(y), m * d))
    A = np.vstack(rows); nz = m * d
    Aub = np.hstack([-A, np.ones((A.shape[0], 1))])
    Aeq = np.r_[A.mean(0), 0.0][None, :]                     # normalisation: mean scaled gap = 1
    res = linprog(np.r_[np.zeros(nz), -1.0], A_ub=Aub, b_ub=np.zeros(A.shape[0]), A_eq=Aeq, b_eq=[1.0],
                  bounds=[(None, None)] * nz + [(None, None)], method="highs")
    return res.x[-1] if res.status == 0 else -1.0

def search(m, p, n, freqs, R=4, iters=60, seed=0):
    rng = np.random.default_rng(seed); best = -9
    for r in range(R):
        x0 = rng.standard_normal(m * p)
        es = cma.CMAEvolutionStrategy(x0, 0.6, {'seed': seed + r + 1, 'verbose': -9, 'popsize': 16 + m, 'maxiter': iters})
        while not es.stop():
            X = es.ask(); F = [lp_margin(np.array(x).reshape(m, p), p, n, freqs) for x in X]
            es.tell(X, [-f for f in F]); best = max(best, max(F))
            if best > 1e-6: return best
    return best

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    allf = tuple(range(1, (p - 1) // 2 + 1))
    for m in [int(v) for v in sys.argv[3].split(",")]:
        r1 = search(m, p, n, (1,)); r2 = search(m, p, n, allf)
        print("p=%d n=%d nu=%d m=%d: best normalised margin  freq1-only %.2e   all freqs %.2e   (%.0fs)" % (p, n, n + 1, m, r1, r2, time.time() - t0), flush=True)
