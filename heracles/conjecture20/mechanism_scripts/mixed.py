# 1-D family with both atom types: sine-pattern atoms e^{lam tau} and positive-input atoms (-1)^tau e^{lam tau}
# (both with the same tilt cost Z(lam)^n).  Exact LP for coefficients; CMA over slopes of both types.
import math, sys, time, numpy as np, cma, warnings
from scipy.optimize import linprog
sys.path.insert(0, '.')
from rw import logZ
from oned import lref
warnings.filterwarnings("ignore")
K = 16

def lp_mixed(lamA, lamB, p, n, b=2.0):
    lam = np.concatenate([lamA, lamB]); sg = np.concatenate([np.zeros(len(lamA)), np.ones(len(lamB))])
    J = len(lam); L = (p - 1) * n; th = math.pi / p; cot = 1 / math.tan(th)
    tau = np.arange(L + 1)
    lz = np.array([logZ(l, p) for l in lam])
    logG = lam[None, :] * tau[:, None] - n * lz[None, :] - 0.5 * math.log(p / 2)
    colmax = logG.max(0); logG = logG - colmax[None, :]
    rowmax = logG.max(1); G = np.exp(logG - rowmax[:, None])
    G = G * np.where((tau[:, None] % 2 == 1) & (sg[None, :] == 1), -1.0, 1.0)
    C, S = np.cos(th * tau)[:, None], np.sin(th * tau)[:, None]
    ReX, ReY, ImX, ImY = G * C, -G * S, G * S, G * C
    crow = np.exp(-rowmax); cs = crow.max(); crow = crow / cs
    A1 = np.hstack([-(ReX - cot * ImX), -(ReY - cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    A2 = np.hstack([-(ReX + cot * ImX), -(ReY + cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    ph = 2 * math.pi * np.arange(K) / K
    A3 = np.vstack([np.hstack([np.cos(ph[k]) * np.eye(J), np.sin(ph[k]) * np.eye(J), -np.eye(J), np.zeros((J, 1))]) for k in range(K)])
    A4 = np.concatenate([np.zeros(2 * J), np.exp(-colmax), [0.0]])[None, :]
    A = np.vstack([A1, A2, A3, A4]); bvec = np.concatenate([np.zeros(2 * (L + 1) + K * J), [1.0]])
    cobj = np.zeros(3 * J + 1); cobj[-1] = -1
    res = linprog(cobj, A_ub=A, b_ub=bvec, bounds=[(None, None)] * (2 * J) + [(0, None)] * J + [(None, None)], method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0: return -1.0, None
    z = res.x; X, Y = z[:J], z[J:2 * J]
    Vt = (X + 1j * Y) * np.exp(-colmax)            # cost-amplitude coefficients
    a = np.abs(Vt)
    if a.max() <= 0: return -1.0, None
    keep = a > 1e-9 * a.max()
    # true field (float) and l_r efficiency
    V = Vt / (math.sqrt(p / 2) * np.exp(n * lz))
    E = np.log(np.abs(V[keep]) + 1e-300)[None, :] + lam[keep][None, :] * tau[:, None]
    Em = E.max(1, keepdims=True)
    sgn = np.where((tau[:, None] % 2 == 1) & (sg[keep][None, :] == 1), -1.0, 1.0)
    F = (np.exp(E - Em) * sgn * np.exp(1j * (np.angle(V[keep])[None, :] + th * tau[:, None]))).sum(1)
    marg = (F.real - cot * np.abs(F.imag)) * np.exp(Em[:, 0])
    nu = n + 1; s = nu / b
    P = nu ** (b / 2) * np.sum(a[keep] ** (b / nu))
    return marg.min() / (P ** s * math.exp(lref(p, n, b))), (lam[keep], sg[keep], Vt[keep])

def cma_mixed(lamA, lamB, p, n, iters=50, sigma=0.12, seed=1):
    mA = len(lamA); x0 = np.concatenate([lamA, lamB])
    f = lambda x: lp_mixed(np.sort(x[:mA]), np.sort(x[mA:]), p, n)[0]
    best = (f(x0), x0)
    es = cma.CMAEvolutionStrategy(x0, sigma, {'seed': seed, 'verbose': -9, 'popsize': 10 + 2 * len(x0), 'maxiter': iters})
    while not es.stop():
        X = es.ask(); F = [f(np.array(x)) for x in X]; es.tell(X, [-v for v in F])
        k = int(np.argmax(F))
        if F[k] > best[0]: best = (F[k], np.array(X[k]))
    return best

if __name__ == "__main__":
    p, n, f = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    x = np.load(f); m = len(x) // 3; lamA = np.sort(x[:m]); t0 = time.time()
    e0 = lp_mixed(lamA, np.array([]), p, n)[0]
    print("p=%d n=%d: A-only (m=%d) %.5f" % (p, n, m, e0), flush=True)
    rng = np.random.default_rng(0)
    for mB in [1, 2, 3]:
        best = (-1, None)
        for trial in range(3):
            lamB = np.sort(rng.choice(lamA, mB, replace=False) + 0.1 * rng.standard_normal(mB))
            for dropA in [0, 1]:
                lA = lamA if dropA == 0 else np.delete(lamA, rng.integers(m))
                v = cma_mixed(lA, lamB, p, n, iters=40)
                if v[0] > best[0]: best = (v[0], v[1], len(lA))
        print("   + %d positive-input atoms (A width %d, total %d): %.5f  (%.0fs)" % (mB, best[2], best[2] + mB, best[0], time.time() - t0), flush=True)
