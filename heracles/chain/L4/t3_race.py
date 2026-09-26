# The race (coordinator's pivot): on structured heavy sets T, is the achievable aim depth >= the depth required under a
# share floor eps?  Achievable: flip-enhanced transfer c(A*, A_s) from grid-LP LOWER bounds on the one-sided (A*) and
# two-sided (A_s) K-mode depths [rigorous modulo LP tolerance + transfer constant computed on a sphere grid], plus a search
# UPPER estimate of the true flip-enlarged l2-ball depth. Required: a_req = (1 - delta_crit/sqrt3)/sqrt3 with delta_crit from
# L5/floor_test.value (SLSQP lower bounds on the value -> delta_crit over-estimated -> a_req UNDER-estimated).
import sys, itertools, numpy as np
from scipy.optimize import linprog, minimize
sys.path.insert(0, '..'); sys.path.insert(0, '../L5')
from transfer_flip import depth as transfer_depth
from floor_test import value as floor_value
rng = np.random.default_rng(3)

def grid_depth(p, S, sym, Lg):
    K = len(S); x = np.arange(p); chi = np.exp(2j * np.pi * 2 * np.outer(S, x) / p)
    if sym: chi = np.hstack([chi, -chi])
    n = chi.shape[1]; Ab = np.vstack([chi.real, chi.imag, np.ones(n)]); tmin = 9
    for idx in itertools.product(range(Lg), repeat=K - 1):
        t = np.r_[0.0, 2 * np.pi * np.array(idx) / Lg]
        r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Ab, -np.r_[np.cos(t), np.sin(t), 0.0]]),
                    b_eq=np.r_[np.zeros(2 * K), 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
        tmin = min(tmin, -r.fun)
    return np.cos(np.pi / Lg) * tmin

def enlarged_search(p, S, extra=12):
    K = len(S); x = np.arange(p); ze = np.exp(-2j * np.pi * 2 * np.outer(S, x) / p); ab = (x[:, None] + x[None, :]) % p
    def h(zc):
        best = -np.inf
        for s1, s2 in ((1, 1), (-1, 1), (1, -1)):
            f1 = s1 * np.real(zc[0] @ ze); f2 = s2 * np.real(zc[1] @ ze); f3 = s1 * s2 * np.real(zc[2] @ ze)
            best = max(best, (f1[:, None] + f2[None, :] + f3[ab]).max())
        return best
    obj = lambda v: h((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K)) / np.linalg.norm((v[:3 * K] + 1j * v[3 * K:]).reshape(3, K), axis=0).sum()
    I = np.arange(K); starts = []
    for T in range(3):
        for ph in (np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), np.pi * I ** 2 / K, rng.uniform(0, 2 * np.pi, K)):
            zc = np.zeros((3, K), complex); zc[T] = np.exp(1j * ph); starts.append(np.r_[zc.real.ravel(), zc.imag.ravel()])
    starts += [rng.normal(size=6 * K) for _ in range(extra)]
    return min(minimize(obj, s + 1e-3 * rng.normal(size=6 * K), method='Powell', options=dict(maxiter=30000, xtol=1e-8, ftol=1e-10)).fun
               for s in starts)

def delta_crit(p, T, eps, Gam):
    if floor_value(p, T, 1.0, eps, Gam, starts=20) >= 1: return 1.0
    lo, hi = 1.0, np.sqrt(3)
    if floor_value(p, T, hi, eps, Gam, starts=20) <= 1: return hi
    for _ in range(9):
        mid = (lo + hi) / 2
        if floor_value(p, T, mid, eps, Gam, starts=20) > 1: hi = mid
        else: lo = mid
    return (lo + hi) / 2

if __name__ == '__main__':
    p = int(sys.argv[1]); Gam = float(sys.argv[2]); sets = eval(sys.argv[3]); eps_list = [0.03, 0.06]
    for name, T in sets:
        K = len(T); Lg = 6 if K <= 6 else 4
        As = grid_depth(p, T, True, Lg); Ao = grid_depth(p, T, False, Lg)
        c, _ = transfer_depth(As, Ao)
        srch = enlarged_search(p, T)
        reqs = []
        for eps in eps_list:
            d = delta_crit(p, T, eps, Gam); reqs.append((eps, d, max(0.0, (1 - d / np.sqrt(3)) / np.sqrt(3))))
        req_s = "  ".join(f"eps={e}: delta_crit {d:.3f} a_req {a:.3f}" for e, d, a in reqs)
        verdict = "RIGOROUS-ish WIN" if c >= max(a for _, _, a in reqs) else ("plausible (search)" if srch >= max(a for _, _, a in reqs) else "LOSES")
        print(f"p={p} {name:14s} K={K}: A*>={Ao:.3f} A_s>={As:.3f} (grid L={Lg}) -> transfer depth >= {c:.3f}; enlarged depth (search) <= {srch:.3f} | {req_s} | {verdict}", flush=True)
