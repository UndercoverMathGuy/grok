# C_hunt library, part 2 (resumed run).
# (1) prog_bb: EXACT restricted programme Gamma^{(S)}_s(p) by branch and bound (concave minimisation of sum v^r over the
#     coverage polyhedron {v >= 0 : K v >= 1}; secant LP relaxation on boxes), followed by an mpmath re-solve of the
#     optimal vertex.  Replaces the reweighted LP (a lower bound on Gamma only; prog_check.out shows it misses at p >= 29).
# (2) slp_full: sequential-LP minimax polish of a class-C network with the FULL near-active set (the old slp capped the
#     active set at 3000 rows; at an IF point all p^n * 2|B| binding rows tie, so it stalled: this is why the earlier
#     p >= 23 runs ended at 0.989-0.9998 from IF seeds).  Scaled l_inf trust region (relative for b, absolute for phases).
# (3) mass_lp: LP in the neuron masses with phases and atom arguments frozen (iteratively reweighted l1 for the concave
#     cost sum m^r; every step is a global LP over all rows; neurons may be switched off).
import heapq, itertools, math
import numpy as np
from scipy.optimize import linprog
import hlib as H


def kmat(p):
    F = (p - 1) // 2
    d = np.arange(1, F + 1)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)          # rows delta = 1..F (delta and p - delta coincide), cols j = 1..F


def _rw(K, r, restarts, rng):
    best = (np.inf, None)
    k = K.shape[1]
    for t in range(restarts):
        w = np.ones(k) if t == 0 else rng.uniform(0.05, 2.0, k)
        v = None
        for it in range(200):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(K.shape[0]), bounds=(0, None), method='highs')
            if res.status != 0: v = None; break
            v = res.x; wn = (v + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-11): break
            w = wn
        if v is None: continue
        v = v * (v > 1e-11); Q = np.sum(v[v > 0] ** r)
        if Q < best[0]: best = (Q, v.copy())
    return best


def prog_bb(p, s, S=None, tol=1e-9, maxnodes=400000, restarts=30, seed=0, verbose=False):
    """Exact Gamma^{(S)}_s(p) (relative optimality gap <= tol, certified by the B&B lower bound).
    Returns dict(G, y (length F), supp, bind, nodes, gap)."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else sorted(S)
    Kf = kmat(p); K = Kf[:, [j - 1 for j in S]]; k = len(S)
    rng = np.random.default_rng(seed)
    Qinc, vinc = _rw(K, r, restarts, rng)
    if r >= 1 - 1e-12:      # s = 1: linear programme
        res = linprog(np.ones(k), A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
        Qinc, vinc = res.fun, res.x; lb = Qinc; nodes = 1
    else:
        U0 = np.full(k, Qinc ** s * (1 + 1e-9))
        def relax(L, U):
            # secant of v^r on [L, U]: a + c v
            c = np.where(U - L > 1e-15, (U ** r - L ** r) / np.maximum(U - L, 1e-300), r * np.maximum(U, 1e-300) ** (r - 1))
            a = L ** r - c * L
            res = linprog(c, A_ub=-K, b_ub=-np.ones(F), bounds=list(zip(L, U)), method='highs')
            if res.status != 0: return np.inf, None
            return res.fun + a.sum(), res.x
        L0 = np.zeros(k)
        lb0, v0 = relax(L0, U0)
        heap = [(lb0, 0, L0, U0, v0)]; cnt = 1; nodes = 0; lb = lb0
        while heap:
            lbn, _, L, U, v = heapq.heappop(heap)
            lb = lbn
            if lbn >= Qinc * (1 - tol): break
            nodes += 1
            if nodes > maxnodes: break
            qv = np.sum(np.clip(v, 0, None) ** r)
            if qv < Qinc: Qinc, vinc = qv, v.copy()
            c = np.where(U - L > 1e-15, (U ** r - L ** r) / np.maximum(U - L, 1e-300), 0.0)
            gapj = np.clip(v, 0, None) ** r - (L ** r + c * (v - L))
            j = int(np.argmax(gapj))
            if gapj[j] <= 1e-14 * max(Qinc, 1): continue
            split = v[j] if (v[j] - L[j]) > 1e-6 * (U[j] - L[j]) and (U[j] - v[j]) > 1e-6 * (U[j] - L[j]) else 0.5 * (L[j] + U[j])
            for (Ln, Un) in ((L.copy(), U.copy()), (L.copy(), U.copy())):
                pass
            L1, U1 = L.copy(), U.copy(); U1[j] = split
            L2, U2 = L.copy(), U.copy(); L2[j] = split
            for (Ln, Un) in ((L1, U1), (L2, U2)):
                lbc, vc = relax(Ln, Un)
                if vc is None or lbc >= Qinc * (1 - tol): continue
                heapq.heappush(heap, (lbc, cnt, Ln, Un, vc)); cnt += 1
            if verbose and nodes % 2000 == 0: print('bb', nodes, len(heap), lb, Qinc, flush=True)
        if not heap: lb = Qinc
    # polish the incumbent: identify its vertex (support T, binding rows B) and re-solve the square system in mpmath
    v = np.clip(vinc, 0, None); v[v < 1e-12 * v.max()] = 0
    cov = K @ v
    T = np.nonzero(v > 0)[0]; B = np.argsort(cov)[:len(T)]
    try:
        import mpmath as mp
        mp.mp.dps = 40
        M = mp.matrix([[mp.mpf(1) - mp.cos(2 * mp.pi * (int(d) + 1) * S[t] / p) for t in T] for d in B])
        sol = mp.lu_solve(M, mp.matrix([1] * len(T)))
        vv = np.zeros(k); vv[T] = [float(z) for z in sol]
        if np.all(vv[T] > 0) and np.all(K @ vv >= 1 - 1e-12):
            Qm = mp.fsum([mp.mpf(z) ** (mp.mpf(1) / s) for z in sol])
            if float(Qm) <= Qinc * (1 + 1e-12): Qinc = float(Qm); v = vv
    except Exception:
        pass
    G = Qinc ** (-s)
    y = np.zeros(F); y[[S[t] - 1 for t in range(k)]] = v / Qinc ** s
    covy = Kf @ y
    bind = [d + 1 for d in range(F) if covy[d] < G * (1 + 1e-8)]
    return dict(G=G, y=y, supp=[j for j in range(1, F + 1) if y[j - 1] > 0], bind=bind, nodes=nodes,
                gap=max(0.0, 1 - lb / Qinc) if np.isfinite(lb) else None)


# ------------------------------------------------------------------------------------------------ class-C polish
def slp_full(cc, x, iters=400, D0=3e-2, free=None, maxrows=120000, tol=1e-14, verbose=False):
    """Sequential LP for max min_k q_k (q = gap / cost^s), all rows within an adaptive threshold of the minimum.
    Trust region: |d Re b|, |d Im b| <= D * bscale, |d phi| <= D."""
    x = cc.normalise(x); m = cc.m
    fr = np.ones(x.size, bool) if free is None else np.asarray(free, bool)
    best = cc.gnorm(x); D = D0; hist = [best]
    for it in range(iters):
        hist.append(best)
        if it >= 40 and (best - hist[-40]) <= 1e-11 * abs(best): break          # stalled: < 1e-11 relative gain in 40 steps
        b, _ = cc.unpack(x)
        bscale = max(np.abs(b).max(), 1e-12)
        sc = np.r_[np.full(2 * m, bscale), np.ones(m * cc.n)]
        q = cc.gaps(x).ravel() / cc.cost(x) ** cc.s
        qmin = q.min()
        # rows whose value can reach the current minimum under a step of size D: |dq| <= Lip * D, Lip ~ 4 |qmin| (nu+1)
        thr = qmin + 8.0 * (cc.n + 2) * D * (abs(qmin) + 1e-15)
        idx = np.nonzero(q <= thr)[0]
        if idx.size > maxrows: idx = np.argsort(q)[:maxrows]
        qk, J = cc.jac_active(x, idx)
        Js = J * sc[None, :]
        nv = Js.shape[1]
        c = np.zeros(nv + 1); c[-1] = -1.0
        A = np.hstack([-Js, np.ones((idx.size, 1))])
        bnd = [((-D, D) if fr[i] else (0.0, 0.0)) for i in range(nv)] + [(None, None)]
        res = linprog(c, A_ub=A, b_ub=qk, bounds=bnd, method='highs')
        if res.status != 0:
            D *= 0.3
            if D < 1e-12: break
            continue
        pred = res.x[-1] - qmin
        xn = cc.normalise(x + res.x[:nv] * sc)
        val = cc.gnorm(xn)
        if val > best + tol * abs(best):
            rho = (val - best) / max(pred, 1e-300)
            x, best = xn, val
            D = min(D * (2.0 if rho > 0.5 else 1.2), 0.5)
        else:
            D *= 0.35
        if verbose and it % 20 == 0: print(it, best, D, idx.size, flush=True)
        if D < 1e-12: break
    return x, best


def mass_lp(cc, x, iters=25, eps_rel=1e-6, verbose=False):
    """Phases xi and atom arguments frozen; masses m_i >= 0 by iteratively reweighted l1:
       min sum w_i m_i  s.t.  gap(x, delta; m) >= 1 for every row  (gaps are linear in m), w_i = (m_i + eps)^{r-1}.
    Every reweighted problem is an exact LP over ALL rows. Returns the best (by true normalised margin) iterate."""
    b, phi = cc.unpack(x)
    arg = np.exp(1j * np.angle(b))
    # gap rows as a linear map of m: g = sum_i m_i * G_i(x, delta), G_i = Re(arg_i P_i(x) c_l(delta)) (kappa - sin form)
    _, P, _, _, _ = cc.fields(cc.pack(arg, phi))                     # (m, N): unit-mass neuron fields
    Gm = np.zeros((cc.N * (cc.p - 1), cc.m))
    for i in range(cc.m):
        k = cc.fidx[i]
        Gi = (arg[i] * P[i]).real[:, None] * cc.Kd[k][None, :] - (arg[i] * P[i]).imag[:, None] * cc.Sd[k][None, :]
        Gm[:, i] = Gi.ravel()
    m = np.abs(b).copy(); best = (cc.gnorm(x), x.copy())
    scale = m.max()
    for it in range(iters):
        w = (m / scale + eps_rel) ** (cc.r - 1)
        res = linprog(w, A_ub=-Gm, b_ub=-np.ones(Gm.shape[0]), bounds=(0, None), method='highs')
        if res.status != 0: break
        mn = res.x
        xn = cc.normalise(cc.pack(mn * arg, phi))
        v = cc.gnorm(xn)
        if verbose: print('mass_lp', it, v, int((mn > 1e-9 * mn.max()).sum()), flush=True)
        if v > best[0]: best = (v, xn.copy())
        if np.allclose(mn / mn.max(), m / max(m.max(), 1e-300), atol=1e-10): break
        m = mn; scale = m.max()
    return best[1], best[0]


def polish(cc, x, lbfgs=True, rounds=None, verbose=False):
    """smooth-min L-BFGS (coarse) -> [mass LP -> full SLP] x rounds; returns (x, normalised margin)."""
    if lbfgs:
        x = cc.lbfgs(x, taus=(3e-3, 1e-3, 3e-4), iters=600)
    best = (cc.gnorm(x), x)
    if rounds is None: rounds = 2 if cc.m <= 20 else 1
    for r_ in range(rounds):
        x, v = mass_lp(cc, best[1])
        if v > best[0]: best = (v, x)
        x, v = slp_full(cc, best[1], iters=250 if cc.m <= 20 else 150, verbose=verbose)
        if v > best[0]: best = (v, x)
    return best[1], best[0]


# ------------------------------------------------------------------------------------------------ exact checks
def exact_margin_table(U, W, dps=30, interval=True):
    """Margin and cost of a product network recomputed from its full table with mpmath interval arithmetic
    (weights taken as the exact binary rationals they are).  Returns (h_lower, h_upper, cost_lower, cost_upper)."""
    import mpmath as mp
    m, n, p = U.shape
    if interval:
        ctx = mp.iv; ctx.dps = dps
        cv = lambda z: ctx.mpf(float(z))
    else:
        ctx = mp.mp; ctx.dps = dps
        cv = lambda z: mp.mpf(float(z))
    Um = [[[cv(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
    Wm = [[cv(W[i, c]) for c in range(p)] for i in range(m)]
    hl = None; hu = None
    for x in itertools.product(range(p), repeat=n):
        pr = []
        for i in range(m):
            z = Um[i][0][x[0]]
            for t in range(1, n): z = z * Um[i][t][x[t]]
            pr.append(z)
        f = []
        for c in range(p):
            acc = pr[0] * Wm[0][c]
            for i in range(1, m): acc = acc + pr[i] * Wm[i][c]
            f.append(acc)
        y = sum(x) % p
        for c in range(p):
            if c == y: continue
            g = f[y] - f[c]
            lo = g.a if interval else g; hi = g.b if interval else g
            if hl is None or lo < hl: hl = lo
            if hu is None or hi < hu: hu = hi
    cost = []
    for i in range(m):
        acc = ctx.mpf(0)
        for t in range(n):
            for a in range(p): acc = acc + Um[i][t][a] ** 2
        for c in range(p): acc = acc + Wm[i][c] ** 2
        cost.append(acc)
    return hl, hu, cost


def prog_best(p, s, S=None, restarts=200, sweeps=6, verbose=False):
    """Gamma^{(S)} for LARGE S: reweighted-LP incumbent, then exact B&B on every support T u {j} (j in S \\ T) and on
    T u {j, k} for the best few j, iterated to a fixed point ('support local search' with exact inner solves).
    Not a certificate for |S| > 12 (prog_bb gap reported separately); an exact optimum over all supports reachable by
    one/two additions from the incumbent."""
    F = (p - 1) // 2
    S = list(range(1, F + 1)) if S is None else sorted(S)
    if len(S) <= 12:
        return prog_bb(p, s, S=S, maxnodes=200000)
    R = prog_bb(p, s, S=S, maxnodes=200, restarts=restarts)
    best = R
    for sw in range(sweeps):
        T = best['supp']; improved = False
        cands = []
        for j in S:
            if j in T: continue
            Rj = prog_bb(p, s, S=sorted(set(T) | {j}), maxnodes=50000, restarts=4)
            cands.append((Rj['G'], j, Rj))
            if Rj['G'] > best['G'] * (1 + 1e-12):
                best = dict(Rj); improved = True
        if not improved:
            cands.sort(key=lambda z: -z[0])
            for (_, j, _) in cands[:4]:
                for (_, k, _) in cands[:8]:
                    if k <= j: continue
                    Rjk = prog_bb(p, s, S=sorted(set(T) | {j, k}), maxnodes=50000, restarts=4)
                    if Rjk['G'] > best['G'] * (1 + 1e-12): best = dict(Rjk); improved = True
        if verbose: print('sweep', sw, best['G'], best['supp'], flush=True)
        if not improved: break
    best = dict(best); best['gap'] = R['gap']; best['method'] = 'support local search'
    return best
