# Continuation in n over slope sets (inner exact LP for coefficients); moves: keep, insert (m+1), delete (m-1).
import math, sys, time, numpy as np, torch, warnings
sys.path.insert(0, '.')
from slopes import value, cma_slopes
torch.set_num_threads(1); warnings.filterwarnings("ignore")

def insert_moves(lam, k=3):
    l = np.sort(lam); g = np.diff(l); out = []
    for j in np.argsort(-g)[:k]:
        out.append(np.sort(np.append(l, 0.5 * (l[j] + l[j + 1]))))
    out.append(np.sort(np.append(l, l[0] - (l[1] - l[0]))))
    out.append(np.sort(np.append(l, l[-1] + (l[-1] - l[-2]))))
    return out

def delete_moves(lam, k=3):
    l = np.sort(lam); out = []
    g = np.diff(l); cand = np.argsort(g)[:k]
    for j in cand:
        out.append(np.delete(l, j + 1 if j + 1 < len(l) - 1 else j))
    return out

if __name__ == "__main__":
    p, n0, n1, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    init = sys.argv[5]                     # npy from slopes.py (params of best at n0)
    keepper = 2; t0 = time.time()
    x0 = np.load(init); m0 = len(x0) // 3
    pool = {m0: [(value(x0[:m0], p, n0, b)[0], np.sort(x0[:m0]))]}
    print("start p=%d n=%d m=%d eff %.5f" % (p, n0, m0, pool[m0][0][0]), flush=True)
    for n in range(n0 + 1, n1 + 1):
        cands = []
        for m, lst in pool.items():
            for e, lam in lst:
                cands.append(lam)
                cands += insert_moves(lam)
                if m > 3: cands += delete_moves(lam)
        # fresh global candidates from elimination paths (guards against losing the branch)
        from elim2 import path
        rr = np.random.default_rng(n)
        th = math.pi / p; L = (p - 1) * n
        mD = int(math.ceil(1 + (th * L - 2 * th) / math.pi - 1e-9))
        for k in range(3):
            J = int(rr.integers(80, 240))
            tg = np.sort(rr.uniform(0.002, p - 1 - 0.002, J)) if k % 2 else np.linspace(0.002, p - 1 - 0.002, J)
            for mm, e, x in path(p, n, b, tg, rr, max(mD - 1, 2)):
                if e > 0 and mm <= n + 2: cands.append(np.sort(x[:mm]))
        res = {}
        for lam in cands:
            v = value(lam, p, n, b)
            if v[1] is None: continue
            m = len(lam)
            res.setdefault(m, []).append((v[0], lam))
        newpool = {}
        for m, lst in res.items():
            lst.sort(key=lambda t: -t[0])
            top = []
            for e, lam in lst[:keepper]:
                vv = cma_slopes(lam, p, n, b, sigma=0.08, iters=40)
                if vv[1] is not None and vv[0] > 0:
                    top.append((vv[0], np.sort(vv[1][:m])))
                    np.save("cs_p%d_n%d_m%d.npy" % (p, n, m), vv[1])
            if top: newpool[m] = sorted(top, key=lambda t: -t[0])[:keepper]
        # keep the best 3 widths
        ms = sorted(newpool, key=lambda m: -newpool[m][0][0])[:3]
        pool = {m: newpool[m] for m in ms}
        print("p=%d n=%d nu=%d  best by m: %s  (%.0fs)" % (p, n, n + 1, sorted([(m, round(pool[m][0][0], 4)) for m in pool]), time.time() - t0), flush=True)
