import math, sys, time, numpy as np, torch
sys.path.insert(0, '.')
from elim import lp, true_eff
from oned3 import lamstar
from cont1d import polish
torch.set_num_threads(1)

def path(p, n, b, tg, rng, mstop, drop_rand=0.0):
    nu = n + 1; r = b / nu
    lam = np.array([lamstar(t, p) for t in tg]); J = len(lam)
    active = np.ones(J, bool); w = np.ones(J); out = []
    while active.sum() >= mstop:
        idx = np.where(active)[0]
        res = lp(lam[idx], p, n, w[idx])
        if res is None or res[0] <= 0: break
        c, Vt, lz = res; a = np.abs(Vt); keep = a > 1e-9 * a.max()
        e, x = true_eff(lam[idx], Vt, lz, p, n, b, keep)
        out.append((int(keep.sum()), e, x))
        newactive = np.zeros(J, bool); newactive[idx[keep]] = True
        if newactive.sum() == active.sum():
            kk = idx[keep]
            if drop_rand and rng.random() < drop_rand:
                j = kk[rng.integers(len(kk))]
            else:
                j = kk[np.argmin(a[keep])]
            newactive[j] = False
        active = newactive
        w = np.ones(J); w[idx[keep]] = (a[keep] / a[keep].max() + 1e-3) ** (r - 1)
    return out

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); b = float(sys.argv[3]); R = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(seed); t0 = time.time()
    th = math.pi / p; L = (p - 1) * n
    mD = int(math.ceil(1 + (th * L - 2 * th) / math.pi - 1e-9))
    best = {}
    for k in range(R):
        J = int(rng.integers(60, 260))
        tg = np.sort(rng.uniform(0.002, p - 1 - 0.002, J)) if k % 2 else np.linspace(0.002, p - 1 - 0.002, J)
        for m, e, x in path(p, n, b, tg, rng, max(mD - 1, 2), drop_rand=0.3 if k % 3 == 2 else 0.0):
            if e > 0 and (m not in best or e > best[m][0]): best[m] = (e, x)
    print("p=%d n=%d nu=%d Descartes m>=%d  raw best by m: %s (%.0fs)" % (p, n, n + 1, mD, sorted([(m, round(v[0], 4)) for m, v in best.items()]), time.time() - t0), flush=True)
    fin = {}
    for m in sorted(best):
        if m > n + 3: continue
        rr, xx = polish(best[m][1], m, p, n, b)
        fin[m] = max(rr, best[m][0])
        np.save("e2_p%d_n%d_m%d.npy" % (p, n, m), xx if rr >= best[m][0] else best[m][1])
    print("p=%d n=%d nu=%d  POLISHED by m: %s  max %.5f (%.0fs)" % (p, n, n + 1, sorted([(m, round(v, 4)) for m, v in fin.items()]), max(fin.values()) if fin else -1, time.time() - t0), flush=True)
