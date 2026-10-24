# Re-check the subgroup classes where the relaxation maximum came out > 1: is it eta = 0 (Gamma^{(S)} underestimated) or a
# genuine profitable cheapening?  Iterate: Gamma <- best coverage found at eta = 0 (an allocation in the r-ball), then re-optimise.
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore')
exec(open('subgroup_scan.py').read().split("t0 = time.time(); ncand")[0].replace("s = float(sys.argv[1]); pmin = int(sys.argv[2]); pmax = int(sys.argv[3]); thr = float(sys.argv[4])", "s = 1.5; pmin = pmax = 0; thr = 0").replace("dmin = int(sys.argv[5]) if len(sys.argv) > 5 else 10; dmax = int(sys.argv[6]) if len(sys.argv) > 6 else 80", "dmin, dmax = 0, 0"))
from scipy.optimize import minimize


def cov_opt(p, S, G0, y0):
    # maximise min_delta cov_y / (sum y^r)^s (eta = 0) by smooth-min L-BFGS from y0 and from many random starts
    Ks = H2.kmat(p)[:, [l - 1 for l in S]]; k = len(S); best = (G0, y0)
    for t in range(40):
        z = np.log(np.maximum(y0, 1e-9 * y0.max())) if t == 0 else rng.normal(0, 1.5, k)
        for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            def f(z):
                y = np.exp(z); N = np.sum(y ** r) ** s; c = Ks @ y / N
                cm = c.min(); w = np.exp(-(c - cm) / (tau * G0)); lam = w / w.sum()
                val = cm - tau * G0 * np.log(w.sum())
                g = y * (lam @ Ks) / N - (lam @ c) * (y ** r / np.sum(y ** r))
                return -val, -g
            z = minimize(f, z, jac=True, method='L-BFGS-B').x
        y = np.exp(z); N = np.sum(y ** r) ** s; v = (Ks @ y / N).min()
        if v > best[0]: best = (v, y / N)
    return best


for (p, d) in [(109, 54), (181, 30), (53, 52), (61, 60)]:
    g = prim_root(p); h = pow(g, (p - 1) // d, p)
    S = sorted({red(pow(h, k, p), p) for k in range(d)})
    R = H2.prog_bb(p, s, S=S, maxnodes=200, restarts=60)
    G = R['G']; ys = R['y'][[l - 1 for l in S]]
    G2, y2 = cov_opt(p, S, G, ys)
    UB = ub_opt(p, S, max(G, G2), y2 if G2 > G else ys)
    print(json.dumps(dict(p=p, d=d, S_size=len(S), Gamma_rwLP=G, Gamma_improved=G2, Kstar_improved=int((y2 > 1e-6 * y2.max()).sum()), UB_vs_improved=UB)), flush=True)
