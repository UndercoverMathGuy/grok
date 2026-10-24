# Sanity check of the reduction  (J'_theta) + (P_theta)  =>  class-C RD_S (IF value), on explicit class-C networks.
# For a class-C network (CC model, units B_p = 1):  Q_l = sum_{i in l} |b_i|^r,  y_l = (Q_l / sum Q)^s,
#   delta* = argmin_delta sum_l y_l kappa_l(delta),  L1 = sum_{l,T} |a_{l,T}| |c_l(delta*)|,
#   theta_net = ( sum_l |A_l| kappa_l(delta*) - min_{x, sign} g(x, sign delta*) ) / L1     (network's own (J') constant),
#   theta_pay = max over features with G_l > 1 of (1 - 1/G_l) / eps_l,  eps_l = sum_T |a_{l,T}| / |A_l|.
# Claim (proved in the report): if theta_net >= theta_pay then
#   h <= sum_l |A_l| kappa_l (1 - theta_net eps_l)_+ <= nu^{1-s} sum_l Q_l^s kappa_l(delta*) <= nu^{1-s} (sum Q)^s Gamma^{(S)},
# i.e. ratio <= 1.  Every inequality of the chain is checked separately.
import sys, os, glob, itertools
import numpy as np
import hlib as H

rng = np.random.default_rng(7)
viol = 0; tested = 0; applicable = 0; worst = {}


def analyse(cc, x, S, G):
    global viol, tested, applicable
    p, n, s, nu, r = cc.p, cc.n, cc.s, cc.nu, cc.r
    b, phi = cc.unpack(x); xi = np.exp(1j * phi)
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
    feats = cc.feats
    A = {}; Q = {}; aT = {}
    for k, l in enumerate(feats):
        sel = cc.fidx == k
        A[l] = b[sel].sum(); Q[l] = np.sum(np.abs(b[sel]) ** r)
        aT[l] = np.array([np.sum(b[sel] * np.prod(xi[sel][:, np.array(T, bool)], axis=1)) for T in Ts])
    Qs = sum(Q.values()); y = {l: (Q[l] / Qs) ** s for l in feats}
    kap = lambda l, d: 1 - np.cos(2 * np.pi * l * d / p)
    F = (p - 1) // 2
    dstar = min(range(1, F + 1), key=lambda d: sum(y[l] * kap(l, d) for l in feats))
    g = cc.gaps(x)
    hmin = min(g[:, dstar - 1].min(), g[:, p - dstar - 1].min())
    h = g.min()
    L1 = sum(np.sum(np.abs(aT[l])) * np.sqrt(2 * kap(l, dstar)) for l in feats)
    clean = sum(abs(A[l]) * kap(l, dstar) for l in feats)
    theta_net = (clean - hmin) / L1 if L1 > 1e-14 else np.inf
    Gl = {l: abs(A[l]) / (nu ** (1 - s) * Q[l] ** s) for l in feats}
    eps = {l: np.sum(np.abs(aT[l])) / abs(A[l]) if abs(A[l]) > 1e-14 else np.inf for l in feats}
    theta_pay = max([(1 - 1 / Gl[l]) / eps[l] for l in feats if Gl[l] > 1 and eps[l] > 0] + [0.0])
    ratio = h / Qs ** s / (nu ** (1 - s) * G)
    tested += 1
    ok = True
    if theta_net >= theta_pay:
        applicable += 1
        th = theta_net
        s1 = hmin <= clean - th * L1 + 1e-12
        s2 = clean - th * L1 <= sum(abs(A[l]) * kap(l, dstar) * max(0.0, 1 - th * eps[l]) for l in feats) + 1e-12
        s3 = all(abs(A[l]) * max(0.0, 1 - th * eps[l]) <= nu ** (1 - s) * Q[l] ** s * (1 + 1e-12) for l in feats)
        s4 = sum(Q[l] ** s * kap(l, dstar) for l in feats) <= Qs ** s * G * (1 + 1e-9)
        ok = s1 and s2 and s3 and s4 and ratio <= 1 + 1e-9
        if not ok:
            viol += 1; print("FAIL", p, n, s, s1, s2, s3, s4, ratio, theta_net, theta_pay, flush=True)
    key = (p, n, round(s, 3))
    worst.setdefault(key, []).append((ratio, theta_net, theta_pay))
    return ok


# 1) random class-C networks (generic, IF-perturbed, cheap profiles), several (p, n, s)
for (p, n, bexp) in [(5, 2, 2.0), (7, 2, 2.0), (11, 2, 2.0), (13, 2, 2.5), (7, 3, 2.0), (5, 3, 3.0), (5, 4, 2.0)]:
    s = (n + 1) / bexp; F = (p - 1) // 2
    G, y, _ = H.programme(p, s)
    for it in range(150):
        k = int(rng.integers(1, F + 1)); S = sorted(rng.choice(range(1, F + 1), k, replace=False).tolist())
        freqs = [l for l in S for _ in range(int(rng.integers(1, n + 3)))]
        cc = H.CC(p, n, freqs, s)
        mode = it % 3
        if mode == 0:
            bb = rng.standard_normal(cc.m) + 1j * rng.standard_normal(cc.m); ph = rng.uniform(0, 2 * np.pi, (cc.m, n))
        else:
            bb = np.array([y[l - 1] + 0.05 for l in freqs]) * np.exp(0.3 * rng.standard_normal(cc.m)) * np.exp(1j * 0.1 * rng.standard_normal(cc.m))
            cnt = {}; ph = np.zeros((cc.m, n))
            for i, l in enumerate(freqs):
                c_ = cnt.get(l, 0); cnt[l] = c_ + 1; ph[i] = 2 * np.pi * c_ / freqs.count(l)
            ph += (0.05 if mode == 1 else 0.5) * rng.standard_normal((cc.m, n))
        x = cc.pack(bb, ph)
        if it % 10 == 0: x = cc.lbfgs(x, taus=(1e-2, 1e-3), iters=300)
        analyse(cc, x, list(range(1, F + 1)), G)
# 2) the hunt's saved best networks (class C)
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'best_*.npz'))):
    z = np.load(f, allow_pickle=True)
    if 'freqs' not in z.files or z['freqs'] is None: continue
    p = int(z['p']); n = int(z['n']); bexp = float(z['b']); s = (n + 1) / bexp
    S = [int(v) for v in z['S']] if 'S' in z.files else list(range(1, (p - 1) // 2 + 1))
    G, _, _ = H.programme(p, s, S=S)
    cc = H.CC(p, n, [int(v) for v in z['freqs']], s)
    analyse(cc, z['x'], S, G)
print(f"networks tested={tested}  reduction applicable (theta_net >= theta_pay)={applicable}  failures={viol}")
for key, v in worst.items():
    v = np.array([(a, b_ if np.isfinite(b_) else 9.9, c) for a, b_, c in v])
    app = v[v[:, 1] >= v[:, 2]]
    print(f"  {key}: max ratio overall={v[:, 0].max():.6f}; applicable={len(app)}; max ratio among applicable="
          f"{app[:, 0].max() if len(app) else float('nan'):.6f}; min theta_net among cheap nets="
          f"{v[v[:, 2] > 0][:, 1].min() if (v[:, 2] > 0).any() else float('nan'):.4f}")
