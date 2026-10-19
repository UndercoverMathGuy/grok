# Network-level sanity check of the L2 relaxation used in keff2.py (class C, n = 2):
#   (i)   h <= min_delta [ sum_l Re(A_l) kappa_l(delta) - rho_p (sum_{l,T} |a_{l,T}|^2 kappa_l(delta))^{1/2} ],  rho_p = sqrt((p-1)/(p+1))
#         (pair detection: GIBBS R3 two-plane Bhatia-Davis; constant 1 for consistent families, Lemma L2 here)
#   (ii)  per feature: G_l <= 1 / (1 - theta(s) eta_l),  theta(s) = (1 - 3^{1-s})/sqrt3   (sharp L2 payment, GIBBS R6)
#   (iii) gamma / D_S <= R(network) := min_delta [ sum G_l y_l kappa_l - rho_p (sum eta_l^2 G_l^2 y_l^2 kappa_l)^{1/2} ] / Gamma^{(S)}
#         with y_l = (Q_l / Q)^s;  and R(network) <= UB(p, s, S) (the keff2 maximum) whenever the optimiser is right.
# Networks: random class-C networks, IF perturbations, cheap profiles, and every saved hunt network (out2/best_a2_*.npz).
import glob, json, itertools
import numpy as np
import hlib as H, hlib2 as H2

rng = np.random.default_rng(17)


def analyse(cc, x, S=None):
    p, n, s = cc.p, cc.n, cc.s; nu = n + 1; F = (p - 1) // 2
    b, phi = cc.unpack(x)
    xi = np.exp(1j * phi)
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
    d = np.arange(1, p)
    h = cc.gaps(x).min(); Q = cc.cost(x)
    clean = np.zeros(p - 1); E = np.zeros(p - 1); feats = {}
    for k, l in enumerate(cc.feats):
        sel = cc.fidx == k
        A = b[sel].sum(); Ql = np.sum(np.abs(b[sel]) ** (1 / s))
        aT = np.array([np.sum(b[sel] * np.prod(xi[sel] ** np.array(T)[None, :], axis=1)) for T in Ts])
        kap = 1 - np.cos(2 * np.pi * l * d / p)
        clean += A.real * kap; E += np.sum(np.abs(aT) ** 2) * kap
        if abs(A) > 1e-12 * np.abs(b).sum():
            G = abs(A) / (nu ** (1 - s) * Ql ** s); eta = np.sqrt(np.sum(np.abs(aT) ** 2)) / abs(A)
            feats[l] = (G, eta, Ql)
    rho = np.sqrt((p - 1) / (p + 1))
    bound_i = np.min(clean - rho * np.sqrt(E))
    theta = (1 - 3 ** (1 - s)) / np.sqrt(3)
    pay = max([G * (1 - theta * eta) for (G, eta, Ql) in feats.values()] or [0])
    T = sorted(feats)
    R = H2.prog_bb(p, s, S=T, maxnodes=20000) if len(T) <= 10 else H2.prog_best(p, s, S=T)
    Ksub = H2.kmat(p)
    y = np.zeros(F); Gv = np.zeros(F); ev = np.zeros(F)
    for l, (G, eta, Ql) in feats.items(): y[l - 1] = (Ql / Q) ** s; Gv[l - 1] = G; ev[l - 1] = eta
    Rnet = np.min(Ksub @ (Gv * y) - rho * np.sqrt(Ksub @ ((ev * Gv * y) ** 2))) / R['G']
    ratio = h / Q ** s / (nu ** (1 - s) * R['G'])
    return dict(h_minus_bound=h - bound_i, max_payment=pay, ratio=ratio, R=Rnet, ok=bool(h <= bound_i + 1e-12 * abs(h) + 1e-15 and pay <= 1 + 1e-9 and ratio <= Rnet + 1e-9))


nets = []
for p in [7, 11, 13, 17]:
    F = (p - 1) // 2
    for s in [1.2, 1.5]:
        for trial in range(25):
            K = int(rng.integers(1, F + 1)); feats = sorted(rng.choice(np.arange(1, F + 1), size=K, replace=False).tolist())
            freqs = []
            for l in feats: freqs += [l] * int(rng.integers(1, 5))
            cc = H.CC(p, 2, freqs, s)
            kind = trial % 3
            b = np.abs(rng.standard_normal(cc.m)) * (np.exp(1j * rng.normal(0, 0.2, cc.m)) if kind else np.exp(1j * rng.uniform(0, 6.3, cc.m)))
            ph = rng.uniform(0, 2 * np.pi, (cc.m, 2)) if kind != 2 else (2 * np.pi * np.arange(cc.m)[:, None] / 3 + rng.normal(0, 0.3, (cc.m, 2)))
            x = cc.pack(b, ph)
            if trial % 5 == 4: x = cc.lbfgs(x, taus=(1e-2, 3e-3), iters=200)       # partially optimised
            nets.append((f'rand p={p} s={s}', cc, x))
for fn in sorted(glob.glob('out2/best_a2_*.npz')):
    d = np.load(fn, allow_pickle=True)
    cc = H.CC(int(d['p']), 2, list(d['freqs']), 3 / float(d['b']))
    nets.append((fn.split('/')[-1], cc, d['x']))
fails = 0; worst_i = -np.inf; worst_pay = 0; worst_gap = -np.inf
for name, cc, x in nets:
    r = analyse(cc, x)
    worst_i = max(worst_i, r['h_minus_bound']); worst_pay = max(worst_pay, r['max_payment'])
    worst_gap = max(worst_gap, r['ratio'] - r['R'])
    if not r['ok']: fails += 1; print('FAIL', name, r, flush=True)
    if name.startswith('best'): print(name, json.dumps(r), flush=True)
print(f"networks={len(nets)} failures={fails} max(h - detection bound)={worst_i:.3e} (<= 0) max G(1 - theta eta)={worst_pay:.6f} (<= 1) "
      f"max(ratio - R(network))={worst_gap:.3e} (<= 0)", flush=True)
