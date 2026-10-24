# Referee check v3: CC8 (copositive second-moment lemma, incl. NON-(PSD+N) copositive Horn matrix; covariance
# identity), pair cone moment, CC9 weight formula, CC10 identity, CC11 linear detection (n = 2, 3, 4).
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

rng = np.random.default_rng(77)
HORN = np.array([[1, -1, 1, 1, -1], [-1, 1, -1, 1, 1], [1, -1, 1, -1, 1], [1, 1, -1, 1, -1], [-1, 1, 1, -1, 1]], float)


def rand_cop(k):
    t = rng.integers(3)
    if t == 0:
        A = rng.normal(size=(k, k)); N = np.abs(rng.normal(size=(k, k))) * (rng.random((k, k)) < 0.5); return A @ A.T + (N + N.T) / 2
    if t == 1:
        N = np.abs(rng.normal(size=(k, k))); return (N + N.T) / 2
    if k == 5:
        return HORN.copy()
    A = rng.normal(size=(k, 2)); return A @ A.T


def c(l, d, p):
    return 1 - np.exp(-2j * np.pi * l * d / p)


stats = dict(cov_err=0, cop_min=np.inf, pcm_min=np.inf, Vnorm=0, w_err=0, dense=0, lin_min=np.inf, lin_ratio=0,
             same_feat_viol=0, cases=0)
for (p, n) in [(5, 2), (7, 2), (11, 2), (7, 3), (5, 3), (5, 4), (13, 2)]:
    F = (p - 1) // 2
    for t in range(5):
        widths = {l: int(rng.integers(1, 5)) for l in range(1, F + 1) if rng.random() < 0.85}
        if not widths: widths = {1: 3}
        U, W, lab = V.rand_classC(p, n, widths, rng, scale={l: rng.uniform(0.3, 2) for l in range(1, F + 1)})
        f = V.table(U, W); g = V.gaps(U, W, f); beta = V.clean(f, p, n); sf = V.signflip(f, p, n)
        Ph = V.Phi(f, p, n)
        Vl = np.array([sum(abs(sf[(l, T)]) ** 2 for T in itertools.product([0, 1], repeat=n) if any(T)) for l in range(1, F + 1)])
        Vdir = np.array([np.mean(np.abs(Ph[l - 1] - Ph[l - 1].mean()) ** 2) for l in range(1, F + 1)])
        stats['Vnorm'] = max(stats['Vnorm'], np.abs(Vl - Vdir).max() / max(Vl.max(), 1e-300))
        # covariance identity on all offsets
        Cv = np.cov(g.T, bias=True)
        D = np.arange(1, p)
        Psi = [np.real(np.outer(c(l, D, p), np.conj(c(l, D, p)))) for l in range(1, F + 1)]
        pred = sum(Vl[l - 1] * Psi[l - 1] for l in range(1, F + 1)) / 2
        stats['cov_err'] = max(stats['cov_err'], np.abs(Cv - pred).max() / np.abs(pred).max())
        # copositive lemma on random Delta
        for k in range(10):
            sz = int(rng.integers(2, min(p - 1, 6) + 1))
            Dl = rng.choice(np.arange(1, p), size=sz, replace=False)
            Q = rand_cop(sz)
            gd = g[:, Dl - 1]; hD = gd.min(); gb = gd.mean(0)
            lhs = (gb - hD) @ Q @ (gb - hD)
            rhs = -0.5 * sum(Vl[l - 1] * (Q * Psi[l - 1][np.ix_(Dl - 1, Dl - 1)]).sum() for l in range(1, F + 1))
            stats['cop_min'] = min(stats['cop_min'], (lhs - rhs) / max(abs(lhs) + abs(rhs), 1e-300))
            stats['cases'] += 1
        # pair cone moment at every delta
        Gd = g.min(0)
        for d in range(1, p):
            ls = np.arange(1, F + 1); ph = 2 * np.pi * ls * d / p; kk = 1 - np.cos(ph)
            Rb = (beta.real * kk).sum(); Ib = -(beta.imag * np.sin(ph)).sum(); hd = min(Gd[d - 1], Gd[p - d - 1])
            lhs = (Rb - hd) ** 2; rhs = Ib ** 2 + (kk * np.cos(ph) * Vl).sum()
            stats['pcm_min'] = min(stats['pcm_min'], (lhs - rhs) / max(lhs + abs(rhs), 1e-300))
        # CC11 linear detection with disjoint slot sets (distinct features) and a same-feature control
        feats = sorted(set(lab))
        Dsym = (V.kap(p) @ beta.real) - 0.5 * (Gd + Gd[::-1])
        for k in range(20):
            kfam = int(rng.integers(1, min(n, len(feats)) + 1))
            perm = rng.permutation(n); cuts = np.sort(rng.choice(np.arange(1, n), size=kfam - 1, replace=False)) if kfam > 1 else []
            Ts = np.split(perm, cuts)
            Ts = [T[:max(1, int(rng.integers(1, len(T) + 1)))] for T in Ts]
            ls = rng.choice(feats, size=kfam, replace=False)
            for d in range(1, p):
                det = sum(abs(sf[(int(l), tuple(1 if tt in T else 0 for tt in range(n)))]) * abs(np.sin(np.pi * l * d / p))
                          for l, T in zip(ls, Ts))
                stats['lin_min'] = min(stats['lin_min'], Dsym[d - 1] - det)
                if Dsym[d - 1] > 0: stats['lin_ratio'] = max(stats['lin_ratio'], det / Dsym[d - 1])
            # control: SAME feature on two disjoint slot sets (excluded by the lemma)
            if n >= 2:
                l = int(rng.choice(feats)); T1 = (1,) + (0,) * (n - 1); T2 = (0, 1) + (0,) * (n - 2)
                for d in range(1, p):
                    det = (abs(sf[(l, T1)]) + abs(sf[(l, T2)])) * abs(np.sin(np.pi * l * d / p))
                    if Dsym[d - 1] - det < -1e-12: stats['same_feat_viol'] += 1
    # CC10
    for d1 in range(5):
        Dl = rng.choice(np.arange(1, p), size=min(p - 1, 5), replace=False)
        S_ = sum(np.real(np.outer(c(l, Dl, p), np.conj(c(l, Dl, p)))) for l in range(1, F + 1))
        stats['dense'] = max(stats['dense'], np.abs(S_ - (p / 2) * (np.eye(len(Dl)) + 1)).max())
# CC9 weight formula vs direct
for t in range(2000):
    p = int(rng.choice([5, 7, 11, 13])); l = int(rng.integers(1, (p - 1) // 2 + 1)); d = int(rng.integers(1, p))
    Q = rand_cop(2); Dl = np.array([d, p - d])
    Ps = np.real(np.outer(c(l, Dl, p), np.conj(c(l, Dl, p))))
    w_direct = -0.5 * (Q * Ps).sum(); kk = 1 - np.cos(2 * np.pi * l * d / p)
    w_form = -kk * ((Q[0, 0] + Q[1, 1]) + 2 * Q[0, 1] * (kk - 1))
    stats['w_err'] = max(stats['w_err'], abs(w_direct - w_form))
for k_, v in stats.items():
    print(f'  {k_:15s} {v}')
print('reading: cov_err/Vnorm/w_err/dense ~ 0 (identities); cop_min, pcm_min >= 0 (inequalities hold);'
      ' lin_min >= 0 (CC11 holds); same_feat_viol counts failures of the excluded same-feature case')
