"""Rough numerical sanity checks for the structural consequences in heracles/CONSEQUENCES.md.

Share programme (s > 1, r = 1/s):  Gamma_s(S) = max over y >= 0 supported on S, sum y^r <= 1,
of min over offsets D = 1..h of sum_k y_k (1 - cos(2 pi k D / p)).
Equivalent margin-one form: minimise sum v^r over the polyhedron {v >= 0, A v >= 1}; the minimum of a
concave cost sits at a vertex, so Gamma_s = (min over vertices of sum v^r)^(-s).  The vertex list does
not depend on s, so we enumerate it once per p and reuse it for every s and every support S.

Checks (claim numbers follow CONSEQUENCES.md):
  C1 plateaus      optimal vertex is piecewise constant in s; list the switches
  C4 dense window  largest s at which the optimum still uses every frequency
  C3 ablation      m_{-j} <= m (1 - w_j)^s at the exact optimum
  C7 bottlenecks   number of tight offsets >= K
  C2 degeneracy    dilates of the optimal support are optimal; count distinct optimal supports
  C5 certificates  supports S with Gamma_s(S) < Gamma_s(p) / sqrt(3) cannot carry a global optimum (nu=3)
  C6 progressions  Gamma_s({a,...,Ka}) <= 1 - cos(2 pi K / p) <= 2 pi^2 K^2 / p^2
  C8 gauge         phase shifts (alpha, beta, alpha+beta) leave the block's logit table unchanged
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/heracles/', 1)[0] + '/heracles/chain')
from common import block_net, cover_matrix, programme, table  # noqa: E402


def fold(x, p):
    x %= p
    return min(x, p - x)


def vertices(p):
    """All vertices v of {v >= 0, A v >= 1} (A = h x h coverage matrix), deduplicated."""
    A = cover_matrix(p)
    h = A.shape[0]
    out = {}
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            Ts = list(itertools.combinations(range(h), K))
            M = np.stack([A[np.ix_(T, S)] for T in Ts])
            ok = np.abs(np.linalg.det(M)) > 1e-10
            if not ok.any():
                continue
            vs = np.linalg.solve(M[ok], np.ones((ok.sum(), K, 1)))[..., 0]
            for vS in vs:
                if (vS < -1e-10).any():
                    continue
                v = np.zeros(h)
                v[list(S)] = np.clip(vS, 0, None)
                if (A @ v < 1 - 1e-9).any():
                    continue
                key = tuple(np.round(v, 9))
                out[key] = v
    V = np.array(list(out.values()))
    return A, V


def mask_of(v):
    return sum(1 << j for j in np.nonzero(v > 1e-12)[0])


def main(ps, s0=1.5):
    print(f'{"":=^100}')
    for p in ps:
        A, V = vertices(p)
        h = A.shape[0]
        r0 = 1 / s0
        cost0 = (V ** r0).sum(1)
        i0 = cost0.argmin()
        G = cost0[i0] ** (-s0)
        Gref, _ = programme(p, s0) if h <= 9 else (np.nan, None)
        print(f'p={p} h={h}: {len(V)} vertices; Gamma_1.5 = {G:.6f} (programme() {Gref:.6f})')

        # C1 plateaus + C4 dense window -- vertices compared modulo dilation k -> fold(a k)
        def canon(v):
            best = None
            for a in range(1, p):
                img = np.zeros(h)
                for j in np.nonzero(v > 1e-12)[0]:
                    img[fold(a * (j + 1), p) - 1] = v[j]
                key = tuple(np.round(img, 7))
                best = key if best is None or key < best else best
            return best
        keys = [canon(v) for v in V]
        classes = {}
        for i, kk in enumerate(keys):
            classes.setdefault(kk, i)
        sgrid = np.concatenate([1 + np.logspace(-4, -1, 400), np.linspace(1.1, 4, 600)])
        win = []
        for s in sgrid:
            c = (V ** (1 / s)).sum(1)
            cm = c.min()
            win.append(frozenset(keys[i] for i in np.nonzero(c <= cm * (1 + 1e-9))[0]))
        plateaus, start = [], 0
        for t in range(1, len(sgrid) + 1):
            if t == len(sgrid) or win[t] != win[start]:
                desc = []
                for kk in sorted(win[start]):
                    v = np.array(kk)
                    sup = [int(j) + 1 for j in np.nonzero(v > 1e-12)[0]]
                    desc.append(f'K={len(sup)} {sup} ratios {np.round(v[v > 1e-12] / v.max(), 3).tolist()}')
                plateaus.append((sgrid[start], sgrid[t - 1], desc))
                start = t
        print(f'  C1 plateaus over s in [1.0001, 4] (modulo dilation): {len(plateaus)} (switches {len(plateaus) - 1})')
        for a, b, desc in plateaus:
            print(f'     s {a:.4f}-{b:.4f}: ' + ' | '.join(desc))
        full = [sgrid[t] for t in range(len(sgrid)) if all(sum(1 for x in kk if x > 1e-12) == h for kk in win[t])]
        sc = max(full) if full else None
        print(f'  C4 dense window: optimum uses all {h} frequencies up to s = {sc}'
              + (f'  (s_c - 1 = {sc - 1:.4g}; (s_c-1)*log p = {(sc - 1) * np.log(p):.3f})' if sc else ''))

        # C3 ablation, C7 bottlenecks at s0
        v = V[i0]
        y = v * cost0[i0] ** (-s0)
        m = (A @ y).min()
        act = np.nonzero(y > 1e-12)[0]
        w = y ** r0
        tight = int((np.abs(A @ y - m) < 1e-9 * max(1, m)).sum())
        print(f'  C7 bottlenecks: K={len(act)}, tight offsets {tight} (claim: >= K)  budget check sum w = {w.sum():.6f}')
        rows = []
        for j in act:
            yj = y.copy()
            yj[j] = 0
            mj = (A @ yj).min()
            loss = 1 - mj / m
            bound = 1 - (1 - w[j]) ** s0
            rows.append((j + 1, w[j], loss, bound, 2 * y[j] / m))
        worst = min(rw[2] / rw[3] for rw in rows)
        print(f'  C3 ablation (loss >= 1-(1-w)^s; also loss <= 2y/m): min loss/bound = {worst:.3f}')
        for k, wj, loss, bound, ub in rows:
            print(f'     k={k:2d} share w={wj:.3f}  loss {loss:.3f}  lower bound {bound:.3f}  upper 2y/m {ub:.3f}')

        # C2 degeneracy
        opt = [i for i in range(len(V)) if abs(cost0[i] - cost0[i0]) < 1e-9 * cost0[i0]]
        supports = {mask_of(V[i]) for i in opt}
        S0 = [j + 1 for j in act]
        orbit = set()
        for a in range(1, p):
            orbit.add(frozenset(fold(a * k, p) for k in S0))
        orbit_opt = all(any(mask_of(V[i]) == sum(1 << (k - 1) for k in Sd) for i in opt) for Sd in orbit)
        print(f'  C2 degeneracy: {len(supports)} distinct optimal supports; dilation orbit of the optimum has '
              f'{len(orbit)} supports (bound (p-1)/(2K) = {(p - 1) / (2 * len(S0)):.2f}); all dilates optimal: {orbit_opt}')

        # C5 certificates (nu = 3): Gamma_s(S) via subset-min DP over vertex supports
        INF = np.inf
        best = np.full(1 << h, INF)
        for i in range(len(V)):
            mk = mask_of(V[i])
            best[mk] = min(best[mk], cost0[i])
        for j in range(h):
            for mk in range(1 << h):
                if mk >> j & 1:
                    best[mk] = min(best[mk], best[mk ^ (1 << j)])
        GS = np.where(np.isfinite(best), best ** (-s0), 0.0)
        thr = G / np.sqrt(3)
        line = []
        for K in range(1, h + 1):
            ms = [mk for mk in range(1, 1 << h) if bin(mk).count('1') == K]
            adm = sum(GS[mk] >= thr for mk in ms)
            line.append(f'K={K}:{adm}/{len(ms)}')
        print(f'  C5 certificate Gamma(S) >= Gamma/sqrt3 survivors by size: ' + '  '.join(line))
        # a minimal-support check: supports that are NOT supersets of an admissible support are rejected
        # C6 progressions
        viol = 0
        worst6 = 0
        for a in range(1, h + 1):
            for K in range(1, h + 1):
                S = {fold(j * a, p) for j in range(1, K + 1)}
                mk = sum(1 << (k - 1) for k in S)
                bnd = 1 - np.cos(2 * np.pi * K / p)
                if GS[mk] > bnd + 1e-12 or GS[mk] > 2 * np.pi ** 2 * K ** 2 / p ** 2 + 1e-12:
                    viol += 1
                if K <= max(2, h // 3):
                    worst6 = max(worst6, GS[mk] / G)
        print(f'  C6 progressions: violations {viol}; best progression (K <= max(2,h/3)) reaches {worst6:.3f} of Gamma')

        # C8 gauge
        net = block_net(p, y)
        z0 = table(p, net)
        rng = np.random.default_rng(p)
        alpha, beta = rng.uniform(0, 2 * np.pi, 2)
        net2, idx = [], 0
        for k0, yk in enumerate(y):
            if yk <= 0:
                continue
            k = k0 + 1
            rho = (4 * yk / 3) ** (1 / 3)
            x = np.arange(p)
            for l in range(3):
                phi = np.pi * l / 3
                c = lambda ph: rho * np.cos(2 * np.pi * k * x / p + ph)
                net2.append((c(phi + alpha), c(phi + beta), c(2 * phi + alpha + beta)))
        z1 = table(p, net2)
        cost_a = sum(float(a @ a + b @ b + c @ c) for a, b, c in net)
        cost_b = sum(float(a @ a + b @ b + c @ c) for a, b, c in net2)
        print(f'  C8 gauge: max |table change| = {np.abs(z1 - z0).max():.2e} (table scale {np.abs(z0).max():.2f}); '
              f'cost change {abs(cost_a - cost_b):.2e}')
        print(f'{"":-^100}')


if __name__ == '__main__':
    ps = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [7, 11, 13, 17, 19]
    main(ps)
