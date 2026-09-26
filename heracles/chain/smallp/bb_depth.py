"""Branch-and-bound certificate that the flip-enlarged aim body B (plain + slot-1 flip + slot-2 flip) contains the
l2-polyball of radius A_target on frequency set T at prime p.

Statement certified: for every direction z = (r_k zhat_k)_k (r_k >= 0, zhat_k unit in C^3), some point g of B has
Re<zhat_k, g_k> >= A_target for every k; hence H(z) >= Re<z, g> >= A_target sum_k r_k (H = support function of B).
Cells: per frequency a box in (alpha, beta, phi_1, phi_2, phi_3), moduli r = (cos a, sin a cos b, sin a sin b).
For a cell, g = sum_i mu_i gen_i with mu from a small LP at the cell centre (exact convex combination of generators),
and the per-frequency lower bound over the whole box is computed by interval reasoning on r_T |g_T| cos(arg g_T - phi_T).
Symmetry: row shifts (a,b) -> (a+x, b+y) rotate frequency k's pattern-1 phase by 2k x (2 pi/p) and pattern-2 phase by
2k y; since 2k0 is invertible we may assume frequency T[0]'s pattern-1 and pattern-2 phases (of zhat, as seen through the
pairing) lie within pi/p of 0.  (Only this symmetry is used.)
Floating point: all quantities O(1); the certification threshold includes a 1e-9 safety margin, far above rounding.
"""
import sys, time, heapq, numpy as np
from scipy.optimize import linprog

def gens(p, T):
    x = np.arange(p); a, b = np.meshgrid(x, x, indexing='ij'); a = a.ravel(); b = b.ravel(); T = np.array(T)
    E = lambda u: np.exp(-2j * np.pi * 2 * np.outer(u, T) / p)
    V1, V2, V3 = E(a), E(b), E((a + b) % p)
    G = [np.stack([s1 * V1, s2 * V2, s1 * s2 * V3], axis=2) for s1, s2 in ((1, 1), (-1, 1), (1, -1))]
    return np.concatenate(G, axis=0)                       # n x K x 3

def unit(box):
    al, be = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
    ph = [(box[4 + 2 * j] + box[5 + 2 * j]) / 2 for j in range(3)]
    r = np.array([np.cos(al), np.sin(al) * np.cos(be), np.sin(al) * np.sin(be)])
    return r * np.exp(1j * np.array(ph))

def r_ranges(box):
    a0, a1, b0, b1 = box[:4]
    return [(np.cos(a1), np.cos(a0)), (np.sin(a0) * np.cos(b1), np.sin(a1) * np.cos(b0)), (np.sin(a0) * np.sin(b0), np.sin(a1) * np.sin(b1))]

def cos_range(center, lo, hi):
    # range of cos(center - phi) for phi in [lo, hi]
    d0, d1 = center - hi, center - lo            # angle interval [d0, d1]
    c0, c1 = np.cos(d0), np.cos(d1)
    cmin, cmax = min(c0, c1), max(c0, c1)
    k0 = np.ceil(d0 / np.pi); k1 = np.floor(d1 / np.pi)
    for m in range(int(k0), int(k1) + 1):
        if m % 2 == 0: cmax = 1.0
        else: cmin = -1.0
    return cmin, cmax

def lower_bound(box, g):
    # min over zhat in box of Re<zhat, g> = sum_T r_T |g_T| cos(arg g_T - phi_T)
    lb = 0.0
    for T, (rlo, rhi) in enumerate(r_ranges(box)):
        mag = abs(g[T])
        if mag == 0: continue
        cmin, _ = cos_range(np.angle(g[T]), box[4 + 2 * T], box[5 + 2 * T])
        lb += (rlo if cmin >= 0 else rhi) * mag * cmin
    return lb

def solve(Gn, U):
    # max s s.t. sum_i mu_i Re<U_k, gen_ik> >= s for all k, mu in simplex
    C = np.real(np.einsum('kt,ikt->ik', np.conj(U), Gn))        # n x K
    n, K = C.shape
    r = linprog(np.r_[np.zeros(n), -1], A_ub=np.c_[-C.T, np.ones(K)], b_ub=np.zeros(K),
                A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method='highs')
    mu = np.clip(r.x[:n], 0, None); mu /= mu.sum()
    return np.einsum('i,ikt->kt', mu, Gn), -r.fun

def split(box):
    w = [(box[1] - box[0]), (box[3] - box[2]) * np.sin(box[1])] + [box[5 + 2 * j] - box[4 + 2 * j] for j in range(3)]
    d = int(np.argmax(w)); lo, hi = box[2 * d], box[2 * d + 1]; mid = (lo + hi) / 2
    b1 = list(box); b2 = list(box); b1[2 * d + 1] = mid; b2[2 * d] = mid
    return tuple(b1), tuple(b2)

def run(p, T, target, max_cells=10 ** 7, nph=3):
    Gn = gens(p, T); K = len(T); target_eff = target + 1e-9
    full = lambda: [(0, np.pi / 2, 0, np.pi / 2) + sum(((2 * np.pi * j / nph, 2 * np.pi * (j + 1) / nph),), ()) for j in range(nph)]
    # initial per-frequency boxes: alpha, beta halves; phases in nph arcs each
    def init_boxes(first):
        out = []
        for a in ((0, np.pi / 4), (np.pi / 4, np.pi / 2)):
            for b in ((0, np.pi / 4), (np.pi / 4, np.pi / 2)):
                if first:
                    for j3 in range(nph):
                        out.append(a + b + (-np.pi / p, np.pi / p, -np.pi / p, np.pi / p, 2 * np.pi * j3 / nph, 2 * np.pi * (j3 + 1) / nph))
                else:
                    for j1 in range(nph):
                        for j2 in range(nph):
                            for j3 in range(nph):
                                out.append(a + b + tuple(x for j in (j1, j2, j3) for x in (2 * np.pi * j / nph, 2 * np.pi * (j + 1) / nph)))
        return out
    import itertools
    stack = [(list(c), None) for c in itertools.product(init_boxes(True), *[init_boxes(False) for _ in range(K - 1)])]
    lp_calls = 0
    processed = 0; leaves = 0; worst_ok = 9; t0 = time.time()
    while stack:
        cell, g = stack.pop(); processed += 1
        if g is not None:                                  # try the parent's aim first (still a point of B)
            lbs = [lower_bound(b, g[k]) for k, b in enumerate(cell)]
            if min(lbs) >= target_eff:
                leaves += 1; worst_ok = min(worst_ok, min(lbs)); continue
        U = np.array([unit(b) for b in cell])
        g, s = solve(Gn, U); lp_calls += 1
        if s < target_eff:
            sizes = [max(b[1] - b[0], b[3] - b[2], *(b[5 + 2 * j] - b[4 + 2 * j] for j in range(3))) for b in cell]
            if max(sizes) < 1e-3:
                return dict(status='FAIL', centre=U.tolist(), value=s, processed=processed)
        lbs = [lower_bound(b, g[k]) for k, b in enumerate(cell)]
        if min(lbs) >= target_eff:
            leaves += 1; worst_ok = min(worst_ok, min(lbs)); continue
        k = int(np.argmin(lbs)); c1, c2 = split(cell[k])
        n1 = list(cell); n1[k] = c1; n2 = list(cell); n2[k] = c2
        stack.append((n1, g)); stack.append((n2, g))
        if processed % 20000 == 0:
            print(f"  processed {processed}, LPs {lp_calls}, stack {len(stack)}, leaves {leaves}, {time.time() - t0:.0f}s", flush=True)
        if processed > max_cells:
            return dict(status='CAP', processed=processed, stack=len(stack), leaves=leaves)
    return dict(status='CERTIFIED', processed=processed, lp_calls=lp_calls, leaves=leaves, min_leaf_bound=worst_ok, seconds=time.time() - t0)

if __name__ == '__main__':
    p = int(sys.argv[1]); T = [int(t) for t in sys.argv[2].split(',')]; A = float(sys.argv[3])
    print(run(p, T, A, max_cells=int(float(sys.argv[4])) if len(sys.argv) > 4 else 10 ** 7))
