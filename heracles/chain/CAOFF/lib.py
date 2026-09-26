"""CAoff tooling (n = 2 operands, nu = 3, q = 2; conventions of chain/common.py).

Per-offset worst margins (all are >= the network margin, so any of them gives a valid CAoff):
  m^u_D = min(min_x gap_D(x), min_x gap_{-D}(x))                        unpaired (smallest, weakest CAoff)
  m^s_D = min_x (gap_D(x) + gap_{-D}(x))/2                              same-row pairing (X_CHARGE's CAoff)
  m^f_D = min_x (gap_D(x1 + D/2, x2) + gap_{-D}(x1 - D/2, x2))/2        slot-1 flip pairing
Budgets (l^{2/3} units, so a junk-free block spends exactly sum |G_k|^{2/3}):
  diag = sum_k cost_k / 3^{1/3},  cost_k = sum_i |e1^(k) e2^(k) w^(k)|^{2/3}      (diagonal atoms)
  P    = (p/6) P_2 / 3^{1/3}                                                          (full weight norm; diag <= P)
need(M) = min { sum_k y_k^{2/3} : y >= 0, A y >= M }, A = cover matrix, M = (p^3/2) m.
CAoff ratio = (need / budget)^{3/2}. CAoff says ratio <= 1. ratio >= T = gamma/gamma_sym always (with the P budget).
"""
import sys, itertools, numpy as np
sys.path.insert(0, '..')
from common import table, gaps, cost2, cover_matrix, programme, gamma_sym

_VCACHE = {}
def vertex_systems(p):
    """all (S, T) with A_TS invertible: list of (S, T, inverse)."""
    if p in _VCACHE:
        return _VCACHE[p]
    A = cover_matrix(p); h = A.shape[0]; out = []
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            for T in itertools.combinations(range(h), K):
                M = A[np.ix_(T, S)]
                if abs(np.linalg.det(M)) > 1e-12:
                    out.append((np.array(S), np.array(T), np.linalg.inv(M)))
    _VCACHE[p] = out
    return out

def need(p, M, return_vertex=False):
    """exact min sum y^{2/3} over {y >= 0, A y >= M} (concave min -> vertex)."""
    A = cover_matrix(p); h = A.shape[0]
    if (M <= 0).all():
        return (0.0, None) if return_vertex else 0.0
    best, arg = np.inf, None
    tol = 1e-10 * max(1.0, np.abs(M).max())
    for S, T, Minv in vertex_systems(p):
        yS = Minv @ M[T]
        if (yS < -tol).any():
            continue
        y = np.zeros(h); y[S] = np.clip(yS, 0, None)
        if (A @ y < M - tol).any():
            continue
        c = (y[y > 0] ** (2 / 3)).sum()
        if c < best:
            best, arg = c, (S, T, y)
    return (best, arg) if return_vertex else best

def margins(p, z):
    g = gaps(z); h = (p - 1) // 2; i2 = pow(2, -1, p)
    a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
    mu = np.zeros(h); ms = np.zeros(h); mf = np.zeros(h)
    for D in range(1, h + 1):
        gD, gm = g[:, :, D], g[:, :, p - D]
        mu[D - 1] = min(gD.min(), gm.min())
        ms[D - 1] = (0.5 * (gD + gm)).min()
        sh = (D * i2) % p
        mf[D - 1] = (0.5 * (gD[(a + sh) % p, b] + gm[(a - sh) % p, b])).min()
    return dict(u=mu, s=ms, f=mf, margin=g[:, :, 1:].min())

def budgets(p, net):
    h = (p - 1) // 2; cost = np.zeros(h)
    for e1, e2, w in net:
        E1, E2, W = np.fft.fft(e1), np.fft.fft(e2), np.fft.fft(w)
        for k in range(1, h + 1):
            cost[k - 1] += abs(E1[k] * E2[k] * W[k]) ** (2 / 3)
    return dict(diag=cost.sum() / 3 ** (1 / 3), P=(p / 6) * cost2(net) / 3 ** (1 / 3), cost=cost)

def evaluate(p, net):
    z = table(p, net); mg = margins(p, z); bd = budgets(p, net)
    Gam = programme(p, 1.5)[0]
    out = dict(T=mg['margin'] / cost2(net) ** 1.5 / gamma_sym(p, Gam), margin=mg['margin'])
    for mk in ('u', 's', 'f'):
        nd = need(p, mg[mk] * p ** 3 / 2)
        for bk in ('diag', 'P'):
            out[f'{mk}_{bk}'] = (nd / bd[bk]) ** 1.5
    out['m'] = mg
    return out

def unpack(v, N, p):
    v = np.asarray(v).reshape(3, N, p)
    return [(v[0, i], v[1, i], v[2, i]) for i in range(N)]
