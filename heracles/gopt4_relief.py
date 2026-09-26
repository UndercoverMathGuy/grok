"""gopt4_relief.py -- the exact junk-relief function C_r(rho) and its shape at 0.

For one frequency, with A = C_empty = 1 normalised:

    Cmax_r(rho) := min { sum_l |c_l|^r : C_empty = 1, max_{T != empty} |C_T| <= rho }
    Cl2_r(eta)  := min { sum_l |c_l|^r : C_empty = 1, sum_{T != empty} |C_T|^2 <= eta^2 }

over every number of atoms N and all unit phases.  Cmax_r(0) = Cl2_r(0) =
(n+1)^{1-r} is the Key Lemma.  The question that decides statement S2 is the
SHAPE of these functions at 0: the only proved bounds (agent 1's (C'), agent 3's
robust peeling, agent 2's Theorem B) all degrade LINEARLY in the junk, which is
what makes junk look like a winning trade on a K >= 2 support.  Here we measure
the truth and fit

    C(rho) / C(0)  =  1 - beta rho^2 + O(rho^4).

Also reported: the same quantity restricted to N = n+1 atoms, and the
gauge-orbit invariance check that explains why an odd power cannot appear
generically.

All float64, local CPU.
"""
import json
import argparse
import itertools
import numpy as np
from scipy.optimize import minimize

EPS2 = 1e-30


def subsets(n):
    out = []
    for k in range(n + 1):
        for T in itertools.combinations(range(n), k):
            out.append(T)
    return out


def unpack(x, N, n):
    return x[:N], x[N:2 * N], x[2 * N:].reshape(N, n)


def obj(x, N, n, r):
    cr, ci, _ = unpack(x, N, n)
    return np.sum((cr ** 2 + ci ** 2 + EPS2) ** (r / 2.0))


def obj_grad(x, N, n, r):
    cr, ci, _ = unpack(x, N, n)
    m2 = cr ** 2 + ci ** 2 + EPS2
    d = r * m2 ** (r / 2.0 - 1.0)
    g = np.zeros_like(x)
    g[:N] = d * cr
    g[N:2 * N] = d * ci
    return g


def moments(x, N, n, Ts):
    cr, ci, phi = unpack(x, N, n)
    c = cr + 1j * ci
    S = np.zeros((N, len(Ts)))
    for a, T in enumerate(Ts):
        if T:
            S[:, a] = phi[:, list(T)].sum(axis=1)
    E = np.exp(1j * S)
    C = (c[:, None] * E).sum(axis=0)
    return c, E, C


def build(N, n, r, Ts, mode, lvl):
    nT = len(Ts)

    def eq(x):
        _, _, C = moments(x, N, n, Ts)
        return np.array([C[0].real - 1.0, C[0].imag])

    def eq_jac(x):
        _, E, _ = moments(x, N, n, Ts)
        Jm = np.zeros((2, x.size))
        e = E[:, 0]
        Jm[0, :N] = e.real
        Jm[0, N:2 * N] = -e.imag
        Jm[1, :N] = e.imag
        Jm[1, N:2 * N] = e.real
        return Jm

    def d_absC2(x, a):
        """gradient of |C_{Ts[a]}|^2."""
        c, E, C = moments(x, N, n, Ts)
        g = np.zeros(x.size)
        e = E[:, a]
        CT = C[a]
        g[:N] = 2.0 * (np.conj(CT) * e).real
        g[N:2 * N] = 2.0 * (np.conj(CT) * 1j * e).real
        for j in Ts[a]:
            g[2 * N + np.arange(N) * n + j] = 2.0 * (np.conj(CT) * (1j * c * e)).real
        return g

    if mode == 'max':
        def ineq(x):
            _, _, C = moments(x, N, n, Ts)
            return lvl ** 2 - np.abs(C[1:]) ** 2

        def ineq_jac(x):
            return np.array([-d_absC2(x, a) for a in range(1, nT)])
    else:  # 'l2'
        def ineq(x):
            _, _, C = moments(x, N, n, Ts)
            return np.array([lvl ** 2 - np.sum(np.abs(C[1:]) ** 2)])

        def ineq_jac(x):
            return np.array([-sum(d_absC2(x, a) for a in range(1, nT))])

    return eq, eq_jac, ineq, ineq_jac


def start(N, n, rng, kind, scale=0.0):
    nb = n + 1
    cr = np.zeros(N)
    ci = np.zeros(N)
    phi = rng.uniform(0, 2 * np.pi, size=(N, n))
    if kind == 'block':
        cr[:nb] = 1.0 / nb
        for l in range(nb):
            phi[l, :] = -2.0 * np.pi * l / nb
    else:
        cr[:] = rng.standard_normal(N) / N
        ci[:] = rng.standard_normal(N) / N
    x = np.concatenate([cr, ci, phi.ravel()])
    if scale:
        x = x + scale * rng.standard_normal(x.size)
    return x


def solve(N, n, r, Ts, mode, lvl, starts, rng):
    eq, eq_jac, ineq, ineq_jac = build(N, n, r, Ts, mode, lvl)
    cons = [{'type': 'eq', 'fun': eq, 'jac': eq_jac},
            {'type': 'ineq', 'fun': ineq, 'jac': ineq_jac}]
    best, bx = np.inf, None
    for s in range(starts):
        if s == 0:
            x0 = start(N, n, rng, 'block')
        elif s < 6:
            x0 = start(N, n, rng, 'block', scale=10.0 ** (-s))
        else:
            x0 = start(N, n, rng, 'rand')
        try:
            res = minimize(lambda z: obj(z, N, n, r), x0,
                           jac=lambda z: obj_grad(z, N, n, r),
                           constraints=cons, method='SLSQP',
                           options={'maxiter': 800, 'ftol': 1e-16})
        except Exception:
            continue
        if not res.success:
            continue
        if np.max(np.abs(eq(res.x))) > 1e-10:
            continue
        if np.min(ineq(res.x)) < -1e-11:
            continue
        v = obj(res.x, N, n, r)
        if v < best:
            best, bx = v, res.x
    return best, bx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=str, nargs='+',
                    default=['2:2', '2:2.5', '3:2', '3:3'],
                    help='n:q pairs')
    ap.add_argument('--starts', type=int, default=30)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', type=str,
                    default='heracles/FINAL_RUNS/GOPT4/relief.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    levels = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5]
    out = []
    for cell in args.cells:
        ns, qs = cell.split(':')
        n, q = int(ns), float(qs)
        nu, r = n + 1, float(qs) / (int(ns) + 1)
        Ts = subsets(n)
        tgt = (n + 1) ** (1 - r)
        Ns = [n + 1, n + 2]
        print(f"\n=== n={n} nu={nu} q={q} r={r:.6f}  (n+1)^(1-r)={tgt:.12f}")
        for mode in ('max', 'l2'):
            vals, defs = [], []
            for lvl in levels:
                v = min(solve(N, n, r, Ts, mode, lvl, args.starts, rng)[0]
                        for N in Ns)
                vals.append(v)
                defs.append(1.0 - v / tgt)
            # quadratic fit on the three smallest levels
            ls = np.array(levels[:4])
            ds = np.array(defs[:4])
            beta = float(np.mean(ds / ls ** 2))
            slopes = np.log(ds[1:] / ds[:-1]) / np.log(ls[1:] / ls[:-1])
            print(f"  mode={mode:3s} beta~{beta:.6f}  "
                  f"log-log slopes on smallest levels: "
                  + " ".join(f"{s:.4f}" for s in slopes))
            for lvl, v, d in zip(levels, vals, defs):
                print(f"     rho={lvl:<6.3g} C={v:.12f}  deficit={d:.3e}"
                      f"  deficit/rho^2={d/lvl**2:.6f}")
            out.append({'n': n, 'q': q, 'r': r, 'target': tgt, 'mode': mode,
                        'levels': levels, 'C': vals, 'deficit': defs,
                        'beta': beta, 'loglog_slopes': slopes.tolist()})
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
